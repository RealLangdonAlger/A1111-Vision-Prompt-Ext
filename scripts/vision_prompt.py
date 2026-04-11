import base64
import requests
from PIL import Image
import io
import json
import os

import gradio as gr

from modules import scripts
from modules.processing import StableDiffusionProcessing

import hashlib

VISION_CACHE = [{} for _ in range(4)]  # one dict per slot, max 5 entries each

# How many image drop zones to show per slot
IMAGES_PER_SLOT = 4

# Seconds to wait for a vision API response before giving up
TIMEOUT_SECONDS = 30

def load_presets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preset_path = os.path.join(base_dir, "..", "presets.json")

    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Vision Prompt] Failed to load presets: {e}")
        return {"Custom": ""}

SYSTEM_PROMPT_PRESETS = load_presets()


def stitch_images_horizontal(images: list, gap: int = 0) -> Image.Image:
    """
    Stitch a list of PIL images side-by-side with an optional gap between them.
    All images are resized to the same height (the tallest one) before stitching.
    """
    if len(images) == 1:
        return images[0]

    target_height = max(img.height for img in images)

    resized = []
    for img in images:
        if img.height != target_height:
            scale = target_height / img.height
            new_w = int(img.width * scale)
            img = img.resize((new_w, target_height), Image.LANCZOS)
        resized.append(img)

    total_width = sum(img.width for img in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_width, target_height), (255, 255, 255))

    x_offset = 0
    for img in resized:
        canvas.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    return canvas


class VisionPromptScript(scripts.Script):

    def title(self):
        return "Vision Prompt Injector"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Vision Prompt", open=False):
            enabled = gr.Checkbox(label="Enable Vision Prompt", value=False)

            api_url = gr.Textbox(
                label="API URL",
                value="http://localhost:8000/v1/chat/completions"
            )
            
            api_key = gr.Textbox(
                label="API Key",
                value="",
                placeholder="sk-... (leave blank for local endpoints)",
                type="password"
            )

            model_name = gr.Textbox(
                label="Model",
                value="gpt-4o-mini"
            )

            tabs_data = []

            with gr.Tabs():
                for i in range(4):
                    with gr.Tab(f"Slot {i+1}"):

                        # ── Multiple image drop zones ──────────────────────
                        gr.Markdown(
                            f"Upload up to **{IMAGES_PER_SLOT} images** — they will be "
                            "stitched side-by-side before being sent to the vision model."
                        )

                        image_inputs = []
                        with gr.Row():
                            for j in range(IMAGES_PER_SLOT):
                                img_input = gr.Image(
                                    type="pil",
                                    label=f"Image {j+1}",
                                    height=192,
                                )
                                image_inputs.append(img_input)

                        # ── Preset / system prompt ─────────────────────────
                        preset_dropdown = gr.Dropdown(
                            label="Preset",
                            choices=list(SYSTEM_PROMPT_PRESETS.keys()),
                            value="Custom"
                        )

                        def apply_preset(choice):
                            return SYSTEM_PROMPT_PRESETS.get(choice, "")

                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value="Describe this image in detail for Stable Diffusion prompting.",
                            height=64
                        )

                        preset_dropdown.change(
                            fn=apply_preset,
                            inputs=[preset_dropdown],
                            outputs=[system_prompt]
                        )

                        weight = gr.Slider(
                            label="Weight",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05
                        )

                        # Pack: [img0, img1, img2, img3, system_prompt, weight]
                        tabs_data.extend(image_inputs)
                        tabs_data.extend([system_prompt, weight])

        return [enabled, api_url, model_name, api_key] + tabs_data

    def process(self, p, enabled, api_url, model_name, api_key, *args):
        if not enabled:
            return

        combined_prompts = []

        # args layout per slot: [img0, img1, img2, img3, system_prompt, weight]
        stride = IMAGES_PER_SLOT + 2  # +2 for system_prompt and weight

        for slot_idx in range(4):
            base = slot_idx * stride

            slot_images   = [args[base + j] for j in range(IMAGES_PER_SLOT)]
            system_prompt = args[base + IMAGES_PER_SLOT]
            weight        = args[base + IMAGES_PER_SLOT + 1]

            # Drop empty upload zones
            valid_images = [img for img in slot_images if img is not None]

            if not valid_images:
                continue

            # ── Stitch images together ─────────────────────────────────────
            if len(valid_images) == 1:
                composed = valid_images[0]
                stitch_desc = "1 image"
            else:
                composed = stitch_images_horizontal(valid_images)
                stitch_desc = f"{len(valid_images)} images stitched"

            print(f"\n[Vision Prompt][Slot {slot_idx+1}] {stitch_desc}")

            # ── Resize + compress before encoding ─────────────────────────
            # Most vision APIs cap payloads at a few MB. Downscale to fit,
            # then encode as JPEG (much smaller than PNG for photos).
            MAX_LONG_EDGE = 4096  # matches OpenAI's "low detail" threshold
            MAX_BYTES     = 4 * 1024 * 1024  # 4 MB hard cap

            send_img = composed.copy()
            if max(send_img.size) > MAX_LONG_EDGE:
                send_img.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.LANCZOS)

            # Try JPEG at decreasing quality until under the byte cap
            jpeg_quality = 90
            while True:
                buffered = io.BytesIO()
                send_img.convert("RGB").save(buffered, format="JPEG", quality=jpeg_quality)
                img_bytes = buffered.getvalue()
                if len(img_bytes) <= MAX_BYTES or jpeg_quality <= 30:
                    break
                jpeg_quality -= 10

            print(
                f"[Vision Prompt][Slot {slot_idx+1}] "
                f"Sending {send_img.size[0]}×{send_img.size[1]}px JPEG "
                f"@ quality={jpeg_quality} ({len(img_bytes)//1024} KB)"
            )

            # ── Cache key: hash all source images + prompt + model ─────────
            hash_parts = b"".join(self._pil_to_bytes(img) for img in valid_images)
            hash_input = hash_parts + system_prompt.encode("utf-8") + model_name.encode("utf-8")
            cache_key  = hashlib.sha256(hash_input).hexdigest()

            slot_cache = VISION_CACHE[slot_idx]

            if cache_key in slot_cache:
                vision_output = slot_cache[cache_key]
                print(f"[Vision Prompt][Slot {slot_idx+1}] Cache HIT")
            else:
                print(f"[Vision Prompt][Slot {slot_idx+1}] Cache MISS — calling API")

                img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": ""},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 512
                }

                try:
                    headers = {"Content-Type": "application/json"}
                    if api_key and api_key.strip():
                        headers["Authorization"] = f"Bearer {api_key.strip()}"
                    response = requests.post(api_url, json=payload, headers=headers, timeout=TIMEOUT_SECONDS)
                    response.raise_for_status()
                    data = response.json()
                    vision_output = data["choices"][0]["message"]["content"]

                    # Evict oldest entry if at capacity
                    if len(slot_cache) >= 5:
                        oldest_key = next(iter(slot_cache))
                        del slot_cache[oldest_key]
                    slot_cache[cache_key] = vision_output
                except requests.exceptions.Timeout:
                    msg = f"[Vision Prompt] Slot {slot_idx+1}: Request timed out after {TIMEOUT_SECONDS}s"
                    print(msg)
                    continue
                except Exception as e:
                    msg = f"[Vision Prompt] Slot {slot_idx+1}: API error: {e}"
                    print(msg)
                    continue

            if not vision_output:
                continue

            vision_output   = vision_output.strip().replace("\n\n", "\n").replace(":", "\\:").replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]")
            weighted_prompt = f"({vision_output}:{weight})"

            print(f"[Vision Prompt][Slot {slot_idx+1}] {weighted_prompt}")
            combined_prompts.append(weighted_prompt)

        # ── Merge all slot outputs into the final prompt ───────────────────
        if not combined_prompts:
            return

        final_injection = "\n\n".join(combined_prompts)
        print(f"[Vision Prompt] Final Injection:\n{final_injection}")

        p.prompt = f"{final_injection}, {p.prompt}"

        if hasattr(p, "all_prompts") and p.all_prompts:
            p.all_prompts = [
                f"{prompt}\n\n{final_injection}" for prompt in p.all_prompts
            ]

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _pil_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()