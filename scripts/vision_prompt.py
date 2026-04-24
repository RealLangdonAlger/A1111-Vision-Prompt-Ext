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

VISION_CACHE = [{} for _ in range(3)]  # one dict per slot

# How many image drop zones to show per slot
IMAGES_PER_SLOT = 3

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

def _write_presets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preset_path = os.path.join(base_dir, "..", "presets.json")
    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(SYSTEM_PROMPT_PRESETS, f, indent=2, ensure_ascii=False)

def save_preset(name: str, prompt: str):
    if not name.strip():
        raise Exception("[Vision Prompt] Preset name cannot be empty.")
    SYSTEM_PROMPT_PRESETS[name.strip()] = prompt
    _write_presets()
    choices = list(SYSTEM_PROMPT_PRESETS.keys())
    return gr.update(choices=choices, value=name.strip()), name.strip()

def delete_preset(name: str):
    if name == "Custom":
        raise Exception("[Vision Prompt] Cannot delete the Custom preset.")
    SYSTEM_PROMPT_PRESETS.pop(name, None)
    _write_presets()
    choices = list(SYSTEM_PROMPT_PRESETS.keys())
    return gr.update(choices=choices, value="Custom"), "Custom"

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

# ── Shared API call helper ─────────────────────────────────────────────────────

def _encode_image(img: Image.Image, max_long_edge: int = 4096, max_bytes: int = 4 * 1024 * 1024):
    """Resize + JPEG-compress a PIL image and return (base64_str, width, height, quality, byte_len)."""
    send_img = img.copy()
    if max(send_img.size) > max_long_edge:
        send_img.thumbnail((max_long_edge, max_long_edge), Image.LANCZOS)

    jpeg_quality = 90
    while True:
        buffered = io.BytesIO()
        send_img.convert("RGB").save(buffered, format="JPEG", quality=jpeg_quality)
        img_bytes = buffered.getvalue()
        if len(img_bytes) <= max_bytes or jpeg_quality <= 30:
            break
        jpeg_quality -= 10

    return base64.b64encode(img_bytes).decode("utf-8"), send_img.size[0], send_img.size[1], jpeg_quality, len(img_bytes)

def _call_vision_api(
    img_b64: str,
    system_prompt: str,
    model_name: str,
    api_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    reasoning_budget: str,
    timeout_seconds: int,
    user_text: str = "",
) -> str | None:
    """
    Send one image (already base64-encoded) to the vision API.
    Returns the raw string content or None on failure.
    """
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
            }
        ],
    }
    if top_k > 0:
        payload["top_k"] = top_k

    _reasoning_budget_map = {
        "none":   0,
        "low":    int(max_tokens * 0.2),
        "medium": int(max_tokens * 0.5),
        "high":   int(max_tokens * 0.8),
    }
    if reasoning_budget in _reasoning_budget_map:
        payload["reasoning_effort"] = reasoning_budget
        payload["thinking_budget_tokens"] = _reasoning_budget_map[reasoning_budget]

    try:
        headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print(f"[Vision Prompt] Request timed out after {timeout_seconds}s")
        return None
    except Exception as e:
        print(f"[Vision Prompt] API error: {e}")
        return None

def _call_text_api(
    system_prompt: str,
    user_text: str,
    model_name: str,
    api_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    reasoning_budget: str,
    timeout_seconds: int,
) -> str | None:
    """
    Text-only API call used for the Merge Multicall merge step.
    """
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
    }
    if top_k > 0:
        payload["top_k"] = top_k

    _reasoning_budget_map = {
        "none":   0,
        "low":    int(max_tokens * 0.2),
        "medium": int(max_tokens * 0.5),
        "high":   int(max_tokens * 0.8),
    }
    if reasoning_budget in _reasoning_budget_map:
        payload["reasoning_effort"] = reasoning_budget
        payload["thinking_budget_tokens"] = _reasoning_budget_map[reasoning_budget]

    try:
        headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        response = requests.post(api_url, json=payload, headers=headers, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        print(f"[Vision Prompt] Merge call timed out after {timeout_seconds}s")
        return None
    except Exception as e:
        print(f"[Vision Prompt] Merge API error: {e}")
        return None

class VisionPromptScript(scripts.Script):

    def __init__(self):
        self.infotext_fields = []
        self.paste_field_names = []

    def title(self):
        return "Vision Prompt Injector"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        gr.HTML("""
            <style>
            .vp-params-box {
                border: 1px solid var(--block-border-color);
                border-radius: 8px;
                padding: 10px;
                margin-top: 8px;
                margin-bottom: 4px;
            }
            .vp-image-box {
                border: 1px solid var(--block-border-color);
                border-radius: 8px;
                padding: 4px;
                margin: 0px;
            }
            .bottom-align {
                display: flex !important;
                flex-direction: column !important;
                margin-top: 8px !important;
                vertical-align: top !important;
            }
            </style>
            """)
        with gr.Accordion("Vision Prompt", open=False):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable Vision Prompt",
                    value=False)

            with gr.Row():
                model_name = gr.Textbox(
                    label="Model",
                    value="gpt-4o-mini",
                    scale=2
                )

                api_url = gr.Textbox(
                    label="API URL",
                    value="http://localhost:8000/v1/chat/completions",
                    scale=3
                )
                api_key = gr.Textbox(
                    label="API Key",
                    value="",
                    placeholder="sk-... (leave blank for local endpoints)",
                    type="password",
                    scale=2
                )

            with gr.Accordion("Advanced Settings", open=False, elem_classes="vp-params-box"):
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.05
                    )

                    top_p = gr.Slider(
                        label="Top-p",
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.05
                    )

                    top_k = gr.Slider(
                        label="Top-k",
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=1
                    )

                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=16,
                        maximum=16384,
                        value=512,
                        step=16
                    )

                with gr.Row():
                    timeout_seconds = gr.Slider(
                        label="Request Timeout (seconds)",
                        minimum=5,
                        maximum=120,
                        value=30,
                        step=1
                    )

                    cache_size = gr.Slider(
                        label="Cache Size (per slot)",
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1
                    )

                with gr.Row(elem_id="reasoning-row"):
                    always_regenerate = gr.Checkbox(
                        label="Skip Cache",
                        value=False,
                        info="Always do a fresh API call.",
                        scale=1,
                        elem_classes=["bottom-align"]
                    )

                    write_png_meta = gr.Checkbox(
                        label="Store Infotext",
                        value=False,
                        info="Writes settings and system prompts to the PNG metadata.",
                        scale=2,
                        elem_classes=["bottom-align"]
                    )

                    reasoning_budget = gr.Radio(
                        label="Model Reasoning", show_label=False,
                        choices=["none", "low", "medium", "high"],
                        value="none",
                        info=("Reasoning mode for models that support it. Needs Max Tokens >2048"),
                        scale=2,
                        elem_classes=["bottom-align"]
                    )

            tabs_data = []

            with gr.Tabs():
                for i in range(3):
                    with gr.Tab(f"Vision Input #{i+1}") as tab:
                        with gr.Row():
                            slot_enabled = gr.Checkbox(
                                label="Enable slot",
                                value=True,
                                scale=0,
                                min_width=120
                            )

                        # ── Multi-image mode selector ──────────────────────────
                        with gr.Row():
                            multi_image_mode = gr.Radio(
                                label="Multi-Image Mode", show_label=False,
                                choices=["Stitch", "Merge Multicall"],
                                value="Stitch",
                                info=("Multi-Image Modes: "
                                    "[Stitch] Fast and cheap. Combine images side-by-side in one call. "
                                    "[Merge] Better quality, but much slower. One call per image, then a final call merges results."
                                ),
                            )

                        gr.Markdown(
                            f"Upload up to **{IMAGES_PER_SLOT} images**"
                        )

                        image_inputs    = []

                        with gr.Row(elem_classes="vp-image-box"):
                            for j in range(IMAGES_PER_SLOT):
                                with gr.Column(min_width=120):
                                    img_input = gr.Image(
                                        type="pil",
                                        label=f"Image {j+1}",
                                        height=256,
                                        elem_classes="vp-image-box",
                                    )

                                image_inputs.append(img_input)

                        # ── Preset / system prompt ─────────────────────────
                        gr.Markdown("### Prompt Preset")
                        with gr.Row(equal_height=True):
                            preset_dropdown = gr.Dropdown(
                                choices=list(SYSTEM_PROMPT_PRESETS.keys()),
                                value="Custom",
                                scale=3,
                                show_label=False
                            )
                            preset_name_input = gr.Textbox(
                                placeholder="Preset name...",
                                show_label=False,
                                scale=2,
                                max_lines=1
                            )

                            save_btn = gr.Button("💾", scale=0, min_width=40)
                            delete_btn = gr.Button("🗑️", scale=0, min_width=40)

                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value="Describe this image in detail for Stable Diffusion prompting."
                        )

                        def apply_preset(choice):
                            return SYSTEM_PROMPT_PRESETS.get(choice, ""), choice

                        preset_dropdown.change(
                            fn=apply_preset,
                            inputs=[preset_dropdown],
                            outputs=[system_prompt, preset_name_input]
                        )
                        save_btn.click(
                            fn=save_preset,
                            inputs=[preset_name_input, system_prompt],
                            outputs=[preset_dropdown, preset_name_input]
                        )
                        delete_btn.click(
                            fn=delete_preset,
                            inputs=[preset_dropdown],
                            outputs=[preset_dropdown, preset_name_input]
                        )

                        weight = gr.Slider(
                            label="Prompt Weight",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05
                        )

                        # Pack
                        tabs_data.append(slot_enabled)
                        tabs_data.append(multi_image_mode)
                        tabs_data.extend(image_inputs)
                        tabs_data.extend([system_prompt, weight])

        # ── Persistent settings & PNG metadata ────────────────────────────
        self.infotext_fields = [
            (enabled,        "VP Enabled"),
            (api_url,        "VP API URL"),
            (model_name,     "VP Model"),
            (temperature,    "VP Temperature"),
            (top_p,          "VP Top-p"),
            (top_k,          "VP Top-k"),
            (max_tokens,     "VP Max Tokens"),
        ]

        # Slot layout: [slot_enabled, multi_image_mode, img0..N, system_prompt, weight]
        SLOT_STRIDE       = 1 + 1 + IMAGES_PER_SLOT * 1 + 2
        SLOT_ENABLED_OFF  = 0
        MULTI_MODE_OFF    = 1
        SYSTEM_PROMPT_OFF = 2 + IMAGES_PER_SLOT * 1
        WEIGHT_OFF        = 2 + IMAGES_PER_SLOT * 1 + 1

        for i in range(3):
            base = i * SLOT_STRIDE
            self.infotext_fields += [
                (tabs_data[base + SLOT_ENABLED_OFF],  f"VP Slot {i+1} Enabled"),
                (tabs_data[base + MULTI_MODE_OFF],    f"VP Slot {i+1} Multi Mode"),
                (tabs_data[base + SYSTEM_PROMPT_OFF], f"VP Slot {i+1} System Prompt"),
                (tabs_data[base + WEIGHT_OFF],        f"VP Slot {i+1} Weight"),
            ]

        self.paste_field_names = [label for _, label in self.infotext_fields]

        return [
            enabled, api_url, model_name, api_key,
            temperature, top_p, top_k, max_tokens,
            timeout_seconds, cache_size,
            always_regenerate, write_png_meta,
            reasoning_budget,
        ] + tabs_data

    def process(self, p, enabled, api_url, model_name, api_key, temperature, top_p, top_k, max_tokens, timeout_seconds, cache_size, always_regenerate, write_png_meta, reasoning_budget, *args):
        if not enabled:
            return

        # ── Write settings to PNG metadata ────────────────────────────────
        if write_png_meta:
            p.extra_generation_params.update({
                "VP Enabled":     enabled,
                "VP API URL":     api_url,
                "VP Model":       model_name,
                "VP Temperature": temperature,
                "VP Top-p":       top_p,
                "VP Top-k":       top_k,
                "VP Max Tokens":  max_tokens,
            })

        # Slot layout: [slot_enabled, multi_image_mode, img0..N, system_prompt, weight]
        SLOT_STRIDE       = 1 + 1 + IMAGES_PER_SLOT * 1 + 2
        SLOT_ENABLED_OFF  = 0
        MULTI_MODE_OFF    = 1
        IMG_OFF           = 2
        SYSTEM_PROMPT_OFF = 2 + IMAGES_PER_SLOT * 1
        WEIGHT_OFF        = 2 + IMAGES_PER_SLOT * 1 + 1

        for i in range(3):
            base = i * SLOT_STRIDE
            if write_png_meta:
                p.extra_generation_params[f"VP Slot {i+1} Enabled"]    = args[base + SLOT_ENABLED_OFF]
                p.extra_generation_params[f"VP Slot {i+1} Multi Mode"] = args[base + MULTI_MODE_OFF]
                p.extra_generation_params[f"VP Slot {i+1} System Prompt"] = args[base + SYSTEM_PROMPT_OFF]
                p.extra_generation_params[f"VP Slot {i+1} Weight"]     = args[base + WEIGHT_OFF]

        combined_prompts = []

        # Shared kwargs for all API helpers
        api_kwargs = dict(
            model_name=model_name,
            api_url=api_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reasoning_budget=reasoning_budget,
            timeout_seconds=timeout_seconds,
        )

        for slot_idx in range(3):
            base = slot_idx * SLOT_STRIDE

            slot_enabled = args[base + SLOT_ENABLED_OFF]
            if not slot_enabled:
                continue

            multi_image_mode = args[base + MULTI_MODE_OFF]   # "Stitch" | "Inline Multicall" | "Merge Multicall"
            slot_images      = [args[base + IMG_OFF  + j]    for j in range(IMAGES_PER_SLOT)]
            system_prompt    = args[base + SYSTEM_PROMPT_OFF]
            weight           = args[base + WEIGHT_OFF]

            # Replace {prompt} and {negative_prompt} placeholders with the main A1111 prompts
            if "{prompt}" in system_prompt:
                system_prompt = system_prompt.replace("{prompt}", p.prompt)
            if "{negative_prompt}" in system_prompt:
                neg = p.negative_prompt if hasattr(p, "negative_prompt") else ""
                system_prompt = system_prompt.replace("{negative_prompt}", neg)

            valid_images = []
            for img in slot_images:
                if img is not None:
                    valid_images.append(img)

            if not valid_images:
                continue

            # ── Route to the chosen strategy ──────────────────────────────
            if multi_image_mode == "Stitch" or len(valid_images) == 1:
                vision_output = self._run_stitch(
                    valid_images, slot_idx, system_prompt,
                    always_regenerate, cache_size,
                    **api_kwargs
                )
            else:  # "Merge Multicall"
                vision_output = self._run_merge_multicall(
                    valid_images, slot_idx, system_prompt,
                    always_regenerate, cache_size,
                    **api_kwargs
                )

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

        p.prompt = f"{p.prompt}\n\n{final_injection}"

        if hasattr(p, "all_prompts") and p.all_prompts:
            p.all_prompts = [
                f"{prompt}\n\n{final_injection}" for prompt in p.all_prompts
            ]

    # ── Strategy: Stitch ──────────────────────────────────────────────────────

    def _run_stitch(self, valid_images, slot_idx, system_prompt, always_regenerate, cache_size, **api_kwargs):
        if len(valid_images) == 1:
            composed     = valid_images[0]
            stitch_desc  = "1 image"
        else:
            composed     = stitch_images_horizontal(valid_images)
            stitch_desc  = f"{len(valid_images)} images stitched"

        print(f"\n[Vision Prompt][Slot {slot_idx+1}] Stitch — {stitch_desc}")

        img_b64, w, h, quality, nbytes = _encode_image(composed)
        print(f"[Vision Prompt][Slot {slot_idx+1}] Sending {w}×{h}px JPEG @ quality={quality} ({nbytes//1024} KB)")

        cache_key = self._cache_key_images(
            valid_images, system_prompt, api_kwargs, suffix="stitch"
        )
        return self._cached_call(
            cache_key, slot_idx, always_regenerate, cache_size,
            lambda: _call_vision_api(img_b64, system_prompt, **api_kwargs)
        )

    # ── Strategy: Merge Multicall ─────────────────────────────────────────────

    def _run_merge_multicall(self, valid_images, slot_idx, system_prompt, always_regenerate, cache_size, **api_kwargs):
        """
        One API call per image (no schema constraints), followed by a second text-only
        call that merges the individual descriptions into the schema defined by the
        original system prompt.
        """
        print(f"\n[Vision Prompt][Slot {slot_idx+1}] Merge Multicall — {len(valid_images)} image(s)")

        # ── Step 1: describe each image independently ──────────────────────
        individual_outputs = []
        individual_cache_keys = []

        for j, img in enumerate(valid_images):
            img_b64, w, h, quality, nbytes = _encode_image(img)
            print(f"[Vision Prompt][Slot {slot_idx+1}] Image {j+1}: {w}×{h}px JPEG @ quality={quality} ({nbytes//1024} KB)")

            cache_key = self._cache_key_images(
                [img], system_prompt, api_kwargs, suffix=f"merge_sub_{j}"
            )
            individual_cache_keys.append(cache_key)

            output = self._cached_call(
                cache_key, slot_idx, always_regenerate, cache_size,
                lambda b64=img_b64: _call_vision_api(b64, system_prompt, **api_kwargs)
            )

            if output:
                individual_outputs.append(output.strip())

        if not individual_outputs:
            return None

        # If there is only one result there is nothing to merge
        if len(individual_outputs) == 1:
            return individual_outputs[0]

        # ── Step 2: merge via a text-only call ────────────────────────────
        descriptions_block = "\n".join(
            f"Input #{j+1}: {desc}" for j, desc in enumerate(individual_outputs)
        )

        merge_system = (
            "You are a formatting assistant. "
            "The user will provide inputs, one per line, that have been generated by a previous system prompt. "
            "Your task is to comine them into a single output that makes sense, "
            "preserving all descriptive detail and wording as closely as possible. "
            "Essentially, you pretend to be a vision model. But instead of an image, you get the already processed output.\n\n"
            f"The following system prompt was used for image analysis:\n{system_prompt}"
        )
        merge_user = (
            f"Process the following {len(individual_outputs)} outputs:\n\n{descriptions_block}"
        )

        # Cache key for the merge step: hash of all sub-keys + merge prompts
        merge_hash_input = (
            "".join(individual_cache_keys)
            + merge_system
            + merge_user
        ).encode("utf-8")
        merge_cache_key = hashlib.sha256(merge_hash_input).hexdigest()

        text_kwargs = {k: v for k, v in api_kwargs.items()}

        merged = self._cached_call(
            merge_cache_key, slot_idx, always_regenerate, cache_size,
            lambda: _call_text_api(merge_system, merge_user, **api_kwargs)
        )

        return merged

    # ── Cache helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key_images(images, system_prompt, api_kwargs, suffix=""):
        hash_parts = b"".join(VisionPromptScript._pil_to_bytes(img) for img in images)
        param_str  = (
            f"{api_kwargs.get('temperature')}|{api_kwargs.get('top_p')}|"
            f"{api_kwargs.get('top_k')}|{api_kwargs.get('max_tokens')}|"
            f"{api_kwargs.get('reasoning_budget', '')}"
        )
        raw = (
            hash_parts
            + system_prompt.encode("utf-8")
            + api_kwargs.get("model_name", "").encode("utf-8")
            + api_kwargs.get("api_url", "").encode("utf-8")
            + param_str.encode("utf-8")
            + suffix.encode("utf-8")
        )
        return hashlib.sha256(raw).hexdigest()

    def _cached_call(self, cache_key, slot_idx, always_regenerate, cache_size, call_fn):
        """
        Look up cache_key in the slot cache. On a miss (or bypass), call call_fn()
        and store the result. Returns the cached or freshly-fetched string.
        """
        slot_cache = VISION_CACHE[slot_idx]

        if not always_regenerate and cache_key in slot_cache:
            print(f"[Vision Prompt][Slot {slot_idx+1}] Cache HIT")
            return slot_cache[cache_key]

        if always_regenerate and cache_key in slot_cache:
            print(f"[Vision Prompt][Slot {slot_idx+1}] Cache bypassed (Skip Cache) — calling API")
        else:
            print(f"[Vision Prompt][Slot {slot_idx+1}] Cache MISS — calling API")

        result = call_fn()

        if result:
            if len(slot_cache) >= cache_size:
                oldest_key = next(iter(slot_cache))
                del slot_cache[oldest_key]
            slot_cache[cache_key] = result

        return result

    # ── Misc helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pil_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()