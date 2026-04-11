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

VISION_CACHE = {}

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

            model_name = gr.Textbox(
                label="Model",
                value="gpt-4o-mini"
            )

            tabs_data = []

            with gr.Tabs():
                for i in range(4):
                    with gr.Tab(f"Slot {i+1}"):

                        image_input = gr.Image(
                            type="pil",
                            label="Input Image",
                            height=256
                        )

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

                        tabs_data.extend([image_input, system_prompt, weight])

        return [enabled, api_url, model_name] + tabs_data

    def process(self, p, enabled, api_url, model_name, *args):
        if not enabled:
            return

        import base64, io, requests

        combined_prompts = []

        # args = [img1, sys1, w1, img2, sys2, w2, ...]
        for i in range(0, len(args), 3):
            image_input = args[i]
            system_prompt = args[i + 1]
            weight = args[i + 2]

            if image_input is None:
                continue

            # Convert image → bytes
            buffered = io.BytesIO()
            image_input.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            # Build cache key
            hash_input = img_bytes + system_prompt.encode("utf-8") + model_name.encode("utf-8")
            cache_key = hashlib.sha256(hash_input).hexdigest()
            
            if cache_key in VISION_CACHE:
                vision_output = VISION_CACHE[cache_key]
                print(f"\n[Vision Prompt][Slot {i//3+1}] Cache HIT")
            else:
                print(f"\n[Vision Prompt][Slot {i//3+1}] Cache MISS")

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
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 512
                }

                try:
                    response = requests.post(api_url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    vision_output = data["choices"][0]["message"]["content"]

                    # Store in cache
                    VISION_CACHE[cache_key] = vision_output

                except Exception as e:
                    print(f"\n[Vision Prompt][Slot {i//3+1}] API error: {e}")
                    continue

            if not vision_output:
                continue

            vision_output = vision_output.strip().replace("\n", " ")

            weighted_prompt = f"({vision_output}:{weight})"

            print(f"\n[Vision Prompt][Slot {i//3+1}] {weighted_prompt}")

            combined_prompts.append(weighted_prompt)

        # 🔥 Merge all outputs
        if not combined_prompts:
            return

        final_injection = "\n\n".join(combined_prompts)

        print(f"[Vision Prompt] Final Injection:\n{final_injection}")

        # Inject into prompt
        p.prompt = f"{final_injection}, {p.prompt}"

        if hasattr(p, "all_prompts") and p.all_prompts:
            p.all_prompts = [
                f"{prompt}\n\n{final_injection}" for prompt in p.all_prompts
            ]