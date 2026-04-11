import base64
import requests
from PIL import Image
import io

import gradio as gr

from modules import scripts
from modules.processing import StableDiffusionProcessing

class VisionPromptScript(scripts.Script):

    def title(self):
        return "Vision Prompt Injector"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Vision Prompt", open=False):
            enabled = gr.Checkbox(label="Enable Vision Prompt", value=False)

            image_input = gr.Image(type="pil", label="Input Image", height=256)
            
            weight = gr.Slider(
                label="Vision Prompt Weight",
                minimum=0.0,
                maximum=2.0,
                value=1.0,
                step=0.05
            )

            api_url = gr.Textbox(
                label="API URL",
                value="http://localhost:8000/v1/chat/completions"
            )

            system_prompt = gr.Textbox(
                label="System Prompt",
                value="Describe this image in detail for Stable Diffusion prompting."
            )

            model_name = gr.Textbox(
                label="Model",
                value="gpt-4o-mini"
            )

        return [enabled, image_input, api_url, system_prompt, model_name, weight]

    def process(self, p, enabled, image_input, api_url, system_prompt, model_name, weight):

        if not enabled or image_input is None:
            return

        import base64, io, requests

        # Convert image → base64
        buffered = io.BytesIO()
        image_input.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

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

        except Exception as e:
            print(f"[Vision Prompt] API error: {e}")
            return

        if not vision_output:
            return

        # Clean up whitespace a bit
        vision_output = vision_output.strip().replace("\n", " ")

        # Apply weight formatting
        weighted_prompt = f"({vision_output}:{weight})"

        print(f"[Vision Prompt] Injecting: {weighted_prompt}")

        # 🔥 CRITICAL PART — update ALL prompt representations

        # Base prompt
        p.prompt = f"{weighted_prompt}, {p.prompt}"

        # Batched prompts (critical)
        if hasattr(p, "all_prompts") and p.all_prompts:
            p.all_prompts = [
                f"{prompt}\n\n{weighted_prompt}" for prompt in p.all_prompts
            ]

        # Negative prompts (optional, but keeps things consistent)
        if hasattr(p, "all_negative_prompts") and p.all_negative_prompts:
            p.all_negative_prompts = list(p.all_negative_prompts)