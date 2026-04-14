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

def apply_mask(image: Image.Image, editor_value: dict, mask_mode: str):
    """
    Apply a mask from a gr.ImageEditor value to a plain PIL image.

    mask_mode "exclude"  — painted areas replaced with grey (LLM ignores them)
    mask_mode "include"  — unpainted areas replaced with grey (LLM focuses only on painted)
    mask_mode "none"     — no masking, image returned as-is

    Returns the (possibly modified) RGB image, or None if image is None.
    """
    import numpy as np

    if image is None:
        return None

    if mask_mode == "none" or editor_value is None:
        return image.convert("RGB")

    # Extract drawn strokes from all layers into a single mask
    combined_mask = Image.new("L", image.size, 0)
    for layer in (editor_value.get("layers") or []):
        if layer is None:
            continue
        if not isinstance(layer, Image.Image):
            layer = Image.fromarray(layer)
        layer_rgba = layer.convert("RGBA").resize(image.size, Image.LANCZOS)
        combined_mask = Image.fromarray(
            np.maximum(
                np.array(combined_mask),
                np.array(layer_rgba.split()[3])
            )
        )

    neutral = Image.new("RGB", image.size, (0, 0, 0))
    base    = image.convert("RGB")

    if mask_mode == "exclude":
        result = base.copy()
        result.paste(neutral, mask=combined_mask)
    else:  # include
        inverted = Image.fromarray(255 - np.array(combined_mask))
        result = base.copy()
        result.paste(neutral, mask=inverted)

    return result

def render_mask_preview(image: Image.Image, editor_value: dict) -> Image.Image | None:
    """
    Returns the original image with red mask strokes blended on top at 50% opacity.
    Used purely for display — not sent to the API.
    """
    import numpy as np

    if image is None or editor_value is None:
        return image

    combined_mask = Image.new("L", image.size, 0)
    for layer in (editor_value.get("layers") or []):
        if layer is None:
            continue
        if not isinstance(layer, Image.Image):
            layer = Image.fromarray(layer)
        layer_rgba = layer.convert("RGBA").resize(image.size, Image.LANCZOS)
        combined_mask = Image.fromarray(
            np.maximum(
                np.array(combined_mask),
                np.array(layer_rgba.split()[3])
            )
        )

    if not np.any(np.array(combined_mask)):
        return image  # no strokes drawn, return original

    overlay     = Image.new("RGB", image.size, (220, 50, 50))
    base        = image.convert("RGB")
    # Scale mask to 50% opacity for the overlay blend
    blend_mask  = Image.fromarray((np.array(combined_mask) * 0.55).astype(np.uint8))
    preview     = base.copy()
    preview.paste(overlay, mask=blend_mask)
    return preview

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
            /* Constrain mask editor canvas to viewport so it never overflows */
            [id^="vp_mask_editor_"] .image-editor-container,
            [id^="vp_mask_editor_"] canvas {
                max-height: 50vh !important;
                object-fit: contain !important;
            }

            [id^="vp_mask_editor_"] .image-editor {
                max-height: 50vh !important;
                overflow: hidden !important;
            }

            /* Style for checkbox in tab label */
            .tab-checkbox {
                display: inline-block;
                margin-left: 8px;
                vertical-align: middle;
            }
            .tab-checkbox input[type="checkbox"] {
                margin: 0;
                vertical-align: middle;
            }
            </style>
            """)
        with gr.Accordion("Vision Prompt", open=False):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable Vision Prompt", 
                    value=False)

                always_regenerate = gr.Checkbox(
                    label="Always Re-Generate (Skip Cache)",
                    value=False
                )
                gr.HTML("<div style='flex-grow: 1;'></div>")

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
                        maximum=8192,
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
					
                with gr.Row():
                    write_png_meta = gr.Checkbox(
                        label="Write settings to PNG metadata",
                        value=False,
                        info="When enabled, VP settings and system prompts are embedded in the output PNG. Disable to keep metadata clean."
                    )
                    
                    reasoning_budget = gr.Dropdown(
                        label="",
                        choices=["none", "low", "medium", "high"],
                        value="none",
                        info="Only for models that support reasoning! Set Max Tokens to at least 1024 for this to work properly."
                    )

            tabs_data = []

            with gr.Tabs():
                for i in range(3):  # Changed from 4 to 3 slots
                    # Create tab with checkbox in label
                    with gr.Tab(f"Vision Input #{i+1}") as tab:
                        # Add a row at the top with the enable checkbox
                        with gr.Row():
                            slot_enabled = gr.Checkbox(
                                label="Enable",
                                value=True,
                                scale=0,
                                min_width=80
                            )
                            # Spacer to push content to the right
                            gr.HTML("<div style='flex-grow: 1;'></div>")

                        # ── Multiple image drop zones ──────────────────────
                        gr.Markdown(
                            f"Upload up to **{IMAGES_PER_SLOT} images** — they will be "
                            "stitched side-by-side before being sent to the vision model."
                        )

                        image_inputs    = []
                        mask_states     = []
                        original_states = []
                        edit_buttons    = []
                        mask_modes      = []  # Per-image mask mode states

                        with gr.Row(elem_classes="vp-image-box"):
                            for j in range(IMAGES_PER_SLOT):
                                with gr.Column(min_width=120):
                                    img_input = gr.Image(
                                        type="pil",
                                        label=f"Image {j+1}",
                                        height=256,
                                        elem_classes="vp-image-box",
                                    )
                                    edit_btn = gr.Button("✏️ Mask", size="sm", min_width=60)

                                    # Per-image mask mode selector (initially hidden)
                                    mask_mode_radio = gr.Radio(
                                        label="Mask mode",
                                        choices=["none", "exclude", "include"],
                                        value="include",
                                        visible=False,
                                    )

                                image_inputs.append(img_input)
                                mask_states.append(gr.State(None))
                                original_states.append(gr.State(None))
                                edit_buttons.append(edit_btn)
                                mask_modes.append(gr.State("include"))

                        # One shared ImageEditor per slot — hidden until a Mask button is clicked
                        with gr.Group(visible=False) as mask_editor_group:
                            gr.Markdown("**Draw over areas to mask (red brush)**")
                            mask_editor = gr.ImageEditor(
                                label="Mask Editor",
                                brush=gr.Brush(colors=["#ff0000"], default_size=64, color_mode="fixed"),
                                eraser=gr.Eraser(),
                                layers=False,
                                type="pil",
                                elem_id=f"vp_mask_editor_{i}",
                            )
                            with gr.Row():
                                apply_mask_btn = gr.Button("✅ Apply mask", variant="primary")
                                clear_mask_btn = gr.Button("🗑️ Clear mask")
                                close_mask_btn = gr.Button("✖ Close")
                            active_mask_idx = gr.State(-1)

                            # Shared mask_mode for editor (synced with active image's mode)
                            mask_mode = gr.Radio(
                                label="Mask mode",
                                choices=["none", "exclude", "include"],
                                value="include",
                            )

                        # Open editor: load image into editor and save original + sync mask_mode
                        for j in range(IMAGES_PER_SLOT):
                            def open_editor(img, current_mode, _j=j):
                                if img is None:
                                    return (
                                        gr.update(),
                                        gr.update(visible=False),
                                        _j,
                                        None,
                                        gr.update(),  # mask_mode update
                                    )
                                # current_mode is passed in as a proper Gradio state value
                                if not current_mode:
                                    current_mode = "include"
                                return (
                                    gr.update(value={
                                        "background": img,
                                        "layers": [],
                                        "composite": img,
                                    }),
                                    gr.update(visible=True),
                                    _j,
                                    img,  # saved into original_states[j]
                                    gr.update(value=current_mode),  # Sync editor's mask_mode
                                )
                            edit_buttons[j].click(
                                fn=open_editor,
                                inputs=[image_inputs[j], mask_modes[j]],
                                outputs=[mask_editor, mask_editor_group, active_mask_idx, original_states[j], mask_mode],
                            )

                        # Reset mask/original/mode state whenever a new image is dropped in
                        for j in range(IMAGES_PER_SLOT):
                            def on_image_change(new_img, _j=j):
                                # Returns: original_state, mask_state, mask_mode_state, mask_mode_radio update
                                if new_img is None:
                                    return None, None, "include", gr.update(value="include", visible=False)
                                # New image — clear stale mask data so the fresh image is used
                                return new_img, None, "include", gr.update(value="include", visible=False)
                            image_inputs[j].change(
                                fn=on_image_change,
                                inputs=[image_inputs[j]],
                                outputs=[original_states[j], mask_states[j], mask_modes[j], mask_mode],
                            )

                        # Apply: save mask state, mask_mode, and composite preview onto the upload zone in-place
                        def do_apply_mask(editor_val, idx, current_mask_mode, *states):
                            # states = [orig0..N, mask0..N, mode0..N]
                            originals = list(states[:IMAGES_PER_SLOT])
                            masks     = list(states[IMAGES_PER_SLOT:IMAGES_PER_SLOT*2])
                            modes     = list(states[IMAGES_PER_SLOT*2:])

                            if 0 <= idx < len(masks):
                                masks[idx] = editor_val
                                modes[idx] = current_mask_mode  # Save the mode

                            image_updates = []
                            mode_updates = []
                            for k in range(IMAGES_PER_SLOT):
                                if k == idx and editor_val is not None and originals[k] is not None:
                                    image_updates.append(gr.update(value=render_mask_preview(originals[k], editor_val)))
                                    mode_updates.append(gr.update(value=current_mask_mode, visible=True))
                                else:
                                    image_updates.append(gr.update())
                                    mode_updates.append(gr.update())

                            return [gr.update(visible=False)] + masks + modes + image_updates

                        apply_mask_btn.click(
                            fn=do_apply_mask,
                            inputs=[mask_editor, active_mask_idx, mask_mode] + original_states + mask_states + mask_modes,
                            outputs=[mask_editor_group] + mask_states + mask_modes + image_inputs,
                        )

                        # Clear: wipe mask state, mode, and restore original image to upload zone
                        def do_clear_mask(idx, *states):
                            originals = list(states[:IMAGES_PER_SLOT])
                            masks     = list(states[IMAGES_PER_SLOT:IMAGES_PER_SLOT*2])
                            modes     = list(states[IMAGES_PER_SLOT*2:])

                            if 0 <= idx < len(masks):
                                masks[idx] = None
                                modes[idx] = "include"  # Reset to default

                            image_updates = [
                                gr.update(value=originals[k]) if k == idx else gr.update()
                                for k in range(IMAGES_PER_SLOT)
                            ]
                            mode_updates = [
                                gr.update(value="include", visible=False) if k == idx else gr.update()
                                for k in range(IMAGES_PER_SLOT)
                            ]
                            return [gr.update(value=None)] + masks + modes + image_updates + mode_updates

                        clear_mask_btn.click(
                            fn=do_clear_mask,
                            inputs=[active_mask_idx] + original_states + mask_states + mask_modes,
                            outputs=[mask_editor] + mask_states + mask_modes + image_inputs + [m for m in mask_modes],  # Update all mode radios
                        )

                        close_mask_btn.click(
                            fn=lambda: gr.update(visible=False),
                            outputs=[mask_editor_group],
                        )

                        # ── Preset / system prompt ─────────────────────────
                        gr.Markdown("**Preset**")
                        with gr.Row(equal_height=True):
                            preset_dropdown = gr.Dropdown(
                                choices=list(SYSTEM_PROMPT_PRESETS.keys()),
                                value="Custom",
                                scale=2,
                                show_label=False
                            )

                            preset_name_input = gr.Textbox(
                                placeholder="Preset name...",
                                show_label=False,
                                scale=1,
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
                            label="Weight",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05
                        )

                        # Pack: [slot_enabled, img0..N, orig0..N, mask0..N, mode0..N, system_prompt, weight]
                        tabs_data.append(slot_enabled)
                        tabs_data.extend(image_inputs)
                        tabs_data.extend(original_states)
                        tabs_data.extend(mask_states)
                        tabs_data.extend(mask_modes)
                        tabs_data.extend([system_prompt, weight])

        # ── Persistent settings & PNG metadata ────────────────────────────
        # Stable label strings become keys in PNG infotext.
        # api_key intentionally excluded — never write secrets to metadata.
        # write_png_meta controls whether any of these are actually written at generation time.
        self.infotext_fields = [
            (enabled,        "VP Enabled"),
            (api_url,        "VP API URL"),
            (model_name,     "VP Model"),
            (temperature,    "VP Temperature"),
            (top_p,          "VP Top-p"),
            (top_k,          "VP Top-k"),
            (max_tokens,     "VP Max Tokens"),
        ]

        # Slot layout: [slot_enabled, img0..N, orig0..N, mask0..N, mode0..N, system_prompt, weight]
        SLOT_STRIDE       = 1 + IMAGES_PER_SLOT * 4 + 2  # 1 for slot_enabled + images + modes + system_prompt + weight
        SLOT_ENABLED_OFF  = 0
        SYSTEM_PROMPT_OFF = 1 + IMAGES_PER_SLOT * 4
        WEIGHT_OFF        = 1 + IMAGES_PER_SLOT * 4 + 1

        for i in range(3):
            base = i * SLOT_STRIDE
            self.infotext_fields += [
                (tabs_data[base + SLOT_ENABLED_OFF],  f"VP Slot {i+1} Enabled"),
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

        # Slot layout: [slot_enabled, img0..N, orig0..N, mask0..N, mode0..N, system_prompt, weight]
        SLOT_STRIDE       = 1 + IMAGES_PER_SLOT * 4 + 2  # 1 for slot_enabled + images + modes + system_prompt + weight
        SLOT_ENABLED_OFF  = 0
        ORIG_OFF          = 1 + IMAGES_PER_SLOT
        MASK_OFF          = 1 + IMAGES_PER_SLOT * 2
        MODE_OFF          = 1 + IMAGES_PER_SLOT * 3
        SYSTEM_PROMPT_OFF = 1 + IMAGES_PER_SLOT * 4
        WEIGHT_OFF        = 1 + IMAGES_PER_SLOT * 4 + 1

        for i in range(3):
            base = i * SLOT_STRIDE
            if write_png_meta:
                p.extra_generation_params[f"VP Slot {i+1} Enabled"] = args[base + SLOT_ENABLED_OFF]
                p.extra_generation_params[f"VP Slot {i+1} System Prompt"] = args[base + SYSTEM_PROMPT_OFF]
                p.extra_generation_params[f"VP Slot {i+1} Weight"] = args[base + WEIGHT_OFF]

        combined_prompts = []

        for slot_idx in range(3):
            base = slot_idx * SLOT_STRIDE

            # Check if this slot is enabled
            slot_enabled = args[base + SLOT_ENABLED_OFF]
            if not slot_enabled:
                continue

            slot_images = [args[base + 1 + j]            for j in range(IMAGES_PER_SLOT)]
            slot_origs  = [args[base + ORIG_OFF + j]     for j in range(IMAGES_PER_SLOT)]
            slot_masks  = [args[base + MASK_OFF + j]     for j in range(IMAGES_PER_SLOT)]
            slot_modes  = [args[base + MODE_OFF + j]     for j in range(IMAGES_PER_SLOT)]
            system_prompt = args[base + SYSTEM_PROMPT_OFF]
            weight        = args[base + WEIGHT_OFF]

            # Apply masks with per-image modes
            valid_images = []
            for img, orig, mask_val, mask_mode in zip(slot_images, slot_origs, slot_masks, slot_modes):
                source = orig if orig is not None else img
                result = apply_mask(source, mask_val, mask_mode)  # Use per-image mode
                if result is not None:
                    valid_images.append(result)

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
            param_str = f"{temperature}|{top_p}|{top_k}|{max_tokens}|{reasoning_budget}"

            # Include all per-image mask modes in cache key
            mask_modes_str = "|".join(slot_modes)
            hash_input = (
                hash_parts
                + system_prompt.encode("utf-8")
                + model_name.encode("utf-8")
                + api_url.encode("utf-8")
                + param_str.encode("utf-8")
                + mask_modes_str.encode("utf-8")
            )
            cache_key  = hashlib.sha256(hash_input).hexdigest()

            slot_cache = VISION_CACHE[slot_idx]

            if not always_regenerate and cache_key in slot_cache:
                vision_output = slot_cache[cache_key]
                print(f"[Vision Prompt][Slot {slot_idx+1}] Cache HIT")
            else:
                if always_regenerate and cache_key in slot_cache:
                    print(f"[Vision Prompt][Slot {slot_idx+1}] Cache bypassed (Always Re-Generate) — calling API")
                else:
                    print(f"[Vision Prompt][Slot {slot_idx+1}] Cache MISS — calling API")

                img_base64 = base64.b64encode(img_bytes).decode("utf-8")

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
                }
                if top_k > 0:
                    payload["top_k"] = top_k

                _reasoning_budget_map = {
					"none": 0,
                    "low":    int(max_tokens * 0.2),
                    "medium": int(max_tokens * 0.5),
                    "high":  int(max_tokens * 0.8),
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
                    vision_output = data["choices"][0]["message"]["content"]

                    # Evict oldest entry if at capacity
                    if len(slot_cache) >= cache_size:
                        oldest_key = next(iter(slot_cache))
                        del slot_cache[oldest_key]
                    slot_cache[cache_key] = vision_output
                except requests.exceptions.Timeout:
                    msg = f"[Vision Prompt] Slot {slot_idx+1}: Request timed out after {timeout_seconds}s"
                    print(msg)
                    continue
                except Exception as e:
                    msg = f"[Vision Prompt]Slot {slot_idx+1}: API error: {e}"
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