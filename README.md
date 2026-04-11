# Vision Prompt Injector (A1111 Extension)

## Overview

Vision Prompt Injector is an extension for the AUTOMATIC1111 Stable Diffusion WebUI that allows you to use vision-capable LLMs to automatically generate prompt text from reference images.

Instead of manually describing style, composition, or characters, you can feed one or more images into a vision model (OpenAI-compatible API), and inject the generated description directly into your Stable Diffusion prompt.

This is especially useful for:

- Style transfer and analysis
- Composition and pose extraction
- Character tagging (e.g. Danbooru-style tags)
- Lighting and mood replication

---

## Features

- **3 independent prompt slots** (reduced from 4)
- Each slot supports **up to 3 images** (auto-stitched horizontally)
- **Per-slot enable/disable toggle** in tab header
- **Image masking system** with three modes: -`none` - No masking -`exclude` - Painted areas are blacked out (LLM ignores them) -`include` - Unpainted areas are blacked out (LLM focuses only on painted areas)
- **System prompt presets** (loaded from JSON file)
- Save / delete custom presets from the UI
- Adjustable **prompt weight (0.0 – 2.0)**
- **OpenAI-compatible API support** (local or remote)
- Optional **API key support**
- **API parameter controls**:
  - Temperature
  - Top-p
  - Top-k
  - Max tokens
- **Automatic caching** (avoids repeated API calls)
- Image resizing and compression before upload
- **PNG metadata embedding** - All settings (except API key) are saved to generated images for reproducibility

---

## Installation

1. Go to your WebUI extensions folder:

```
   stable-diffusion-webui/extensions/
```

2. Clone or copy this repository:

```
   git clone <this-repo-url>
```

3. Restart the WebUI.

---

## Usage

1. Open the **txt2img** or **img2img** tab
2. Expand the **Vision Prompt** accordion
3. Enable the feature using the main checkbox

### Per Slot:

- Click the checkbox in the tab header to enable/disable the slot
- Upload up to 3 images (they will be stitched horizontally)
- Use the **✏️ Mask** button on any image to draw masks:
  - Red brush paints areas to mask
  - Choose mask mode:`none`,`exclude`, or`include`
  - Click **✅ Apply mask** to save, **🗑️ Clear mask** to reset
- Select a preset or write a custom system prompt
- Adjust the weight slider

### Global Settings:

- Set your API endpoint (OpenAI-compatible)
- Enter API key if required
- Choose model name
- Adjust API parameters:
  - **Temperature** (0.0 – 2.0) - Controls randomness
  - **Top-p** (0.0 – 1.0) - Nucleus sampling
  - **Top-k** (0 – 100) - Top-k sampling
  - **Max Tokens** (16 – 1024) - Response length limit
- (Optional) Go to WebUI Settings > Defaults > View Changes & Apply

### Generate

- Click **Generate** as usual
- The extension will:
  1. Send masked image(s) to the vision model
  2. Receive a description
  3. Inject it into your prompt automatically

---

## Prompt Injection Behavior

Generated text is wrapped like this:

```
(generated prompt text:weight)
```

All active slots are combined and appended to your prompt in order.

---

## Presets

Presets are stored in:

```
extensions/vision_prompt_ext/presets.json
```

You can:

- Select from existing presets
- Save new ones
- Delete unwanted presets (except "Custom")

---

## Caching

The extension caches results based on:

- Image content (including masks)
- System prompt
- Model name
- API parameters
- Per-image mask modes

If nothing changes, the API will not be called again.

Cache is stored in memory and resets when WebUI restarts.

---

## PNG Metadata

All settings (except API key) are embedded into generated images as infotext. This allows you to:

- Reproduce exact settings by dragging the image back into the WebUI
- See which vision model and parameters were used
- Preserve slot configurations for future reference

Embedded information includes:

- Enabled status
- API URL and model name
- Temperature, top-p, top-k, max tokens
- Per-slot: enabled status, system prompt, weight

---

## Notes

- Large images are automatically resized and compressed before sending
- Multiple images per slot are stitched side-by-side
- Special characters in LLM output are escaped for compatibility
- Masks are visualized with red overlay at 55% opacity in the UI
- Each image in a slot can have its own independent mask mode

---

## Recommended Use Cases

- Use one slot for **style**, another for **composition**, etc.
- Use lower weights (0.3–0.8) for subtle influence
- Use higher weights (1.2+) for strong guidance
- Use **include** mode to focus the LLM on specific areas
- Use **exclude** mode to hide irrelevant parts of the image
- Combine multiple images in one slot for complex references

---

## Requirements

- AUTOMATIC1111 WebUI
- A vision-capable LLM endpoint (OpenAI or compatible)

---

## License

None
