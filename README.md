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

- Up to **4 independent prompt slots**
- Each slot supports **multiple images (auto-stitched)**
- **System prompt presets** (loaded from JSON file)
- Save / delete custom presets from the UI
- Adjustable **prompt weight (0.0 – 2.0)**
- **OpenAI-compatible API support** (local or remote)
- Optional **API key support**
- **Automatic caching** (avoids repeated API calls)
- Image resizing and compression before upload

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
3. Enable the feature using the checkbox

### Per Slot:

- Upload up to 4 images (they will be stitched horizontally)
- Select a preset or write a custom system prompt
- Adjust the weight slider

### Global Settings:

- Set your API endpoint (OpenAI-compatible)
- Enter API key if required
- Choose model name
- (Optional) Go to the WebUI Settings > Defaults > View Changes & Apply

### Generate

- Click **Generate** as usual
- The extension will:
  1. Send image(s) to the vision model
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

- Image content
- System prompt
- Model name

If nothing changes, the API will not be called again.

Cache is stored in memory and resets when WebUI restarts.

---

## Notes

- Large images are automatically resized and compressed before sending
- Multiple images per slot are stitched side-by-side
- Special characters in LLM output are escaped for compatibility

---

## Recommended Use Cases

- Use one slot for **style**, another for **composition**, etc.
- Use lower weights (0.3–0.8) for subtle influence
- Use higher weights (1.2+) for strong guidance

---

## Requirements

- AUTOMATIC1111 WebUI
- A vision-capable LLM endpoint (OpenAI or compatible)

---

## License

None
