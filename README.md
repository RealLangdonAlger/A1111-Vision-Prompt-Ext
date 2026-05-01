# Vision Prompt Injector (A1111 Extension)

## Overview

Vision Prompt Injector is an extension for the AUTOMATIC1111 Stable Diffusion WebUI that uses vision-capable LLMs to automatically generate prompt text from reference images.

Instead of manually describing style, composition, or characters, you can feed one or more images into a vision model (OpenAI-compatible API), and inject the generated description directly into your Stable Diffusion prompt.

This is especially useful for:

- Style transfer and analysis
- Composition and pose extraction
- Character tagging (e.g. Danbooru-style tags)
- Lighting and mood replication

---

## Features

- **3 independent prompt slots** that run in parallel
- Each slot supports **up to 3 images** with two multi-image modes:
  - **Stitch**: Combine images side-by-side in one API call (fast, cheap)
  - **Merge Multicall**: One API call per image, then text-only merge (better quality, slower)
- **Per-slot enable/disable toggle** in tab header
- **Text-Only Mode** per slot — skip image processing entirely, use the LLM to generate prompt content from the system prompt alone
- **Prompt placeholders** for dynamic system prompts:
  - `{prompt}` - Full positive prompt text
  - `{negative_prompt}` - Full negative prompt text
  - `{prompt+N}` - Prompt starting from line N (0-indexed)
  - `{prompt-N}` - Prompt excluding last N lines
  - `{negative_prompt+N}`, `{negative_prompt-N}` - Same for negative prompt
- **System prompt presets** (loaded from JSON file)
- Save / delete custom presets from the UI (saved to a separate user file, survives extension updates)
- Adjustable **prompt weight (0.0 – 2.0)**
- **OpenAI-compatible API support** (local or remote)
- Optional **API key support**
- **Response style selector**: strict, balanced, or creative
- **Reasoning toggle** for thinking models (e.g., o1, Claude 3.7 Sonnet)
- **Skip cache** option to force fresh API calls
- **Skip Prompt** mode — replace the main prompt with the vision output while preserving LoRA tokens
- **Configurable timeout** (5–600 seconds)
- **Automatic caching** with LRU eviction (avoids repeated API calls)
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
3. Enable the feature using the main checkbox

### Per Slot:

- Click the checkbox in the tab header to enable/disable the slot
- Select **Multi-Image Mode**:
  - **Stitch**: Faster, cheaper. Images are combined horizontally and sent in one API call.
  - **Merge Multicall**: Better quality. Each image gets its own API call, then a final text-only call merges the results.
- Toggle **Text-Only Mode** to generate prompt content from the system prompt alone (no images needed)
- Upload up to 3 images per slot (ignored in Text-Only Mode)
- Select a preset or write a custom system prompt
- Use prompt placeholders to include parts of your main prompt or negative prompt
- Adjust the weight slider

### Global Settings:

- Set your API endpoint (OpenAI-compatible)
- Enter API key if required
- Choose model name
- Configure response style:
  - **Strict** (temp=0.2, top_p=0.2) - Consistent, deterministic output
  - **Balanced** (temp=0.7, top_p=0.9) - Moderate variability
  - **Creative** (temp=1.2, top_p=1.0) - High variability
- Enable reasoning for models that support it
- Set request timeout
- Toggle skip cache to bypass the cache

### Skip Prompt Mode

When **Skip Prompt** is enabled, the main prompt is replaced entirely with the vision-generated output. LoRA tokens (`<lora:name:weight>`) are extracted from the original prompt and prepended to the final output so they still get loaded by A1111.

### Generate

- Click **Generate** as usual
- The extension will:
  1. Process all enabled slots in parallel
  2. Send images to the vision model(s)
  3. Receive descriptions
  4. Inject them into your prompt automatically

---

## Prompt Injection Behavior

Generated text is wrapped like this:

```
(generated prompt text:weight)
```

All active slots are combined and appended to your prompt in order.

---

## Multi-Image Modes

### Stitch (Default)
Images are resized to the same height and combined horizontally into a single image. One API call is made. Fast and token-efficient.

### Merge Multicall
Each image is sent to the API independently (in parallel). The individual descriptions are then merged via a text-only API call that renumbers any labels (e.g., `<Character_1>`, `<Character_2>` become sequential). Better for complex multi-character or multi-scene references, but uses more tokens and time.

---

## Prompt Placeholders

System prompts support placeholders that get replaced at generation time:

| Placeholder | Replaced With |
|-------------|---------------|
| `{prompt}` | Full positive prompt text |
| `{negative_prompt}` | Full negative prompt text |
| `{prompt=1}` | Only the first line of the prompt |
| `{prompt=3}` | Only the third line of the prompt |
| `{prompt=0}` | Empty (for consistency) |
| `{prompt+0}` | Entire prompt from line 0 (first line) |
| `{prompt+2}` | Prompt starting from line 3 |
| `{prompt-1}` | Prompt excluding the last line |
| `{negative_prompt+1}` | Negative prompt from line 2 onwards |

This is useful for instructing the vision model to respect existing prompt structure or to focus on specific parts of your prompt.

---

## Presets

Presets come from two sources:

1. **Base presets** (`extensions/vision_prompt_ext/presets.json`) — shipped with the extension
2. **User presets** (`tmp/vision-prompt-presets_user.json`) — your custom additions, saved separately so they survive extension updates

When a user preset has the same name as a base preset, the user version takes precedence.

You can:

- Select from existing presets
- Save new ones (saved to user presets file)
- Delete unwanted presets (except "Custom")

---

## Caching

The extension caches results based on:

- Image pixel data
- System prompt (after placeholder resolution)
- Model name and API URL
- Response style and reasoning settings

If nothing changes, the API will not be called again. Cache is stored in memory (50 entries per slot, LRU eviction) and resets when WebUI restarts.

Use the **Skip Cache** checkbox to force fresh API calls when needed.

---

## Hires Fix & Batch Support

Vision prompt injection works with Hires Fix and batch/grid generation. The generated content is injected into `hr_prompt`, batch prompts, and grid prompts so the upscaling pass and all batch images include the vision-generated descriptions.

---

## Notes

- Large images are automatically resized and compressed before sending
- Multiple images per slot are processed according to the selected multi-image mode
- Special characters in LLM output are escaped for Stable Diffusion compatibility (`:`, `(`, `)`, `[`, `]`, `{`, `}`)
- LoRA tokens (`<lora:name:weight>`) are preserved when Skip Prompt is enabled
- All enabled slots run in parallel for faster processing
- The Merge Multicall mode performs Step 1 (per-image calls) in parallel

---

## Recommended Use Cases

- Use one slot for **style**, another for **composition**, etc.
- Use **Text-Only Mode** to have the LLM generate descriptive content from a system prompt alone (e.g., "Write a detailed description of a cyberpunk cityscape at night")
- Use lower weights (0.3–0.8) for subtle influence
- Use higher weights (1.2+) for strong guidance
- Use **Stitch** mode for related images (same character, consistent style)
- Use **Merge Multicall** for complex multi-character scenes requiring consistent labeling
- Use prompt placeholders to reference the main prompt in your vision instructions

---

## Requirements

- AUTOMATIC1111 WebUI
- A vision-capable LLM endpoint (OpenAI-compatible API)

---

## License

None
