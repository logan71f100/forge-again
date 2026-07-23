# Forge AI Assistant

A floating chat panel (🤖 button, bottom-right) that puts a **local vision LLM copilot**
inside Forge. The LLM sees every slider, field, dropdown and checkbox on your current tab
(txt2img, img2img, and extension tabs like Replacer), can change them for you, look at your
generated images, and run generations — freeing VRAM for Forge while it works.

## Works out of the box

A patched `llama-server` (with `/sleep` + `/wake` fast VRAM-hibernate endpoints) is bundled
in `forge-llm/`, and the vision model (Qwen3-VL-30B, ~18 GB) downloads automatically on the
first launch to `models/llm/`. Nothing to configure — click 🤖 and go. Set `FORGE_NO_LLM=1`
before launching to skip the model download if you don't want the assistant, or point it at
a different GGUF, your own `llama-server`, or the Claude API under **Settings → AI Assistant**.
(The bundled binary is a Windows/CUDA build; on Linux/macOS build your own `llama-server` and
set its path, or skip the assistant.)

## How it works

- **Backend** (`scripts/forge_ai_assistant.py`): adds `/forge-ai/*` API routes to Forge.
  Starts/stops the bundled `llama-server` as a subprocess, unloading the Forge model weights
  first so the LLM gets the VRAM. Chat is proxied to llama-server's OpenAI-compatible API.
- **Frontend** (`javascript/forge_ai_assistant.js`): the chat panel. Every request includes a
  fresh snapshot of all visible controls. The LLM answers with small JSON tool blocks
  (`set`, `get_image`, `generate`, `switch_tab`, …) that the browser executes.
- **The VRAM hibernate**: your conversation lives in the browser, so when the LLM calls
  `generate` it can hibernate *itself* — freeing its VRAM while keeping its KV cache in RAM —
  let Forge generate, wake back up in ~1.5 s, and critique the result, without losing the chat.

## Usage

1. Launch Forge (`start.bat`). On first run the vision model downloads (~18 GB, once).
2. Click the 🤖 bubble bottom-right.
3. The default model is preselected; hit **⏻ Start LLM** (or enable auto-start in Settings).
   This unloads the Forge model (Forge reloads it on the next generate) and boots the LLM.
4. Chat. The assistant automatically sees your source/inpaint image and the latest result —
   attached whenever they change (👁 in the header toggles this; duplicate frames are never
   re-sent). 🖼 / 📷 force-attach manually if you ever need to.
5. **⏹ Stop** kills the LLM and gives the VRAM back to Forge at any time.

Settings live under **Settings → AI Assistant**: provider (local `llama-server` or the Claude
API), model, task guidance, your own per-checkpoint notes, binary/model paths, launch args,
max tokens, and temperature.

## Vision

Image feedback ("does this look right?") needs a **multimodal** model — one with a
`mmproj-*.gguf` vision projector alongside the main GGUF. The bundled default is
**Qwen3-VL-30B-A3B** (+ `mmproj-F16`), a mixture-of-experts model (~3 B active per token) that
runs at usable speed on mid-range GPUs with experts spilling into system RAM via llama.cpp's
auto layer split, and handles multi-image compare (base vs. result). To use a different vision
model, drop its GGUF and sibling `mmproj-*.gguf` into a subfolder of the models dir and pick it
in the chat dropdown — the extension finds the mmproj and passes `--mmproj` automatically.

Without a vision model everything else still works; the assistant is told to admit it can't see
images rather than hallucinate.

## Notes / limits

- The `generate` tool waits on `/sdapi/v1/progress`, so it works with long jobs (Replacer,
  upscales) too.
- Gradio 6 dropdowns are custom widgets; the assistant sets them by clicking the matching
  option. If a dropdown refuses to change, set it manually and tell the assistant.
- **Stop** kills the process it spawned (by the PID listening on the API port).
- Chat history is kept in the browser tab — reloading the page clears it (your Forge UI
  settings persist regardless).
