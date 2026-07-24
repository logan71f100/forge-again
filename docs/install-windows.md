# Installing forge-again on Windows

**This is the primary development and test platform.** Everything is exercised here
first, including the AI assistant with its prebuilt VRAM-hibernate server.

## Requirements

- Windows 10 or 11
- An **NVIDIA GPU** (dependencies install CUDA 12.6 wheels) and a current driver
- ~12 GB of disk for Python and dependencies, plus your models
- Nothing else — no system Python, no Git, no CUDA toolkit. `curl` and `tar` ship with
  Windows 10+ and that's all the bootstrap needs.

Developed on an RTX 2080 Ti (11 GB). Everything, including Flux and the 30B assistant
model, runs in 11 GB via offloading and VRAM hibernate — more VRAM is just faster.

## Install

```
git clone https://github.com/logan71f100/forge-again
cd forge-again
start.bat
```

On first run `start.bat` downloads a portable Python 3.12, builds a venv, installs
PyTorch and every dependency, and launches. It's idempotent — re-runs skip finished
steps and start in seconds. A browser opens automatically.

Start in a specific mode with `start.bat sd`, `start.bat xl`, or `start.bat flux`
(default is whatever you used last). Modes also switch from the UI without a restart.

## Models

You supply your own checkpoints:

```
models\checkpoints\sd\      SD 1.5 checkpoints
models\checkpoints\xl\      SDXL checkpoints
models\checkpoints\flux\    Flux checkpoints
models\Lora  models\VAE  models\text_encoder  models\ESRGAN
```

Flux additionally expects `ae.safetensors` in `models\VAE` and `clip_l.safetensors` +
`t5xxl_fp8_e4m3fn.safetensors` in `models\text_encoder`.

To keep models on another drive, set `FORGE_MODELS_DIR` before launching.

## The AI assistant

**Works out of the box on Windows** — this is the only platform where it does, with no
build step. The repo ships a prebuilt `llama-server` in `forge-llm/` patched with
`/sleep` + `/wake` endpoints, and the vision model (Qwen3-VL-30B, ~18 GB) downloads
automatically on first launch.

To skip that download, set `FORGE_NO_LLM=1` before running `start.bat`. Everything —
provider, model, paths, task guidance — is configurable under **Settings → AI Assistant**;
you can point it at a different GGUF, your own `llama-server`, or the Claude API instead.

Don't trust a committed binary? Fair. The full source patch and build instructions are
in [`forge-llm/patches/`](../forge-llm/patches) so you can build it yourself and point
Settings at your own binary.

## Extra launch arguments

Put them on a single line in `extra-args.txt` next to the launcher (gitignored — create
it), or set `FORGE_EXTRA_ARGS`. Both are appended to every launch, including automatic
restarts. See the [README](../README.md#launch-arguments) for the useful ones.

## Notes and gotchas

- **Long paths.** Cloning deep into nested folders can trip Windows' 260-character path
  limit during dependency installation. Clone somewhere short, like `C:\forge-again` or
  `F:\forge-again`.
- **Antivirus.** Real-time scanning can dramatically slow the first dependency install
  and model loading. Consider excluding the install folder.
- **`--listen` is on by default**, so other machines on your LAN can reach the UI. That's
  usually wanted; if not, see [SECURITY.md](../SECURITY.md) for locking it down.
- **Set `FORGE_PORT`** to move off 7860 if something else already uses it.
