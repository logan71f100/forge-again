# Installing forge-again on macOS

> **Best-effort and untested.** The launcher exists and the stack is
> macOS-compatible in principle, but no part of this fork has been verified on a Mac.
> Expect rough edges, and please [open an issue](https://github.com/logan71f100/forge-again/issues)
> with a console traceback if you hit one — that's the only way it improves.

## Requirements

- Apple Silicon (M-series) strongly recommended. Intel Macs have no usable GPU
  acceleration here and will fall back to CPU, which is impractically slow for
  image generation.
- macOS recent enough for PyTorch's **MPS** backend
- ~12 GB of disk for Python and dependencies, plus your models

There is **no CUDA on macOS**. Torch uses the Metal (MPS) backend instead, which is a
genuinely different code path from the CUDA one everything else is tested against.

## Install

```bash
git clone https://github.com/logan71f100/forge-again
cd forge-again
./start-macos.sh
```

First run downloads a portable Python 3.12, builds the venv, installs dependencies, and
launches on `http://0.0.0.0:7860`. Add `sd`, `xl`, or `flux` to pick a startup mode.

## Models

```
models/checkpoints/sd|xl|flux/     one folder per mode
models/Lora  models/VAE  models/text_encoder  models/ESRGAN
```

Point `FORGE_MODELS_DIR` at an existing collection to avoid copying.

## What to expect

- **CUDA-specific launch arguments don't apply.** `--cuda-malloc` is a no-op, and the
  launcher passes `--disable-xformers` already (xformers is CUDA-only).
- **Memory pressure behaves differently.** Apple Silicon shares one pool between CPU and
  GPU, so Forge's VRAM management heuristics — written for discrete cards — don't map
  cleanly. SDXL should be workable on a machine with plenty of unified memory; Flux is
  demanding.
- **Some ControlNet preprocessors may not work.** Several depend on CUDA-only packages
  or prebuilt wheels with no macOS/arm64 build.
- **Speed.** MPS is considerably slower than an equivalent NVIDIA card. This is a
  PyTorch/Metal reality, not something the fork controls.

## The AI assistant

The bundled `llama-server` is a Windows CUDA binary and won't run here. You can build a
Metal version from the patch in [`forge-llm/patches/`](../forge-llm/patches):

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 33ca0dcb9d78c7c3a3b543db4c5fc9182abfe519
git apply /path/to/forge-again/forge-llm/patches/0001-sleep-wake-vram-hibernate.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j "$(sysctl -n hw.ncpu)"
```

then point **Settings → AI Assistant → llama-server binary path** at the result.

**Important caveat:** the hibernate path is written for discrete VRAM — it frees GPU
memory by moving weights and KV cache to host RAM. On unified memory that distinction
largely doesn't exist, so `/sleep` and `/wake` are **untested on Metal** and may not
help. The assistant should still function; the VRAM-reclaim benefit likely won't apply.

Otherwise, run with the assistant disabled so nothing tries to fetch the ~18 GB model:

```bash
FORGE_NO_LLM=1 ./start-macos.sh
```

## Recommendation

Use macOS for prompting, inpainting and lighter SD/SDXL work. For Flux, heavy ControlNet
pipelines, or the local assistant, a CUDA machine (native or [Docker](install-docker.md))
is the supported path.
