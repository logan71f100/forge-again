# Installing forge-again on Linux

Two options: run it **natively** with `start.sh`, or run it in **[Docker](install-docker.md)**.

**Docker is verified** — the image has been built and run, the GPU confirmed visible
inside the container, and SDXL generation tested end to end. **The native launcher is
untested**: it shares its logic with the Windows launcher, which is exercised daily, but
nobody has run `start.sh` on a clean Linux box and confirmed it. If you do, an
[issue](https://github.com/logan71f100/forge-again/issues) either way is genuinely useful.

If you want the path most likely to just work, use Docker.

## Requirements

- x86_64 Linux
- An **NVIDIA GPU** (dependencies install CUDA 12.6 wheels) and a current driver
- `curl` and `tar` (standard everywhere)
- ~12 GB of disk for Python and dependencies, plus your models

No system Python and no CUDA toolkit are needed — the launcher fetches a portable
Python 3.12 and the torch wheels bring their own CUDA runtime.

## Native install

```bash
git clone https://github.com/logan71f100/forge-again
cd forge-again
./start.sh
```

First run downloads a portable Python, builds the venv, installs PyTorch and all
dependencies, then launches on `http://0.0.0.0:7860`. Re-runs skip completed steps.

Start in a specific mode with `./start.sh sd|xl|flux`; modes also switch from the UI
without restarting.

If you hit missing shared libraries when OpenCV loads (common on minimal/server
installs), you need the usual graphics stubs:

```bash
sudo apt-get install -y libgl1 libglib2.0-0        # Debian/Ubuntu
sudo dnf install -y mesa-libGL glib2               # Fedora/RHEL
```

## Models

```
models/checkpoints/sd|xl|flux/     one folder per mode
models/Lora  models/VAE  models/text_encoder  models/ESRGAN
```

Flux also expects `ae.safetensors` in `models/VAE` and `clip_l.safetensors` +
`t5xxl_fp8_e4m3fn.safetensors` in `models/text_encoder`. Point `FORGE_MODELS_DIR` at an
existing collection to avoid copying.

## The AI assistant — needs a build step

The bundled `llama-server` is a **Windows CUDA binary** and will not run here. To get the
assistant on Linux you build it from the patch in
[`forge-llm/patches/`](../forge-llm/patches):

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 33ca0dcb9d78c7c3a3b543db4c5fc9182abfe519
git apply /path/to/forge-again/forge-llm/patches/0001-sleep-wake-vram-hibernate.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j "$(nproc)"
```

Set `CMAKE_CUDA_ARCHITECTURES` for your card (`75` Turing/20xx, `86` Ampere/30xx,
`89` Ada/40xx, `120` Blackwell/50xx). This needs the **CUDA toolkit** (`nvcc`) installed —
the only part of the project that does.

Then set **Settings → AI Assistant → llama-server binary path** to your build. That gives
you the same `/sleep` + `/wake` VRAM hibernate the Windows build has.

Until you do that, run with `FORGE_NO_LLM=1` so the launcher doesn't try to fetch the
~18 GB vision model:

```bash
FORGE_NO_LLM=1 ./start.sh
```

The [Docker image](install-docker.md) automates this whole build — set
`WITH_ASSISTANT=1` and it compiles the patched server for you.

## Recommendation

If you want the assistant without hand-building anything, **use Docker** and set
`WITH_ASSISTANT=1`. If you just want Forge and prefer no container layer, the native
`start.sh` with `FORGE_NO_LLM=1` is the leanest route.

## Notes

- **`--listen` is on by default** — the UI binds to all interfaces. On a shared or
  internet-facing machine, put authentication in front of it. See [SECURITY.md](../SECURITY.md).
- `FORGE_PORT` changes the port; `FORGE_NO_BROWSER=1` suppresses auto-opening a browser
  (useful for headless/systemd).
- Extra arguments go in `extra-args.txt` (one line) or `FORGE_EXTRA_ARGS`.
