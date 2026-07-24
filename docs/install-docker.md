# Running forge-again in Docker

Runs on **any host that can run Docker with an NVIDIA GPU**. The image bakes in
Python 3.12, torch 2.13+cu126 and every dependency, so the container starts in
about 30 seconds instead of bootstrapping ~12 GB on first run.

Nothing about the image is host-specific — if `docker run --gpus all` works on your
machine, this works. Verified end to end on an RTX 2080 Ti (11 GB): the image builds,
the GPU is visible inside the container, and SDXL generation produces images.

## Requirements

- **Docker Engine** with the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** (this is what gives containers GPU access — plain Docker is not enough)
- An NVIDIA GPU + driver on the host. No CUDA toolkit needed on the host; the torch wheels bring their own runtime.
- ~15 GB of disk for the image, plus your models.

Confirm GPU passthrough works before going further:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi
```

If that prints your GPU, you're ready. If it fails, fix the Container Toolkit first —
nothing below will work until it does.

## Quick start

```bash
git clone https://github.com/logan71f100/forge-again
cd forge-again
mkdir -p models outputs && touch config.json
docker compose up -d
```

The first build takes a while (it downloads torch and the dependency set). Watch it with:

```bash
docker compose logs -f
```

Then open **http://localhost:7860**.

`touch config.json` matters: if that file doesn't exist, Docker creates a *directory*
with that name and Forge fails to save settings.

## Models

Checkpoints are **not** in the image — they're bind-mounted from `./models`, so they
survive rebuilds. Use the same layout as a native install:

```
models/checkpoints/sd|xl|flux/     one folder per mode
models/Lora  models/VAE  models/text_encoder  models/ESRGAN
```

To point at an existing models folder elsewhere, edit the volume in
`docker-compose.yml`:

```yaml
    volumes:
      - /path/to/your/models:/app/models
```

## Choosing the startup mode

`FORGE_MODE` in `docker-compose.yml` selects `sd`, `xl`, or `flux` at boot; you can
switch modes from the UI afterwards. Or pass it directly:

```bash
docker run --rm --gpus all -p 7860:7860 -v ./models:/app/models forge-again:latest flux
```

## The AI assistant (opt-in)

The assistant is **off by default** in Docker. The prebuilt `llama-server` in the repo
is a Windows CUDA binary that cannot run in a Linux container, so enabling the
assistant means compiling it from [`forge-llm/patches/`](../forge-llm/patches) inside
the image.

To enable it, set `WITH_ASSISTANT: "1"` in `docker-compose.yml` (and `CUDA_ARCH` to
match your GPU: `75`=Turing/20xx, `86`=Ampere/30xx, `89`=Ada/40xx, `120`=Blackwell/50xx),
then rebuild:

```bash
docker compose build && docker compose up -d
```

Be aware this adds a full CUDA compile stage to the build and pulls a **~18 GB** vision
model on first start. Leaving it off keeps the image far smaller, and Forge itself is
completely unaffected either way.

When enabled, you get the same `/sleep` + `/wake` VRAM hibernate as the Windows build:
the LLM parks itself in system RAM and hands the VRAM back to Forge while it generates.

## Known limitations

These are real and worth knowing before you file an issue:

- **Some ControlNet preprocessors need extra downloads.** Depth-Anything, HandRefiner
  and similar install their own packages the first time you use them. Those land in the
  container's filesystem and are lost when it's recreated. Everything else in ControlNet
  works normally.
- **The IDM-VTON space is unavailable.** It requires `basicsr`, which doesn't build
  against modern torchvision. It's absent from a working native install too.
- **A harmless startup warning from Replacer** — it tries to read its version from git,
  and `.git` isn't in the image. The extension itself works fine.
- **`--listen` is on**, so the UI binds to all interfaces in the container. Publishing
  the port to `0.0.0.0` exposes it to your network. Bind to localhost
  (`"127.0.0.1:7860:7860"`) or put authentication in front of it — see
  [SECURITY.md](../SECURITY.md).

## Common problems

**`could not select device driver "nvidia"`** — the NVIDIA Container Toolkit isn't
installed or the daemon wasn't restarted after configuring it:

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

**`torch.cuda.is_available()` is False inside the container** — you started it without
`--gpus all` (or without the compose `deploy.resources` block).

**Out of memory during generation** — the container gets the whole GPU; nothing is
reserved. Close other GPU applications, or lower resolution/batch size.

**Settings don't persist** — `config.json` was created as a directory. Remove it,
`touch config.json`, and recreate the container.

## Useful commands

```bash
docker compose logs -f          # follow logs
docker compose restart          # restart
docker compose down             # stop and remove
docker compose build --no-cache # rebuild from scratch
docker exec -it forge-again bash
```
