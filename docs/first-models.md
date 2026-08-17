# Getting your first models

forge-again ships no checkpoints — you supply those. This walks through one
complete, working setup step by step, with the exact files, where each one goes,
and the size of everything before you start downloading.

Every link here was checked against a working install: the file sizes below are
the bytes those URLs actually serve, and the VAE is byte-identical to the one in
a verified setup.

## Where files go

```
models/checkpoints/sd/      SD 1.5 checkpoints      <- "sd" mode
models/checkpoints/xl/      SDXL checkpoints        <- "xl" mode
models/checkpoints/flux/    Flux / Chroma           <- "flux" mode
models/VAE/                 ae.safetensors (flux)
models/text_encoder/        clip_l + t5xxl  (flux)
models/Lora/                LoRAs
models/ESRGAN/              upscalers
```

One folder per mode, because the mode switcher only scans the folder for the
mode you are in — a flux checkpoint in `models/checkpoints/xl/` will not appear.
`FORGE_MODELS_DIR` points the whole tree somewhere else if you already have a
collection.

## SD 1.5 or SDXL

Drop any `.safetensors` checkpoint into `models/checkpoints/sd/` or
`models/checkpoints/xl/` and start with `sd` or `xl`. Nothing else is needed —
these carry their own VAE and text encoder. Civitai and Hugging Face both work;
which checkpoint to pick is a matter of taste, so this guide does not choose one
for you.

```
start.bat xl          # Windows
./start.sh xl         # Linux
./start-macos.sh xl   # macOS
```

## Chroma (flux mode), step by step

Chroma is an 8.9B Apache-2.0 model based on FLUX.1-schnell. It is
**transformer-only**: unlike SD 1.5 and SDXL it does not contain a VAE or text
encoders, so those are three separate downloads. Miss them and the load fails
with `You do not have CLIP state dict!`.

### 1. Pick a quantization

Take the largest that leaves room to actually generate. The checkpoint has to
fit *alongside* the working memory a run needs, and on Apple Silicon it shares
one pool with the operating system.

| File | Size | Suits |
|---|---|---|
| [`Chroma1-HD-Q4_K_M.gguf`](https://huggingface.co/silveroxides/Chroma1-HD-GGUF/resolve/main/Chroma1-HD-Q4_K_M.gguf) | 5.18 GiB | 16 GB unified memory, 8 GB VRAM |
| [`Chroma1-HD-Q5_K_M.gguf`](https://huggingface.co/silveroxides/Chroma1-HD-GGUF/resolve/main/Chroma1-HD-Q5_K_M.gguf) | 6.19 GiB | 12 GB VRAM |
| [`Chroma1-HD-Q6_K.gguf`](https://huggingface.co/silveroxides/Chroma1-HD-GGUF/resolve/main/Chroma1-HD-Q6_K.gguf) | 7.13 GiB | 12–16 GB VRAM |
| [`Chroma1-HD-Q8_0.gguf`](https://huggingface.co/silveroxides/Chroma1-HD-GGUF/resolve/main/Chroma1-HD-Q8_0.gguf) | 9.07 GiB | 24 GB VRAM |

GGUF works on every backend — CUDA, ROCm and Metal. The dequantizer is plain
PyTorch and follows whatever device the tensors are on; there is no CUDA kernel
involved. On macOS the launcher already exports
`PYTORCH_ENABLE_MPS_FALLBACK=1`, which covers the integer bit operations Metal
does not implement.

→ save to `models/checkpoints/flux/`

### 2. The VAE

→ [`ae.safetensors`](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors) — 335,304,388 bytes — save to `models/VAE/`

> **Do not use the Black Forest Labs link for this file.** The obvious source,
> `black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors`, is gated behind
> a license acceptance and answers **HTTP 401** to a plain download — which looks
> like a broken install rather than a permissions prompt. The repackaged copy
> above is the same file, byte for byte, and needs no account.

### 3. The text encoders

→ both into `models/text_encoder/`

- [`clip_l.safetensors`](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors) — 0.23 GiB
- [`t5xxl_fp8_e4m3fn.safetensors`](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors) — 4.56 GiB

Take the **fp8** T5, not fp16: it halves the memory for no meaningful quality
loss at this size, and fp16 will not fit next to the checkpoint on a 16 GB
machine.

### 4. Start in flux mode

```
start.bat flux          # Windows
./start.sh flux         # Linux
./start-macos.sh flux   # macOS
```

The mode switcher writes the matching defaults (resolution, sampler, CFG) and
selects the VAE/text-encoder trio automatically. Chroma wants **CFG above 1** —
it is not a distilled model, so CFG 1 produces flat, blurry output. Around 4 is
a reasonable starting point, with Distilled CFG left alone.

Totals: **≈ 10.3 GiB** with Q4_K_M, **≈ 12.3 GiB** with Q6_K.

## Detection models (optional)

Replacer and Segment Anything need SAM and GroundingDINO weights in
`extensions/sd-webui-segment-anything/models/sam` and `.../grounding-dino`.
Everything else runs without them.

## If it does not load

- **`You do not have CLIP state dict!`** — a flux/Chroma checkpoint without its
  text encoders. Step 3.
- **HTTP 401 downloading a model** — a gated repository. Log in to Hugging Face
  and accept the license, or use an ungated mirror as in step 2.
- **The checkpoint is not in the dropdown** — it is in the wrong mode folder, or
  the UI is in a different mode. One folder per mode.
- **`fill` checkpoints produce errors in txt2img** — those are inpainting models.
  Use them from img2img/inpaint or Replacer.
- **Generation is far slower than expected** — the checkpoint plus its working
  memory does not fit, so weights are streaming from system RAM. Lower **GPU
  Weights** at the top of the page, or drop one quantization level.
