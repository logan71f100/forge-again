# Performance and stack guide

What actually makes forge-again faster on the hardware it targets, what the
knobs in this repo do, and where the stack goes next. Written after a research
pass in September 2026 (sources at the end); the advice is ranked for an 11 GB
card, with an RTX 2080 Ti as the worked example because that is the reference
machine.

## What your card can and cannot do

| GPU | Arch | Has | Does not have |
|---|---|---|---|
| **RTX 2080 Ti** (11 GB) | Turing, sm_75 | fp16 tensor cores, int8, fp16 accumulation (measured +15 %) | bf16 (emulated, slow), TF32, flash SDPA, fp8, Triton (dropped after 3.2, so no `torch.compile` and no SageAttention 2) |
| RTX 3060/3080/3090 | Ampere, sm_80/86 | + bf16, TF32, flash SDPA, cuDNN SDPA, SageAttention 2, Triton | fp8 matmul |
| RTX 4070-4090 | Ada, sm_89 | + fp8 `_scaled_mm`, SageAttention 2++ | NVFP4 |
| RTX 50xx | Blackwell, sm_120 | + NVFP4, SageAttention 3 | needs a cu130 torch build (cu126 has no Blackwell SASS) |
| AMD RDNA3/4 (ROCm 7.1) | gfx11xx/12xx | AOTriton flash SDPA (torch 2.13 stable), bf16 | fp8, SageAttention, Nunchaku |

So on a 2080 Ti every attention-kernel and quantized-matmul trick of 2025-26
is off the table. What transfers is **doing less work per image**: skipping
model passes, better solvers so fewer steps look good, and cheaper memory
traffic. That is what this repo now ships.

## Ships in this repo

### First Block Cache (txt2img / img2img accordion, all models)

Runs only the first transformer block (Flux, Chroma) or the first UNet stage
(SD1.x, SD2.x, SDXL). If its output moved less than a threshold since the last
fully computed step, the rest of the network is skipped and that step's cached
residual is reused. This is the TeaCache / WaveSpeed "FBCache" scheme that
ComfyUI (EasyCache), Forge Classic (sd-forge-blockcache) and reForge users run.

- Published numbers: SDXL 896x1152 on a 3060, 45 s -> 35 s (FBCache) or 25 s
  (TeaCache); Flux 1.5-2x at threshold 0.12.
- **Measured on the reference RTX 2080 Ti** (Chroma1-HD Q6_K GGUF, 1024x1024,
  16 steps, CFG 1 with the flash-heun LoRA, start 0.15): threshold 0.12 ran
  **1.37x faster** (5 of 16 forward passes skipped) and was visually
  indistinguishable from the uncached image (mean pixel difference 3.66/255);
  threshold 0.08 gave 1.20x at 2.06/255. On the everyday Chroma setup (Heun,
  20 steps, flash-heun LoRA, 896x1152) it was **1.52-1.59x** -- Heun's
  corrector call lands close to its predictor, so 14 of 39 forwards skip.
  **On by default in flux mode** (set_mode.py), off in sd and xl:
- **Measured on SDXL and SD 1.5, and left off there.** JuggernautXL, DPM++ 2M
  SDE, 30 steps: only 1.08-1.11x (2-3 of 30 skipped -- the SDE sampler's
  per-step noise keeps the first stage from settling). SD 1.5 epicrealism,
  28 steps: 1.15x, but with visible composition drift at 512x640, where the
  latent is small (0.08: near-identical, 1.06x). Not worth the drift on
  images that already take 2-11 s; enable per run if you want it.
- Threshold 0.05 is conservative, 0.12 the usual Flux default, 0.2+ trades
  visible detail. Start at 0.10-0.20 of the run keeps the composition steps
  exact. "Max consecutive skips" forces a full step after N cached ones if
  you see repetitive texture.
- Chroma's first-block residual moves more than Flux's, so it caches less
  often at the same threshold; raise it or accept fewer hits.
- Do not combine with FreeU or other block patches; the cached residual would
  carry the patch's old contribution.
- The console prints "skipped N of M model forward passes" after each run.

Implementation: `backend/misc/first_block_cache.py`, hooked in
`backend/nn/flux.py`, `backend/nn/chroma.py`, `backend/nn/unet.py`; UI in
`extensions-builtin/sd_forge_first_block_cache`. Unit tests in
`tests/unit/test_first_block_cache.py`.

### New samplers and a scheduler (Sampling method dropdown)

Ported from ComfyUI's 2025 solver set: **Res Multistep** (and Ancestral),
**Gradient Estimation**, **ER SDE**, **SEEDS 2**, **SEEDS 3**, plus the
**Linear Quadratic** schedule. Same cost per step as Euler/DPM++ (SEEDS 2/3
are 2- and 3-stage, so 2x/3x model calls per step like Heun) with better
quality at 15-20 steps; ER SDE keeps saturation better than DPM++ SDE,
Gradient Estimation gives cleaner edges on line-art. The rectified-flow
variants (Flux/Chroma) use the flow SNR conversions, not the Karras ones.
Code: `backend/modules/k_diffusion_extra.py`; tests in
`tests/unit/test_samplers.py`.

### CFG-Zero* (accordion)

Projects the conditional prediction onto the unconditional one and rescales
the unconditional branch before applying the guidance scale (the ComfyUI
CFGZeroStar node), with optional zero-init of the first step(s). Built for
flow models with real CFG (Chroma, Flux with CFG > 1, SD3); harmless at CFG 1.
Code: `extensions-builtin/sd_forge_cfg_zero_star`.

### Launch flags (put them on one line in `extra-args.txt`)

| Flag | What | 2080 Ti | Ampere+ |
|---|---|---|---|
| `--cudnn-benchmark` | cuDNN autotunes conv algorithms for every new shape, trial-running each candidate -- including memory-hungry ones | **no**, if you change resolution or run near full VRAM: on the reference 2080 Ti the autotune trials spilled into system RAM and turned a VAE decode into a multi-minute stall, and the flag was removed for it | yes, if you keep the same size |
| `--fast-fp16-accumulation` | fp16 matmuls accumulate in fp16 (`allow_fp16_accumulation`). **On by default on NVIDIA**; `--no-fast-fp16-accumulation` turns it off. ComfyUI measured +10-15 % on SD1.5/SDXL at batch 1, +25-33 % at batch 2+, on 3090/4090. Covers GGUF Flux/Chroma too, which dequantize to fp16 on Turing. | **yes -- measured +15 % on Chroma, +13-15 % on SDXL, ~5 % on SD 1.5**; no visible change at 1:1 up to 1304x2048, no NaNs; stacks with First Block Cache (Chroma 1.71x together) | yes |
| `--tf32` | TF32 matmul/conv | no effect | yes |
| `--use-sage-attention` | INT8 attention | not available (needs Triton for sm_75) | yes, 20-40 % on Flux, head dim 64/128 only |
| `--cuda-stream` / `--pin-shared-memory` | Async weight swap on a second CUDA stream, pinned host memory. Also the **Swap Method = Async** and **Swap Location = Shared** radios at the top of the page. ComfyUI made both default in Dec 2025 and measured 10-50 % when weights must be offloaded; reForge reports 15-25 % on SDXL. | **yes** whenever the model does not fit (Flux) | yes |
| `--vram-fraction 0.9` | Cap this process's VRAM so over-allocation raises OOM (which the VAE catches with a tiled retry) instead of Windows silently paging the card into system RAM | see below | same |

**On "bleeding into system RAM".** When the driver's sysmem fallback kicks in
the run keeps going but every touched page crosses PCIe; it can look fine for a
model that only just overflows and awful for one that overflows by gigabytes,
and Forge's own OOM handling never fires because no error is raised. If it
performs well for you, leave it; if a run ever "hangs" at 100 % GPU with low
power draw, `--vram-fraction` turns that into a fast, recoverable OOM.

### Memory settings that matter on 11 GB

- **Flux**: GGUF Q4_K_M or Q5_K_M (7-8 GB) is the practical format on Turing;
  fp8 weights still work as a memory saver (dequantized on the fly) but bring
  no matmul speedup without fp8 hardware. Keep the T5 text encoder in fp8 or
  on CPU.
- **SDXL**: keep the VAE in fp16 with madebyollin's `sdxl-vae-fp16-fix`
  instead of forcing fp32 (Turing has no bf16, which is what Ampere uses
  here); decode is ~2x faster and half the memory, so fewer tiled decodes.
- **Previews**: `TAESD` or `Approx NN` in Settings > Live previews; never
  "Full" on a card that swaps.
- **Diffusion in Low Bits = Automatic** already stores in float8 only when
  fp16 would not fit.

### Fewer steps: distillation LoRAs

Cheapest speedup there is, no code involved. SDXL: DMD2 4-step LoRA (weight
~0.7, LCM sampler, CFG 1), Hyper-SD, Lightning. Flux.1-dev: Turbo-Alpha
(8 steps, guidance 3.5). Chroma: Chroma1-Flash checkpoint (8 steps, CFG 1,
heun / dpmpp_2s_ancestral). At CFG 1 the negative prompt is ignored; CFG-Zero*
does nothing there either.

## Not worth it on a 2080 Ti (and why)

- **SageAttention 1/2/3, FlashAttention 2/3/4, cuDNN SDPA**: all need sm_80+
  (Sage on sm_75 needs a Triton fork). Torch's mem-efficient SDPA kernel is
  what you get, and it is about equal to xformers-cutlass on Turing.
- **torch.compile / Triton / CUDA graphs**: Triton dropped sm_75 after 3.2;
  Forge's changing shapes would recompile constantly anyway.
- **fp8 `_scaled_mm`, torchao fp8/int8, NVFP4**: Ada / Blackwell only. fp8
  dynamic quant is *slower* than fp16 on conv UNets even on Ada.
- **Nunchaku (SVDQuant int4)**: 3x faster Flux where it runs and Turing is
  supported since 1.2.0, but there is no wheel for torch 2.13 and it needs its
  own checkpoints and loader. Worth revisiting when a 2.13/2.14 wheel ships.
- **TaylorSeer / DeepCache / MagCache**: TaylorSeer's cache is tens of GB;
  DeepCache is superseded by First Block Cache and fights ControlNet;
  MagCache needs per-model calibration.

## Stack: applied now, and what comes next

Applied in this branch (each checked against the gradio 6.20 dependency
ranges): `Pillow>=12.3.0` (four 2026 CVEs), `safetensors>=0.8.0`,
`starlette>=1.3.1` and `python-multipart>=0.0.30` (2026 DoS CVEs),
`spandrel 0.4.2` + `spandrel-extra-arches 0.2.0` (MoSR, SeemoRe, RealPLKSR
DySample/LayerNorm, MoESR, RCAN, FDAT, AuraSR upscalers), `einops 0.8.2`,
`accelerate 1.15.0`, `torchdiffeq 0.2.5`.

Next, in this order, each with a `python tests/run_tests.py` run because the
installer cannot be exercised from a static session:

1. **protobuf**: 3.20.3 has no cp312 wheel, so Python 3.12 runs the pure-Python
   backend, which is the code path of CVE-2025-4565 / CVE-2026-0994 (fixed in
   5.29.6+). The cap comes from `open-clip-torch==2.20.0`, which nothing in
   this repo imports at runtime (only `launch_utils.py` installs it);
   `open-clip-torch>=2.23` dropped the protobuf cap and 3.3.0 has no protobuf
   dependency at all. The blocker is `mediapipe` (installed by
   `forge_legacy_preprocessors`), which pins its own protobuf range; resolve
   the three together or drop open-clip.
2. **pytorch_lightning 1.9.4 -> 2.6.6** (or replace with a stub in
   `modules/safe.py`, it is only an unpickle target), which lifts the
   `setuptools<81` cap. 2.6.2/2.6.3 were compromised on PyPI; use 2.6.6.
3. **pydantic 2.13.x -> fastapi 0.141 -> gradio 6.28** (6.21-6.28 is UI
   churn, test Gallery/ImageEditor; no gradio CVE is fixed above 6.20).
4. **transformers 5.17 + diffusers 0.40** (weekly minors keep shipping breaks;
   test CLIP/T5 loading).
5. **torch 2.14 / torchvision 0.29**: cu126 is "legacy" in 2.14 and removed in
   2.15. The next index is **cu130** (driver >= 580), which keeps Turing and
   adds Blackwell but drops Pascal/Volta. Pair triton-windows 3.8.0.post28.
   `bitsandbytes 0.50.2` ships cu126 and cu130 binaries and tests against
   torch 2.13 (the installer pins 0.45.3 today). xformers 0.0.35 is on the
   cu126 index and torch-ABI-stable; smoke-test it before trusting it.
6. **Python stays 3.12**: numpy 2.5 needs >= 3.12, 3.13 is workable once the
   above lands, 3.14 and free-threaded builds are not (no kernel wheels).

## New models worth adding (research summary, not implemented)

Ranked by quality gain x fits-in-11-GB x porting effort in this backend:

1. **Z-Image-Turbo / Z-Image-Omni-Base** (Alibaba, Apache-2.0, Nov 2025 /
   Jan 2026): 6B single-stream DiT, Qwen3-4B text encoder, Flux's 16-channel
   VAE; 8 steps at CFG 1. fp8 ~8 GB, GGUF ~6 GB. The de-facto Flux-dev
   replacement on consumer cards; ComfyUI PR #10892 is the reference.
   Cost: new model class + text-encoder path + LoRA key map.
2. **FLUX.2 [klein] 4B** (BFL, Apache-2.0, Jan 2026): text-to-image and
   multi-reference editing in one 4-step model, fp8 ~8-9 GB. Cost: Flux.2
   blocks, Qwen3 encoder, new 32-channel VAE (ComfyUI PR #11890).
3. **Chroma1-HD / Chroma1-Flash**: drop-in checkpoints for the Chroma path
   that already exists here.
4. **Guidance**: NAG (negative prompts for distilled CFG-1 models), APG, TCFG
   as further post-CFG functions like CFG-Zero*.
5. **Qwen-Image-Edit-2511**: best open editor, but 20B; needs Q3/Q2 GGUF plus
   the Lightning LoRA on 11 GB. Borderline.

## Sources

ComfyUI EasyCache PR #9496 and WaveSpeed FBCache (chengzeyi/Comfy-WaveSpeed);
Haoming02/sd-forge-blockcache measurements; ComfyUI fp16 accumulation PR
#6453; ComfyUI async-offload blog (2026-01-09); thu-ml/SageAttention and
woct0rdho wheels; triton-lang/triton-windows; nunchaku-ai/nunchaku releases;
madebyollin/sdxl-vae-fp16-fix; ComfyUI samplers (`comfy/k_diffusion/sampling.py`)
and PRs #6554 / #6731; WeichenFan/CFG-Zero-star; PyPI metadata and the
download.pytorch.org wheel indexes (2026-09-25); GitHub security advisories
for gradio, starlette, python-multipart, Pillow, protobuf, pytorch-lightning.
