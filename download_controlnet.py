#!/usr/bin/env python3
"""Fetch ControlNet models on first launch.

Forge auto-downloads ControlNet *preprocessors* on first use, but the ControlNet
models themselves are expected to be placed by hand. This fetches a working set
so ControlNet works out of the box, rather than shipping ~18 GB of weights in the
repo or a distribution archive.

Runs ONCE. After a successful run a stamp file is written and this script does
nothing on every later launch -- deliberately, so that deleting a model you don't
want does not cause it to come back on the next start.

Environment:
    FORGE_NO_CONTROLNET=1   skip entirely (nothing is downloaded, no stamp)
    FORGE_CONTROLNET=full   also fetch the optional extras (~+9 GB)
    FORGE_CONTROLNET=skip   same as FORGE_NO_CONTROLNET
    FORGE_MODELS_DIR        destination root (default: ./models)

Every URL here was verified to resolve. Models that live in gated repositories
are intentionally absent -- an auto-downloader that 401s is worse than one that
doesn't try. SDXL inpainting is covered by the union-promax model, which
includes an inpaint mode.
"""
import os
import sys

from download_llm import download, _fmt   # resumable .part downloader

HF = "https://huggingface.co"

# (filename, url, approx_GB, note)
ESSENTIAL = [
    ("controlnet-union-promax-sdxl_canny-depth-openpose-tile-scribble-lineart-softedge-seg-normal-inpaint.safetensors",
     f"{HF}/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors",
     2.5, "SDXL all-in-one: canny, depth, openpose, tile, scribble, lineart, softedge, seg, normal, inpaint"),
    ("control_v11p_sd15_canny_fp16.safetensors",
     f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors",
     0.7, "SD1.5 canny"),
    ("control_v11f1p_sd15_depth_fp16.safetensors",
     f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors",
     0.7, "SD1.5 depth"),
    ("control_v11p_sd15_openpose_fp16.safetensors",
     f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors",
     0.7, "SD1.5 openpose"),
    ("control_v11p_sd15_inpaint_fp16.safetensors",
     f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors",
     0.7, "SD1.5 inpaint (used by Replacer in sd mode)"),
    ("control_v11f1e_sd15_tile_fp16.safetensors",
     f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors",
     0.7, "SD1.5 tile (upscaling)"),
]

EXTRA = [
    ("xinsir-canny-sdxl.safetensors",
     f"{HF}/xinsir/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
     2.5, "SDXL canny, dedicated (higher quality than the union model)"),
    ("xinsir-depth-sdxl.safetensors",
     f"{HF}/xinsir/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
     2.5, "SDXL depth, dedicated"),
    ("controlnet++_openpose_sdxl.safetensors",
     f"{HF}/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
     2.5, "SDXL openpose, dedicated"),
    ("control_v11p_sd15_normalbae_fp16.safetensors",
     f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors",
     0.7, "SD1.5 normal map"),
    ("control_v11e_sd15_ip2p_fp16.safetensors",
     f"{HF}/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors",
     0.7, "SD1.5 instruct-pix2pix"),
]

STAMP = ".controlnet_downloaded"


def main():
    mode = str(os.environ.get("FORGE_CONTROLNET", "")).strip().lower()
    if os.environ.get("FORGE_NO_CONTROLNET") or mode == "skip":
        print("[controlnet] skipping download (FORGE_NO_CONTROLNET)")
        return 0

    here = os.path.dirname(os.path.abspath(__file__))
    models_root = os.environ.get("FORGE_MODELS_DIR", os.path.join(here, "models"))
    dest = os.path.join(models_root, "ControlNet")
    stamp = os.path.join(dest, STAMP)

    # Run-once. A model you deleted on purpose must stay deleted.
    if os.path.exists(stamp):
        return 0

    wanted = list(ESSENTIAL) + (list(EXTRA) if mode == "full" else [])

    os.makedirs(dest, exist_ok=True)
    need = [(fn, url, gb, note) for fn, url, gb, note in wanted
            if not (os.path.exists(os.path.join(dest, fn))
                    and os.path.getsize(os.path.join(dest, fn)) > 0)]

    if not need:
        # Everything already present (e.g. copied in by hand) -- stamp so we
        # never look again.
        open(stamp, "w", encoding="utf-8").write("all models already present\n")
        return 0

    total = sum(gb for _fn, _u, gb, _n in need)
    print(f"[controlnet] Fetching {len(need)} ControlNet model(s), ~{total:.1f} GB, first run only.")
    print("[controlnet] Set FORGE_NO_CONTROLNET=1 to skip, or FORGE_CONTROLNET=full for the extras.")

    ok = True
    got = []
    for fn, url, gb, note in need:
        out = os.path.join(dest, fn)
        print(f"[controlnet] {fn}  (~{gb:.1f} GB) -- {note}")
        try:
            download(url, out)
            got.append(fn)
        except Exception as e:
            ok = False
            print(f"[controlnet] WARNING: could not download {fn}: {e}")

    if ok:
        # Only stamp on a clean run, so a partial failure resumes next launch
        # instead of being silently abandoned.
        with open(stamp, "w", encoding="utf-8") as fh:
            fh.write("downloaded by download_controlnet.py; delete to re-run\n")
            for fn in got:
                fh.write(fn + "\n")
        print("[controlnet] ControlNet models ready.")
    else:
        print("[controlnet] Some downloads failed; re-run the launcher to resume the rest.")
    return 0     # never block Forge from launching


if __name__ == "__main__":
    sys.exit(main())
