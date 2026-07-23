#!/usr/bin/env python3
"""Download the AI assistant's vision model on first launch.

Called by the start scripts after set_mode, before Forge launches. The model is
large (~18 GB) and public (unsloth GGUF), so it is fetched on demand rather than
bundled. Resumable and idempotent: skips files that are already complete.

Skip entirely with the FORGE_NO_LLM environment variable (the assistant simply
stays unavailable until the model is present). Destination follows
FORGE_MODELS_DIR so the LLM sits beside the SD models, matching the assistant's
default models dir (<models>/llm/).
"""
import os
import sys
import time
import urllib.request
import urllib.error

REPO = "unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF"
SUBDIR = "Qwen3-VL-30B-A3B-Thinking"
FILES = [
    "Qwen3-VL-30B-A3B-Thinking-UD-Q4_K_XL.gguf",  # ~17.7 GB main weights
    "mmproj-F16.gguf",                            # ~1 GB vision projector
]
CHUNK = 8 * 1024 * 1024


def _fmt(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


def download(url, out):
    part = out + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "forge-again-llm-fetch/1.0"})
    if have:
        req.add_header("Range", f"bytes=%d-" % have)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if have and e.code == 416:      # already complete
            os.replace(part, out)
            return True
        raise
    mode = "ab"
    if have and resp.status != 206:     # server ignored Range -> restart clean
        have = 0
        mode = "wb"
    total = have + int(resp.headers.get("Content-Length") or 0)
    done = have
    started = last = time.time()
    with open(part, mode) as f:
        while True:
            buf = resp.read(CHUNK)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            now = time.time()
            if now - last >= 2.0:
                last = now
                speed = (done - have) / max(now - started, 0.001)
                pct = f"{done * 100 // total}%" if total else _fmt(done)
                sys.stdout.write(f"\r    {pct} of {_fmt(total)}  ({_fmt(speed)}/s)   ")
                sys.stdout.flush()
    resp.close()
    sys.stdout.write("\n")
    if total and done < total:
        raise IOError("connection ended early (%s of %s) -- rerun to resume" % (_fmt(done), _fmt(total)))
    os.replace(part, out)
    return True


def _config(here):
    """Read the assistant's relevant settings from config.json if present, so we
    respect an existing setup instead of re-downloading."""
    cfg = {}
    try:
        import json
        with open(os.path.join(here, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        pass
    return cfg


def main():
    if os.environ.get("FORGE_NO_LLM"):
        print("[ai] FORGE_NO_LLM set -- skipping assistant model download")
        return 0
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = _config(here)

    # Claude/API provider needs no local model.
    if str(cfg.get("forge_ai_provider", "local")).strip().lower() not in ("", "local"):
        print("[ai] assistant provider is not 'local' -- no local model needed")
        return 0

    # Where the assistant will actually look (mirrors the extension defaults):
    # a configured models dir wins; else <FORGE_MODELS_DIR or ./models>/llm.
    default_dir = os.path.join(os.environ.get("FORGE_MODELS_DIR", os.path.join(here, "models")), "llm")
    models_dir = str(cfg.get("forge_ai_models_dir") or "").strip() or default_dir
    model_rel = str(cfg.get("forge_ai_model") or "").strip() or f"{SUBDIR}/{FILES[0]}"

    # Already have the configured model somewhere (e.g. an existing text-gen setup)? Done.
    configured = os.path.join(models_dir, model_rel.replace("/", os.sep))
    if os.path.exists(configured) and os.path.getsize(configured) > 0:
        return 0

    # If the config points at a CUSTOM model we don't host, don't guess — let the
    # user supply it. Only auto-fetch our known default.
    if model_rel.replace("\\", "/") != f"{SUBDIR}/{FILES[0]}":
        print(f"[ai] configured model '{model_rel}' not found under {models_dir}; provide it or change Settings > AI Assistant.")
        return 0

    dest = os.path.join(default_dir, SUBDIR)
    os.makedirs(dest, exist_ok=True)
    need = [f for f in FILES if not (os.path.exists(os.path.join(dest, f)) and os.path.getsize(os.path.join(dest, f)) > 0)]
    if not need:
        return 0

    print("[ai] Fetching the assistant vision model (~18 GB, first run only).")
    print("[ai] Set FORGE_NO_LLM=1 to skip this if you do not want the AI assistant.")
    ok = True
    for fn in need:
        out = os.path.join(dest, fn)
        url = f"https://huggingface.co/{REPO}/resolve/main/{fn}"
        print(f"[ai] downloading {fn} ...")
        try:
            download(url, out)
        except Exception as e:
            ok = False
            print(f"[ai] WARNING: could not download {fn}: {e}")
            print("[ai] The assistant will be unavailable until this succeeds; rerun the launcher to resume.")
    if ok:
        print("[ai] assistant model ready.")
    return 0   # never block Forge from launching


if __name__ == "__main__":
    sys.exit(main())
