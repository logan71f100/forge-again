#!/usr/bin/env bash
# =============================================================================
# forge-again launcher (macOS, Apple Silicon or Intel) -- fully self-contained.
# NOTE: this fork is developed and tested on Windows/NVIDIA. The macOS path
# (PyTorch MPS backend, no CUDA) is provided best-effort and is UNTESTED.
# First run downloads a portable Python 3.12, builds the venv and installs
# PyTorch (MPS/CPU from PyPI) + all dependencies.
# Usage:  ./start-macos.sh [sd|xl|flux]
# Env:    FORGE_MODELS_DIR  models folder   (default: ./models)
#         FORGE_PORT        UI port         (default: 7860)
# =============================================================================
set -u
cd "$(dirname "$0")" || exit 1

FORGE_MODELS_DIR="${FORGE_MODELS_DIR:-$PWD/models}"
FORGE_PORT="${FORGE_PORT:-7860}"

ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    PYARCH="aarch64-apple-darwin"
else
    PYARCH="x86_64-apple-darwin"
fi
PYURL="https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.12.13+20260718-${PYARCH}-install_only.tar.gz"

fail() {
    echo
    echo "[bootstrap] SETUP FAILED. Fix the error above and re-run. Partial state"
    echo "[bootstrap] is kept so a re-run resumes where it stopped."
    exit 1
}

# ------------------------------------------------------------- mode select
MODE="${1:-}"
[ -z "$MODE" ] && [ -f current_mode.txt ] && MODE="$(cat current_mode.txt)"
# no argument and no saved mode: default to sd -- modes switch in one click
# from the UI once running, so there is no reason to block on a menu here
[ -z "$MODE" ] && MODE=sd
case "$MODE" in
    1|sd|SD)
        echo "  -> SD 1.5 mode"; MODENAME=sd
        export REPLACER_DEF_SAMPLER="DPM++ 2M" REPLACER_DEF_SCHEDULER="Karras"
        export REPLACER_DEF_WIDTH=512 REPLACER_DEF_HEIGHT=512 REPLACER_DEF_STEPS=25
        export REPLACER_DEF_CFG=7.0 REPLACER_DEF_DENOISE=0.5 REPLACER_FLUX_GUIDANCE=3.5 ;;
    3|flux|Flux|FLUX)
        echo "  -> Flux Fill mode"; MODENAME=flux
        export REPLACER_DEF_SAMPLER="Euler" REPLACER_DEF_SCHEDULER="Simple"
        export REPLACER_DEF_WIDTH=1024 REPLACER_DEF_HEIGHT=1024 REPLACER_DEF_STEPS=20
        export REPLACER_DEF_CFG=1.0 REPLACER_DEF_DENOISE=1.0 REPLACER_FLUX_GUIDANCE=30 ;;
    *)
        echo "  -> SDXL mode"; MODENAME=xl
        export REPLACER_DEF_SAMPLER="DPM++ 2M" REPLACER_DEF_SCHEDULER="Karras"
        export REPLACER_DEF_WIDTH=1024 REPLACER_DEF_HEIGHT=1024 REPLACER_DEF_STEPS=25
        export REPLACER_DEF_CFG=5.0 REPLACER_DEF_DENOISE=0.75 REPLACER_FLUX_GUIDANCE=3.5 ;;
esac
export REPLACER_DEF_MASK_EXPAND=15 REPLACER_DEF_BOX_THRESHOLD=0.35
export REPLACER_DEF_MASK_BLUR=6 REPLACER_DEF_PADDING=48 REPLACER_DEF_FILL=original

# --------------------------------------------------------------- bootstrap
if [ ! -x python/bin/python3 ]; then
    echo "[bootstrap] Downloading portable Python 3.12 (${PYARCH}) ..."
    curl -L --fail -o _py.tar.gz "$PYURL" || fail
    echo "[bootstrap] Extracting Python ..."
    rm -rf _pytmp && mkdir _pytmp
    tar -xzf _py.tar.gz -C _pytmp || fail
    mv _pytmp/python python || fail
    rm -rf _pytmp _py.tar.gz
fi

if [ ! -x venv/bin/python ]; then
    echo "[bootstrap] Creating virtual environment ..."
    python/bin/python3 -m venv venv || fail
fi

if [ ! -f venv/.deps_installed ]; then
    echo "[bootstrap] Upgrading pip ..."
    venv/bin/python -m pip install --upgrade pip || fail
    echo "[bootstrap] Installing PyTorch (MPS/CPU build from PyPI) ..."
    venv/bin/python -m pip install torch==2.13.0 torchvision==0.28.0 || fail
    echo "[bootstrap] Installing requirements (CUDA pins filtered for macOS) ..."
    grep -vE '^(--extra-index-url|torch==|torchvision==)' requirements_versions.txt > _requirements_macos.txt
    venv/bin/python -m pip install --no-build-isolation -r _requirements_macos.txt || fail
    rm -f _requirements_macos.txt
    echo ok > venv/.deps_installed
    echo "[bootstrap] Environment ready."
fi

# --------------------------------------------------------------- configure
venv/bin/python set_mode.py "$MODENAME"

# AI assistant vision model (~18GB, first run only; set FORGE_NO_LLM=1 to skip).
# NOTE: the bundled llama-server is a CUDA/Windows build; on macOS point
# forge_ai_server_bin at a llama.cpp server you built for Metal (or set
# FORGE_NO_LLM=1 to skip the assistant).
venv/bin/python download_llm.py

# extra launch arguments: one line in extra-args.txt (optional, next to this
# script) and/or the FORGE_EXTRA_ARGS environment variable
EXTRA_ARGS=""
[ -f extra-args.txt ] && EXTRA_ARGS="$(head -n 1 extra-args.txt)"

# open the UI in the default browser once it is up (set FORGE_NO_BROWSER=1
# to suppress, e.g. for headless/service use)
AUTOLAUNCH="--autolaunch"
[ -n "${FORGE_NO_BROWSER:-}" ] && AUTOLAUNCH=""

export SD_WEBUI_RESTART=1
export HF_HOME="$FORGE_MODELS_DIR/hf-cache"
export PYTORCH_ENABLE_MPS_FALLBACK=1
while :; do
    CKMODE="$(cat current_mode.txt)"
    venv/bin/python launch.py --listen --port "$FORGE_PORT" --api \
        --no-half-vae --disable-xformers --skip-python-version-check --skip-torch-cuda-test \
        --ckpt-dir "$FORGE_MODELS_DIR/checkpoints/$CKMODE" \
        --lora-dir "$FORGE_MODELS_DIR/Lora" \
        --vae-dir "$FORGE_MODELS_DIR/VAE" \
        --text-encoder-dir "$FORGE_MODELS_DIR/text_encoder" \
        --esrgan-models-path "$FORGE_MODELS_DIR/ESRGAN" \
        $AUTOLAUNCH $EXTRA_ARGS ${FORGE_EXTRA_ARGS:-}
    # UI-triggered restarts relaunch through this loop: mark them so the
    # server does not open another browser tab each time
    if [ -f tmp/restart ]; then rm -f tmp/restart; export SD_WEBUI_RESTARTING=1; continue; fi
    break
done
