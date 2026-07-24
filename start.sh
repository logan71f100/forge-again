#!/usr/bin/env bash
# =============================================================================
# forge-again launcher (Linux x86_64, NVIDIA) -- fully self-contained.
# First run downloads a portable Python 3.12, builds the venv and installs
# PyTorch (CUDA 12.6) + all dependencies. Later runs skip finished steps.
# Usage:  ./start.sh [sd|xl|flux]
# Env:    FORGE_MODELS_DIR  models folder   (default: ./models)
#         FORGE_PORT        UI port         (default: 7860)
# =============================================================================
set -u
cd "$(dirname "$(readlink -f "$0")")"

FORGE_MODELS_DIR="${FORGE_MODELS_DIR:-$PWD/models}"
FORGE_PORT="${FORGE_PORT:-7860}"
PYURL="https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.12.13+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz"

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
# launch.py clones three helper repos (assets, huggingface_guess, BLIP) and runs
# `git rev-parse` on them even when they already exist, so git is required, not
# optional. Fail here with something actionable rather than inside a traceback.
if ! command -v git >/dev/null 2>&1; then
    echo "[bootstrap] git was not found on PATH, and Forge needs it to fetch three"
    echo "[bootstrap] helper repositories. Install it and re-run:"
    echo "[bootstrap]   Debian/Ubuntu:  sudo apt install git"
    echo "[bootstrap]   Fedora/RHEL:    sudo dnf install git"
    echo "[bootstrap]   Arch:           sudo pacman -S git"
    exit 1
fi

if [ ! -x python/bin/python3 ]; then
    echo "[bootstrap] Downloading portable Python 3.12 ..."
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

# The stamp records a fingerprint of requirements_versions.txt, not just "ok".
# With a boolean stamp an upgraded install never re-ran pip, so dependency
# changes (including security bumps) only ever reached fresh installs.
REQHASH="$(venv/bin/python -c "import hashlib;print(hashlib.sha256(open('requirements_versions.txt','rb').read()).hexdigest()[:16])" 2>/dev/null || true)"

if [ ! -f venv/.deps_installed ] || [ -z "$REQHASH" ] || ! grep -qx "$REQHASH" venv/.deps_installed 2>/dev/null; then
    [ -f venv/.deps_installed ] && echo "[bootstrap] requirements_versions.txt changed since last install -- updating ..."
    echo "[bootstrap] Upgrading pip ..."
    venv/bin/python -m pip install --upgrade pip || fail
    echo "[bootstrap] Installing PyTorch for CUDA 12.6, large download ..."
    venv/bin/python -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126 || fail
    echo "[bootstrap] Installing requirements ..."
    venv/bin/python -m pip install --no-build-isolation -r requirements_versions.txt || fail
    echo "$REQHASH" > venv/.deps_installed
    echo "[bootstrap] Environment ready."
fi

# Extension installers run on every startup and can pull pinned packages past
# their documented caps (onnxruntime wants protobuf>=4, open-clip-torch caps it
# <4). Restore any drift so the pins are self-correcting.
venv/bin/python check_pins.py

# --------------------------------------------------------------- configure
# Fatal: a failed mode write leaves the UI pointed at the wrong model folder.
venv/bin/python set_mode.py "$MODENAME" || fail

# AI assistant vision model (~18GB, first run only; set FORGE_NO_LLM=1 to skip)
# Deliberately not fatal -- a failed download shouldn't stop Forge starting.
venv/bin/python download_llm.py \
    || echo "[warn] AI assistant model download did not complete; starting without it."

# ControlNet models (~6GB, first run only). Forge fetches preprocessors on demand
# but expects the models themselves to be placed by hand; this gets a working set.
# Runs once -- see download_controlnet.py for the environment variables.
venv/bin/python download_controlnet.py \
    || echo "[warn] ControlNet model download did not complete; starting without them."

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
while :; do
    CKMODE="$(cat current_mode.txt)"
    venv/bin/python launch.py --listen --port "$FORGE_PORT" --api --cuda-malloc \
        --no-half-vae --disable-xformers --skip-python-version-check \
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
