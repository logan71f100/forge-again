#!/usr/bin/env bash
# =============================================================================
# forge-again container entrypoint.
#
# This mirrors start.sh, minus the bootstrap: the portable Python, the venv and
# every dependency are already baked into the image, so all that's left is
# selecting the mode and running the server (with the same UI-restart loop).
#
# Usage:  docker run ... forge-again [sd|xl|flux]
# =============================================================================
set -u
cd /app

FORGE_MODELS_DIR="${FORGE_MODELS_DIR:-/app/models}"
FORGE_PORT="${FORGE_PORT:-7860}"

# ------------------------------------------------------------- mode select
MODE="${1:-${FORGE_MODE:-}}"
[ -z "$MODE" ] && [ -f current_mode.txt ] && MODE="$(cat current_mode.txt)"
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

# ------------------------------------------------------- assistant on/off
# The assistant is opt-in on Linux. An image built without WITH_ASSISTANT=1 has
# no llama-server, so force FORGE_NO_LLM to avoid the ~18 GB model download.
if [ ! -x /app/forge-llm/llama-server ]; then
    if [ "${FORGE_NO_LLM:-}" != "1" ] && [ -n "${FORGE_ENABLE_LLM:-}" ]; then
        echo "[entrypoint] FORGE_ENABLE_LLM is set, but this image was built"
        echo "[entrypoint] without the assistant. Rebuild with:"
        echo "[entrypoint]   docker build -f docker/Dockerfile --build-arg WITH_ASSISTANT=1 ..."
    fi
    export FORGE_NO_LLM=1
fi

# ---------------------------------------------------------- writable dirs
mkdir -p "$FORGE_MODELS_DIR" /app/outputs /app/tmp 2>/dev/null || true

# --------------------------------------------------------------- configure
python set_mode.py "$MODENAME"

# Vision model for the assistant. download_llm.py is itself config-aware and
# exits immediately when FORGE_NO_LLM=1 or a model is already present.
python download_llm.py

EXTRA_ARGS=""
[ -f extra-args.txt ] && EXTRA_ARGS="$(head -n 1 extra-args.txt)"

export SD_WEBUI_RESTART=1
export HF_HOME="${HF_HOME:-$FORGE_MODELS_DIR/hf-cache}"

# --skip-install: dependencies are baked into the image, so Forge must not try
# to pip-install at runtime (it would be slow and would not persist anyway).
while :; do
    CKMODE="$(cat current_mode.txt)"
    python launch.py --listen --port "$FORGE_PORT" --api --cuda-malloc \
        --no-half-vae --disable-xformers --skip-python-version-check \
        --skip-install \
        --ckpt-dir "$FORGE_MODELS_DIR/checkpoints/$CKMODE" \
        --lora-dir "$FORGE_MODELS_DIR/Lora" \
        --vae-dir "$FORGE_MODELS_DIR/VAE" \
        --text-encoder-dir "$FORGE_MODELS_DIR/text_encoder" \
        --esrgan-models-path "$FORGE_MODELS_DIR/ESRGAN" \
        $EXTRA_ARGS ${FORGE_EXTRA_ARGS:-}
    if [ -f tmp/restart ]; then rm -f tmp/restart; export SD_WEBUI_RESTARTING=1; continue; fi
    break
done
