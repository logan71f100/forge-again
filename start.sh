#!/usr/bin/env bash
# =============================================================================
# forge-again launcher (Linux x86_64, AMD Vulkan / ROCm / NVIDIA CUDA / Apple MPS)
# Auto-detects GPU type, installs matching PyTorch, builds/deployes llama.cpp
# binaries for the detected backend.
#
# Usage:  ./start.sh [sd|xl|flux]
# Env:    FORCE_GPU=nvidia|rocm|vulkan|cpu|mps   override auto-detection
# =============================================================================
set -e
cd "$(dirname "$(readlink -f "$0")")"

# ------------------------------------------------------------------- helpers
fail() {
  echo
  echo "[bootstrap] setup failed. fix the error above and re-run. partial"
  echo "[bootstrap] state is kept so a re-run resumes where it stopped." >&2
  exit 1
}

# ------------------------------------------------------------------- gpu detect
detect_gpu() {
  # 1. forced?
  if [ -n "${FORCE_GPU:-}" ]; then
    echo "[bootstrap] Force GPU=$FORCE_GPU"
    GPU="$FORCE_GPU"
    return
  fi

  # 2. NVIDIA – nvidia-smi must exist and talk to a driver.
  if command -v nvidia-smi >/dev/null 2>&1; then
    # try to get a name; failures look like 'not supported' or 'no data'
    local gname
    gname="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" || true
    if [ -n "$gname" ] &&
       ! echo "$gname" | grep -qiE 'not supported|no data|unavailable'; then
      echo "[bootstrap] Detected NVIDIA GPU: $gname"
      GPU=nvidia
      return
    fi
  fi

  # 3. AMD – rocm-smi first; second-best: /opt/rocm exists.
  if command -v rocm-smi >/dev/null 2>&1; then
    if rocm-smi --showall 2>&1 | grep -qiE 'amg|asrock|amd'; then
      echo "[bootstrap] Detected AMD GPU via rocm-smi"
      GPU=rocm
      return
    fi
  fi
  # fallback heuristic – /opt/rocm presence
  if [ -d /opt/rocm ]; then
    echo "[bootstrap] AMD ROCm: /opt/rocm found"
    GPU=rocm
    return
  fi

  # 4. Vulkan (AMD iGPU/dGPU via ggml-vulkan backend)
  if ldconfig -p 2>/dev/null | grep -q libvulkan_radeon; then
    if [ -d /home/logan/llama.cpp ] || [ -d ./llama.cpp ]; then
      echo "[bootstrap] Detected AMD Vulkan (libvulkan_radeon) for ggml-vulkan backend"
      GPU=vulkan
      return
    fi
  fi
  # fallback: any AMD vulkan driver
  if ldconfig -p 2>/dev/null | grep -qE 'libvulkan_(rade|amd)'; then
    echo "[bootstrap] Detected AMD Vulkan GPU (no llama.cpp found, will build)"
    GPU=vulkan
    return
  fi

  # 5. macOS → MPS (Metal)
  if [ "$(uname)" = "Darwin" ]; then
    echo "[bootstrap] macOS → MPS"
    GPU=mps
    return
  fi

  echo "[bootstrap] No GPU detected; falling back to CPU." >&2
  echo "[bootstrap] Set FORCE_GPU=nvidia|rocm|vulkan|cpu|mps to override." >&2
  GPU=cpu
}

detect_gpu

# ------------------------------------------------------------------- model / mode select
MODE="${1:-}"
[ -z "$MODE" ] && [ -f current_mode.txt ] && MODE="$(cat current_mode.txt)"
[ -z "$MODE" ] && MODE=sd          # default – UI can hot-swap later
case "$MODE" in
  1|sd|SD)
    echo "  ->  SD 1.5 mode"; MODENAME=sd
    export REPLACER_DEF_SAMPLER="DPM++ 2M" REPLACER_DEF_SCHEDULER="Karras"
    export REPLACER_DEF_WIDTH=512  REPLACER_DEF_HEIGHT=512 REPLACER_DEF_STEPS=25
    export REPLACER_DEF_CFG=7  REPLACER_DEF_DENOISE=0.5 REPLACER_FLUX_GUIDANCE=3.5
    ;;
  3|flux|Flux|FLUX)
    echo "  ->  Flux Fill mode"; MODENAME=flux
    export REPLACER_DEF_SAMPLER="Euler" REPLACER_DEF_SCHEDULER="Simple"
    export REPLACER_DEF_WIDTH=1024 REPLACER_DEF_HEIGHT=1024 REPLACER_DEF_STEPS=20
    export REPLACER_DEF_CFG=1  REPLACER_DEF_DENOISE=1    REPLACER_FLUX_GUIDANCE=30
    ;;
  *)
    echo "  ->  SDXL mode"; MODENAME=xl
    export REPLACER_DEF_SAMPLER="DPM++ 2M" REPLACER_DEF_SCHEDULER="Karras"
    export REPLACER_DEF_WIDTH=1024 REPLACER_DEF_HEIGHT=1024 REPLACER_DEF_STEPS=25
    export REPLACER_DEF_CFG=5  REPLACER_DEF_DENOISE=0.75 REPLACER_FLUX_GUIDANCE=3.5
    ;;
esac
export REPLACER_DEF_MASK_EXPAND=15 REPLACER_DEF_BOX_THRESHOLD=0.35
export REPLACER_DEF_MASK_BLUR=6   REPLACER_DEF_PADDING=48  REPLACER_DEF_FILL=original

# ------------------------------------------------------------------- ROCm env vars  (set unconditionally when GPU=rocm)
if [ "$GPU" = "rocm" ]; then
  # HSA graph rebuild fix for iGPUs (Strix Point RDNA3.5 / gfx115x)
  # gfx1150 (0x5585) is accepted when gfx1151 (0x5586) is in the table
  export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"

  # MIOpen: FAST skips exhaustive conv searches that often hit MIOPEN_ENO_CONFIG on
  # iGPUs / new archs. Immediate-mode heuristic kernels are ~16 s/512 px and
  # proven working.
  export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"

  # Shared-memory allocator tuning for AMD iGPU, ~11 GB.
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.9,max_split_size_mb:512}"

  # Disable DMA engine (stability for iGPU)
  export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"

  # Limit parallelism to avoid thrashing
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
fi

# ------------------------------------------------------------------- Vulkan env vars (set when GPU=vulkan — ggml-vulkan backend)
if [ "$GPU" = "vulkan" ]; then
  # Vulkan shader cache directory
  export VULKAN_SHADER_CACHE_DIR="${VULKAN_SHADER_CACHE_DIR:-$FORGE_MODELS_DIR/shader-cache}"
  mkdir -p "$VULKAN_SHADER_CACHE_DIR" 2>/dev/null || true

  # GGML Vulkan memory mode — balanced for iGPU
  export GGML_VK_MEMORY_MODEL="${GGML_VK_MEMORY_MODEL:-host}"

  # Limit Vulkan device memory — leave room for Forge (diffusion) models
  export GGML_VK_MEMORY_RATIO="${GGML_VK_MEMORY_RATIO:-0.75}"

  # Disable validation layers in production
  export VK_LAYER_PATH=""
fi

# ------------------------------------------------------------------- build & deploy llama.cpp binaries
LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-/home/logan/llama.cpp}"
LLAMA_CPP_BUILD="${LLAMA_CPP_BUILD:-/tmp/forge-llama-build}"
LLAMA_CPP_DEPLOY="${LLAMA_CPP_DEPLOY:-forge-llm}"

build_llama() {
  local backend="$1"    # vulkan | rocm | cuda
  local cmake_cfg=""
  local build_dir="$LLAMA_CPP_BUILD/$backend"
  local deploy_dir="$LLAMA_CPP_DEPLOY/$backend"

  mkdir -p "$build_dir" "$deploy_dir"

  case "$backend" in
    vulkan) cmake_cfg="-DGGML_VULKAN=ON" ;;
    rocm)   cmake_cfg="-DGGML_HIP=ON" ;;
    cuda)   cmake_cfg="-DGGML_CUDA=ON" ;;
  esac

  # skip if deployed binaries already exist (fast re-launch)
  if [ -f "$deploy_dir/llama-server" ] && [ -f "$deploy_dir/libggml-$backend.so" ]; then
    echo "[bootstrap] $backend: llama.cpp binaries already deployed"
    return 0
  fi

  if [ ! -d "$LLAMA_CPP_SRC" ]; then
    echo "[bootstrap] WARNING: $backend llama.cpp source not found at $LLAMA_CPP_SRC — LLM AI assistant will be unavailable" >&2
    return 1
  fi

  echo "[bootstrap] building llama.cpp for $backend backend ..."
  cmake -B "$build_dir" -DCMAKE_BUILD_TYPE=Release $cmake_cfg "$LLAMA_CPP_SRC" || fail
  cmake --build "$build_dir" --target llama-server --target all -j "$(nproc 2>/dev/null || echo 4)" || fail

  echo "[bootstrap] deploying llama.cpp $backend binaries → $deploy_dir"
  cp "$build_dir/examples/llama-server" "$deploy_dir/" 2>/dev/null || true
  cp "$build_dir/bin/libggml-$backend.so" "$deploy_dir/" 2>/dev/null || true
  cp "$build_dir/bin/libggml-base.so" "$deploy_dir/" 2>/dev/null || true
  cp "$build_dir/bin/libggml-cpu.so" "$deploy_dir/" 2>/dev/null || true
  cp "$build_dir/bin/libllama.so" "$deploy_dir/" 2>/dev/null || true
  cp "$build_dir/bin/libllama-common.so" "$deploy_dir/" 2>/dev/null || true
  cp "$build_dir/bin/libllama-server-impl.so" "$deploy_dir/" 2>/dev/null || true
  # Make binaries executable
  chmod +x "$deploy_dir/llama-server" 2>/dev/null || true
  # Update symlinks so ldconfig can find the .so files at runtime
  if [ -f "$deploy_dir/libggml-base.so" ]; then
    ( cd "$deploy_dir" && ln -sf libggml-base.so libggml-base.so.0 2>/dev/null || true )
  fi
  if [ -f "$deploy_dir/libggml-$backend.so" ]; then
    ( cd "$deploy_dir" && ln -sf "libggml-$backend.so" "libggml-$backend.so.0" 2>/dev/null || true )
  fi
  if [ -f "$deploy_dir/libllama.so" ]; then
    ( cd "$deploy_dir" && ln -sf libllama.so libllama.so.0 2>/dev/null || true )
  fi
  if [ -f "$deploy_dir/libllama-common.so" ]; then
    ( cd "$deploy_dir" && ln -sf libllama-common.so libllama-common.so.0 2>/dev/null || true )
  fi
  if [ -f "$deploy_dir/libllama-server-impl.so" ]; then
    ( cd "$deploy_dir" && ln -sf libllama-server-impl.so libllama-server-impl.so.0 2>/dev/null || true )
  fi
  echo "[bootstrap] $backend: llama.cpp binaries deployed successfully"
  return 0
}

# Build & deploy for all GPU backends that matter
case "$GPU" in
  vulkan) build_llama vulkan ;;
  rocm)   build_llama rocm ; build_llama vulkan ;;  # vulkan is fallback
  cpu)    build_llama cuda ;;  # fallback to cuda for AI assistant
esac

# ------------------------------------------------------------------- bootstrap (portable python, venv)
# launch.py needs `git` to clone assets / huggingface_guess / BLIP. Fail
# before a half-formed traceback if git is missing.
if ! command -v git >/dev/null 2>&1; then
  echo "[bootstrap] git not found; install it and re-run. example:"
  echo "[bootstrap]   apt install  -y  git    # Debian/Ubuntu"
  echo "[bootstrap]   dnf install -y  git    # RHEL/Fedora"
  echo "[bootstrap]   pacman -S git                      # Arch"
  exit 1
fi

PYURL="https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.12.13+20260718-x86_64-unknown-linux-gnu-install_only.tar.gz"
if [ ! -x python/bin/python3 ]; then
  echo "[bootstrap] downloading portable python 3.12 ..."
  curl -L --fail -o _py.tar.gz "$PYURL" || fail
  echo "[bootstrap] extracting python ..."
  rm -rf _pytmp; mkdir _pytmp
  tar -xzf _py.tar.gz -C _pytmp || fail
  mv _pytmp/python python || fail
  rm -rf _pytmp _py.tar.gz
fi
if [ ! -x venv/bin/python ]; then
  echo "[bootstrap] creating virtual environment ..."
  python/bin/python3 -m venv venv || fail
fi

# Stamp tracks requirements_versions.txt hash so deps get re-installed on updates.
REQHASH="$(venv/bin/python -c "import hashlib;print(hashlib.sha256(open('requirements_versions.txt','rb').read()).hexdigest()[:16])" 2>/dev/null || true)"
if [ ! -f venv/.deps_installed ] || [ -z "$REQHASH" ] || ! grep -qx "$REQHASH" venv/.deps_installed 2>/dev/null; then
  [ -f venv/.deps_installed ] && echo "[bootstrap] requirements changed since last install; updating ..."
  echo "[bootstrap] upgrading pip ..."
  venv/bin/python -m pip install --upgrade pip || fail

  # Install third-party torch (platform) BEFORE requirements so the latter
  # satisfies `torch`, `torchvision` pins instead of fighting pip's solver.
  echo "[bootstrap] installing torch ($GPU) ..."
  case "$GPU" in
    rocm)
      venv/bin/python -m pip install \
        --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
        --extra-index-url https://pypi.org/simple \
        numpy==1.26.2 \
        torch==2.7.1+rocm6.1 \
        torchvision==0.22.1+rocm6.1 || fail
      ;;
    nvidia)
      venv/bin/python -m pip install \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.13.0+cu126 \
        torchvision==0.28.0+cu126 || fail
      ;;
    vulkan)
      # Vulkan uses ggml-vulkan backend (no PyTorch CUDA); use CPU/MPS torch
      venv/bin/python -m pip install torch torchvision || fail
      ;;
    mps|cpu)
      venv/bin/python -m pip install torch torchvision || fail
      ;;
  esac

  echo "[bootstrap] installing requirements ..."
  venv/bin/python -m pip install --no-build-isolation -r requirements_versions.txt || fail
  echo "$REQHASH" > venv/.deps_installed
  echo "[bootstrap] environment ready."
fi

# ------------------------------------------------------------------- launch args  ($FORGE_MODELS_DIR, etc.)
export HF_HOME="${FORGE_MODELS_DIR:-./models}/hf-cache"

EXTRA_ARGS=""
[ -f extra-args.txt ] && EXTRA_ARGS="$(head -n 1 extra-args.txt)"

AUTOLAUNCH="--autolaunch"
[ -n "${FORGE_NO_BROWSER:-}" ]   && AUTOLAUNCH=""

# cuda-malloc is a no-op on ROCm; still safe to pass but we omit it to be
# honest about what it does.  --skip-torch-cuda-test is needed when the
# torch wheel is not cuda-enabled (ROCm nightly, PyPI cpu/mps).
case "$GPU" in
  nvidia) LAUNCH_FLAGS="--cuda-malloc" ; _skip_cuda_test=false ;;
  rocm)   LAUNCH_FLAGS=""             ; _skip_cuda_test=true  ;;
  vulkan) LAUNCH_FLAGS=""             ; _skip_cuda_test=true  ;;
  mps)    LAUNCH_FLAGS=""             ; _skip_cuda_test=true  ;;
  cpu)    LAUNCH_FLAGS=""             ; _skip_cuda_test=true  ;;
esac

# ------------------------------------------------------------------- run / restart loop
while :; do
  CKMODE="$(cat current_mode.txt 2>/dev/null || echo xl)"
  venv/bin/python launch.py \
    --listen --port 7860 \
    --api \
    ${LAUNCH_FLAGS:---cuda-malloc} \
    --no-half-vae --disable-xformers \
    --skip-python-version-check \
    --skip-torch-cuda-test="${_skip_cuda_test:-true}" \
    --ckpt-dir    "${FORGE_MODELS_DIR:-./models}/checkpoints/$CKMODE" \
    --lora-dir    "${FORGE_MODELS_DIR:-./models}/Lora" \
    --vae-dir     "${FORGE_MODELS_DIR:-./models}/VAE" \
    --text-encoder-dir "${FORGE_MODELS_DIR:-./models}/text_encoder" \
    --esrgan-models-path "${FORGE_MODELS_DIR:-./models}/ESRGAN" \
    $AUTOLAUNCH $EXTRA_ARGS ${FORGE_EXTRA_ARGS:-}
  # restart loop
  if [ -f tmp/restart ]; then rm -f tmp/restart; break; fi
  break
done

# =============================================================================