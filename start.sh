#!/usr/bin/env bash
# =============================================================================
# forge-again launcher (Linux x86_64, AMD ROCm/Vulkan / NVIDIA CUDA / Apple MPS)
# Auto-detects the GPU, installs the matching PyTorch build, and compiles the
# AI assistant's patched llama.cpp for the detected backend on first run.
#
# Usage:  ./start.sh [sd|xl|flux]
# Env:    FORCE_GPU=nvidia|rocm|vulkan|cpu|mps   override auto-detection
#         FORGE_PORT        UI port         (default: 7860)
#         FORGE_MODELS_DIR  external models folder (default: ./models)
#         FORGE_NO_BROWSER  set to skip auto-opening a browser
#         FORGE_NO_LLM      set to skip the AI assistant build + model download
#         TORCH_ROCM_INDEX  override the ROCm wheel index (default: rocm7.1)
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

  # 2. macOS -> MPS. Checked early: the probes below are Linux-only (ldconfig,
  #    rocm-smi) and just waste time on Darwin.
  if [ "$(uname)" = "Darwin" ]; then
    echo "[bootstrap] macOS → MPS"
    GPU=mps
    return
  fi

  # 3. NVIDIA – nvidia-smi must exist and talk to a driver.
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

  # 4. AMD. Detect the CARD, not the system ROCm install: the PyTorch ROCm
  #    wheels bundle their own ROCm runtime, so an AMD GPU can run torch on ROCm
  #    with no rocm-smi and no /opt/rocm anywhere on the machine. Keying off
  #    those tools (as this used to) misclassified every stock AMD laptop as
  #    "no ROCm" and sent it down a CPU-only path.
  #    The PCI class check covers VGA (0300) and Display (0380) controllers --
  #    AMD iGPUs like Strix report the latter.
  if lspci -nn 2>/dev/null | grep -iE '\[03[0-9a-f]{2}\]' | grep -q '\[1002:'; then
    echo "[bootstrap] Detected AMD GPU: $(lspci -nn 2>/dev/null | grep -iE '\[03[0-9a-f]{2}\]' | grep '\[1002:' | sed 's/.*\[AMD\/ATI\] //; s/ \[1002:.*//' | head -1)"
    GPU=rocm
    return
  fi
  # legacy fallbacks: an AMD box where lspci is unavailable
  if command -v rocm-smi >/dev/null 2>&1; then
    if rocm-smi --showid 2>/dev/null | grep -qiE '^GPU\[[0-9]+\]'; then
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

  # 5. Vulkan (AMD iGPU/dGPU via the ggml-vulkan backend). Presence of the
  #    driver is the whole test -- llama.cpp is fetched and built on demand, so
  #    it must NOT be a precondition for detecting the GPU.
  if ldconfig -p 2>/dev/null | grep -qE 'libvulkan_(radeon|amd)'; then
    echo "[bootstrap] Detected AMD Vulkan driver → ggml-vulkan backend"
    GPU=vulkan
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

# ------------------------------------------------------------------- build & deploy llama.cpp (AI assistant)
# The assistant needs a llama-server carrying our /sleep + /wake hibernate patch
# (forge-llm/patches). Windows ships a prebuilt CUDA binary because Windows boxes
# rarely have a compiler; on Linux/macOS a toolchain is one package away, and it
# is the only way to get a ROCm or Vulkan build for arbitrary hardware -- so we
# fetch the pinned source, apply the patch and build on demand.
#
# Everything here is BEST EFFORT. The assistant is optional, so a missing
# compiler or a failed build must warn and let Forge start anyway -- never `fail`.
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-33ca0dcb9d78c7c3a3b543db4c5fc9182abfe519}"   # base revision the patch applies to
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-forge-llm/src/llama.cpp}"
LLAMA_CPP_BUILD="${LLAMA_CPP_BUILD:-forge-llm/src/build}"
LLAMA_CPP_DEPLOY="${LLAMA_CPP_DEPLOY:-forge-llm}"
LLAMA_PATCH="forge-llm/patches/0001-sleep-wake-vram-hibernate.patch"

# Cache key: a deployed build is reused unless the backend, the upstream commit
# or the patch itself changes. That is the "only rebuild when it makes sense"
# part -- a normal relaunch does no compiling at all.
llama_stamp_now() {
  local patch_hash
  patch_hash="$(sha256sum "$LLAMA_PATCH" 2>/dev/null | cut -c1-16)"
  echo "$1|$LLAMA_CPP_COMMIT|${patch_hash:-nopatch}"
}

llama_have_tools() {
  command -v git >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1 &&
    { command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || command -v clang >/dev/null 2>&1; }
}

# Fetch the pinned source once, patched. Returns non-zero if it can't.
llama_prepare_src() {
  if [ -f "$LLAMA_CPP_SRC/.forge-patched" ] &&
     [ "$(cat "$LLAMA_CPP_SRC/.forge-patched" 2>/dev/null)" = "$LLAMA_CPP_COMMIT" ]; then
    return 0
  fi
  echo "[bootstrap] fetching llama.cpp @ ${LLAMA_CPP_COMMIT:0:12} ..."
  mkdir -p "$(dirname "$LLAMA_CPP_SRC")" || return 1
  if [ ! -d "$LLAMA_CPP_SRC/.git" ]; then
    rm -rf "$LLAMA_CPP_SRC"
    git clone --filter=blob:none "$LLAMA_CPP_REPO" "$LLAMA_CPP_SRC" >/dev/null 2>&1 || return 1
  fi
  ( cd "$LLAMA_CPP_SRC" &&
    git fetch --depth 1 origin "$LLAMA_CPP_COMMIT" >/dev/null 2>&1 &&
    git checkout -q --force "$LLAMA_CPP_COMMIT" &&
    git reset -q --hard "$LLAMA_CPP_COMMIT" &&
    git clean -qfd ) || return 1
  echo "[bootstrap] applying sleep/wake hibernate patch ..."
  ( cd "$LLAMA_CPP_SRC" && git apply "$OLDPWD/$LLAMA_PATCH" ) || {
    echo "[bootstrap] WARNING: hibernate patch did not apply to ${LLAMA_CPP_COMMIT:0:12}" >&2
    return 1
  }
  echo "$LLAMA_CPP_COMMIT" > "$LLAMA_CPP_SRC/.forge-patched"
  return 0
}

build_llama() {
  local backend="$1"    # vulkan | rocm | cuda | metal
  local build_dir="$LLAMA_CPP_BUILD/$backend"
  local deploy_dir="$LLAMA_CPP_DEPLOY/$backend"
  local stamp="$deploy_dir/.build-stamp"
  local want; want="$(llama_stamp_now "$backend")"

  # up to date? nothing to do.
  if [ -x "$deploy_dir/llama-server" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$want" ]; then
    echo "[bootstrap] $backend: llama-server up to date"
    return 0
  fi

  if ! llama_have_tools; then
    echo "[bootstrap] NOTE: git + cmake + a C compiler are needed to build the AI assistant's" >&2
    echo "[bootstrap]       llama-server ($backend). Skipping it; Forge will start normally." >&2
    return 1
  fi

  local cmake_cfg
  case "$backend" in
    vulkan) cmake_cfg="-DGGML_VULKAN=ON" ;;
    rocm)   cmake_cfg="-DGGML_HIP=ON" ;;
    cuda)   cmake_cfg="-DGGML_CUDA=ON" ;;
    metal)  cmake_cfg="-DGGML_METAL=ON" ;;
    *)      echo "[bootstrap] unknown llama backend '$backend'" >&2; return 1 ;;
  esac

  llama_prepare_src || {
    echo "[bootstrap] NOTE: could not prepare patched llama.cpp source; skipping the AI assistant build." >&2
    return 1
  }

  echo "[bootstrap] building llama-server ($backend) — first time only, a few minutes ..."
  # Only the server target: LLAMA_BUILD_EXAMPLES/TESTS off keeps this from
  # compiling the whole example suite we never ship.
  if ! cmake -B "$build_dir" -DCMAKE_BUILD_TYPE=Release $cmake_cfg \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=ON "$LLAMA_CPP_SRC" >/dev/null; then
    echo "[bootstrap] WARNING: cmake configure failed for $backend — AI assistant unavailable." >&2
    return 1
  fi
  if ! cmake --build "$build_dir" --target llama-server -j "$(nproc 2>/dev/null || echo 4)"; then
    echo "[bootstrap] WARNING: llama-server ($backend) build failed — AI assistant unavailable." >&2
    return 1
  fi

  # Deploy: the binary plus the shared libs it links against. Layout differs
  # between llama.cpp revisions, so search rather than assume a path.
  mkdir -p "$deploy_dir"
  local server_bin
  server_bin="$(find "$build_dir" -name llama-server -type f -perm -u+x 2>/dev/null | head -1)"
  if [ -z "$server_bin" ]; then
    echo "[bootstrap] WARNING: build produced no llama-server binary — AI assistant unavailable." >&2
    return 1
  fi
  cp -f "$server_bin" "$deploy_dir/" || return 1
  find "$build_dir" -name '*.so*' -type f -exec cp -f {} "$deploy_dir/" \; 2>/dev/null || true
  chmod +x "$deploy_dir/llama-server" 2>/dev/null || true

  # soname symlinks (libfoo.so.0 -> libfoo.so) so the loader resolves them.
  ( cd "$deploy_dir" && for so in *.so; do
      [ -e "$so" ] && ln -sf "$so" "$so.0" 2>/dev/null || true
    done ) 2>/dev/null || true

  echo "$want" > "$stamp"
  echo "[bootstrap] $backend: llama-server ready → $deploy_dir"
  return 0
}

# Build only for the detected backend. Each is optional: a failure warns above
# and leaves Forge to start without the assistant.
if [ "${FORGE_NO_LLM:-}" = "1" ]; then
  echo "[bootstrap] FORGE_NO_LLM=1 — skipping the AI assistant's llama.cpp build"
else
  case "$GPU" in
    vulkan) build_llama vulkan || true ;;
    # AMD: Vulkan FIRST for llama.cpp. ggml-vulkan needs only the Mesa driver
    # every AMD box already has, whereas ggml-hip needs a full system ROCm
    # (hipcc, headers) that the self-contained torch wheels do NOT provide --
    # so on a stock AMD laptop the HIP build fails and Vulkan is what works.
    # (torch still goes the ROCm route; the two backends are independent.)
    rocm)   build_llama vulkan || build_llama rocm || true ;;
    nvidia) build_llama cuda   || true ;;
    mps)    build_llama metal  || true ;;
    # cpu: no GPU backend to build; the assistant would be unusably slow, so skip.
  esac
fi

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
  # NOTE: $GPU picks the llama.cpp backend for the AI assistant AND the torch
  # build, but they are NOT the same decision. PyTorch has no usable Vulkan
  # compute backend, so "vulkan" can only ever mean CPU-only image generation --
  # on an AMD card the GPU answer for torch is always ROCm. Keep the versions
  # identical to requirements_versions.txt (torch 2.13.0 / torchvision 0.28.0)
  # so the requirements install below is satisfied instead of fighting pip.
  echo "[bootstrap] installing torch ($GPU) ..."
  case "$GPU" in
    rocm)
      # Official ROCm wheels: self-contained (they bundle the ROCm runtime), so
      # no system ROCm install is required. Verified present for cp312 at
      # rocm7.1. Override the level with TORCH_ROCM_INDEX for other cards.
      venv/bin/python -m pip install \
        --index-url "${TORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.1}" \
        torch==2.13.0 \
        torchvision==0.28.0 || fail
      ;;
    nvidia)
      venv/bin/python -m pip install \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.13.0+cu126 \
        torchvision==0.28.0+cu126 || fail
      ;;
    vulkan|mps|cpu)
      # CPU wheels, explicitly. A bare `pip install torch` pulls the CUDA build
      # from PyPI, which aborts on any non-NVIDIA box with a _preload_cuda_deps
      # error -- that killed the whole bootstrap on an AMD laptop.
      if [ "$GPU" = "mps" ]; then
        venv/bin/python -m pip install torch==2.13.0 torchvision==0.28.0 || fail
      else
        venv/bin/python -m pip install \
          --index-url https://download.pytorch.org/whl/cpu \
          torch==2.13.0 torchvision==0.28.0 || fail
        echo "[bootstrap] NOTE: torch is the CPU build — image generation will be slow."
      fi
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

# cuda-malloc is a no-op on ROCm; we omit it there to be honest about what it
# does. --skip-torch-cuda-test is a store_true flag: it must be passed BARE or
# not at all -- "--skip-torch-cuda-test=true" makes argparse abort with
# "ignored explicit argument", which stopped launch.py on every platform.
# Non-CUDA wheels (cpu/mps/vulkan) can't pass the test, so skip it there.
# ROCm reports torch.cuda.is_available()==True, but skipping is still correct:
# a card ROCm can't drive should surface as a clear runtime error, not a
# bootstrap abort.
case "$GPU" in
  nvidia) LAUNCH_FLAGS="--cuda-malloc" ;;
  *)      LAUNCH_FLAGS="--skip-torch-cuda-test" ;;
esac

# ------------------------------------------------------------------- run / restart loop
while :; do
  CKMODE="$(cat current_mode.txt 2>/dev/null || echo xl)"
  venv/bin/python launch.py \
    --listen --port "${FORGE_PORT:-7860}" \
    --api \
    ${LAUNCH_FLAGS:---cuda-malloc} \
    --no-half-vae --disable-xformers \
    --skip-python-version-check \
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