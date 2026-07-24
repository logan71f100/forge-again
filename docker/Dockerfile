# =============================================================================
# forge-again container image (Linux, NVIDIA)
#
# Two stages. The first only runs when you ask for the AI assistant, and builds
# a patched llama-server from forge-llm/patches/ so Linux gets the same
# /sleep + /wake VRAM hibernate the bundled Windows binary has. The second is
# the runtime image: dependencies are installed at BUILD time so the container
# starts in seconds instead of bootstrapping 12 GB on first run.
#
# Build (no assistant, the default -- smaller and much faster):
#   docker build -f docker/Dockerfile -t forge-again .
#
# Build with the assistant (adds a CUDA compile stage, and a large image):
#   docker build -f docker/Dockerfile --build-arg WITH_ASSISTANT=1 -t forge-again:assistant .
# =============================================================================

ARG WITH_ASSISTANT=0

# ----------------------------------------------------------------------------
# Stage 1: build the patched llama-server.
#
# This is split into llama-build-1 (real) and llama-build-0 (no-op) which are
# selected by WITH_ASSISTANT below. A plain `COPY --from=<stage>` would force
# the CUDA compile to run on EVERY build regardless of the flag, since Docker
# builds any stage that is referenced -- this indirection avoids that.
# ----------------------------------------------------------------------------
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS llama-build-1

# The upstream revision forge-llm/patches/ is cut against. Keep in sync with
# the base revision documented in forge-llm/patches/README.md.
ARG LLAMA_COMMIT=33ca0dcb9d78c7c3a3b543db4c5fc9182abfe519
# 75=Turing(20xx) 86=Ampere(30xx) 89=Ada(40xx) 120=Blackwell(50xx).
# Semicolon-separate for a multi-arch build, e.g. "75;86;89".
ARG CUDA_ARCH=75

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY forge-llm/patches/ /patches/

# Shallow-fetch just the pinned commit rather than cloning full history.
RUN git init llama.cpp \
    && cd llama.cpp \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth 1 origin ${LLAMA_COMMIT} \
    && git checkout FETCH_HEAD \
    && git apply --verbose /patches/0001-sleep-wake-vram-hibernate.patch

RUN cd llama.cpp \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=ON \
    && cmake --build build --target llama-server -j "$(nproc)"

# Collect the server plus its shared libs into one directory to copy out.
RUN mkdir -p /out \
    && cp llama.cpp/build/bin/llama-server /out/ \
    && find llama.cpp/build -name "*.so*" -exec cp -a {} /out/ \; \
    && ls -la /out

# No-op counterpart: an empty /out, so the runtime stage can copy
# unconditionally without dragging in the CUDA toolchain.
FROM busybox:1.36 AS llama-build-0
RUN mkdir -p /out

# Resolves to llama-build-0 or llama-build-1 depending on the build arg.
ARG WITH_ASSISTANT
FROM llama-build-${WITH_ASSISTANT} AS llama-out

# ----------------------------------------------------------------------------
# Stage 2: runtime
# ----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ARG WITH_ASSISTANT=0
ENV WITH_ASSISTANT=${WITH_ASSISTANT}

# Runtime OS deps. libgl1/libglib2.0-0 are needed by opencv (used by ControlNet
# preprocessors and Replacer); git is used by Forge's extension machinery.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies FIRST, from just the requirements file, so edits to the
# source don't invalidate the (very expensive) dependency layer.
COPY requirements_versions.txt /app/requirements_versions.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        torch==2.13.0+cu126 torchvision==0.28.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install --no-cache-dir --no-build-isolation \
        -r requirements_versions.txt

# Extension dependencies. Forge normally installs these at runtime through each
# extension's install.py, but the container runs with --skip-install (correct
# for a prebuilt image), so they have to be installed at build time instead.
#
# Copied on their own, ahead of the source tree, so editing code doesn't
# invalidate this layer.
#
# forge_space_idm_vton is deliberately omitted: it wants basicsr, which does not
# build against modern torchvision, and it isn't present in a working native
# install either -- the space simply stays unavailable, as it does on Windows.
COPY extensions/sd-webui-segment-anything/requirements.txt      /app/_extreq/segment-anything.txt
COPY extensions-builtin/sd_forge_controlnet/requirements.txt    /app/_extreq/controlnet.txt
COPY extensions-builtin/forge_legacy_preprocessors/requirements.txt /app/_extreq/preprocessors.txt
RUN python -m pip install --no-cache-dir --no-build-isolation \
        -r /app/_extreq/segment-anything.txt \
        -r /app/_extreq/controlnet.txt \
        -r /app/_extreq/preprocessors.txt \
    # onnxruntime declares protobuf>=4.25.8, but protobuf is capped <4 by
    # open-clip-torch (SDXL text encoder). Re-pin to the tested version, which
    # is exactly how a working native install ends up.
    && python -m pip install --no-cache-dir "protobuf==3.20.3" \
    # joblib is imported by the builtin soft-inpainting script but declared in
    # no requirements file -- on a native install it arrives transitively, so
    # the container has to ask for it explicitly.
    && python -m pip install --no-cache-dir joblib \
    && rm -rf /app/_extreq

# Now the application itself.
COPY . /app

# The bundled llama-server is a Windows CUDA build and is useless here; drop it
# so it isn't shipped inside a Linux image. The patch source stays.
RUN find /app/forge-llm -maxdepth 1 -type f -delete 2>/dev/null || true

# Bring in the Linux llama-server only when the assistant was requested.
COPY --from=llama-out /out/ /tmp/llama-out/
RUN if [ "$WITH_ASSISTANT" = "1" ]; then \
        mkdir -p /app/forge-llm && cp -a /tmp/llama-out/. /app/forge-llm/ && \
        chmod +x /app/forge-llm/llama-server && \
        echo "assistant: llama-server installed"; \
    else \
        echo "assistant: disabled (build with --build-arg WITH_ASSISTANT=1 to enable)"; \
    fi; \
    rm -rf /tmp/llama-out

# llama-server needs cuBLAS. Rather than shipping it, reuse the copy that comes
# with the torch wheel -- the same trick the Windows launcher uses.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/torch/lib

# Defaults. FORGE_NO_LLM is derived from the build arg so an image built without
# the assistant never attempts the ~18 GB model download.
ENV FORGE_PORT=7860 \
    FORGE_MODELS_DIR=/app/models \
    FORGE_NO_BROWSER=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf-cache
RUN if [ "$WITH_ASSISTANT" != "1" ]; then echo "FORGE_NO_LLM=1" >> /etc/environment; fi

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 7860
VOLUME ["/app/models", "/app/outputs"]

# Fails until the UI is actually serving, so `docker compose ps` tells the truth.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${FORGE_PORT}/internal/ping" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []
