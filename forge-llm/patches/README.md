# llama.cpp sleep/wake VRAM hibernate patch

This directory holds the source patch for the modified `llama-server` that the
[AI assistant extension](../../extensions/forge-ai-assistant) uses. The prebuilt
binary shipped in `forge-llm/` is a **Windows CUDA** build — if you're on Linux or
macOS, or you'd rather build from source than trust a committed binary, apply this
patch to upstream llama.cpp and build it yourself.

## The problem it solves

The assistant runs a vision LLM locally, on the same GPU Forge generates on. A 30B
model holds ~10 GB of VRAM, which is VRAM Forge can't use. Unloading the model
frees the memory but throws away the conversation — reloading it takes tens of
seconds and loses all context.

This patch adds a **hibernate** path instead: it releases *all* VRAM held by the
llama.cpp context (model weights, compute buffers, and the KV cache) by moving the
state into pinned host RAM, and reclaims it on wake. **The KV cache survives the
cycle**, including image embeddings, so the assistant picks up mid-conversation
with nothing lost.

Measured on an RTX 2080 Ti (11 GB) with Qwen3-VL-30B:

| | VRAM | time |
|---|---|---|
| awake | ~9.8 GB | — |
| after `/sleep` | ~0.8 GB | ~1.5 s |
| after `/wake` | ~9.8 GB | ~1.5 s |

## What it adds

- **Public API** in `include/llama.h`:
  ```c
  LLAMA_API void llama_sleep(struct llama_context * ctx);
  LLAMA_API void llama_wake (struct llama_context * ctx);
  ```
- **HTTP endpoints** on `llama-server`: `POST /sleep` and `POST /wake`
- **New files**: `src/llama-hibernate.cpp`, `src/llama-hibernate.h`
- 16 files changed, ~305 insertions

Requests that arrive while the server is asleep are held until it wakes rather than
failing, and `/slots` reports the sleeping state.

## Base revision

The patch applies cleanly to **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
commit `33ca0dcb9d78c7c3a3b543db4c5fc9182abfe519`** (2026-07-07).

## Building

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 33ca0dcb9d78c7c3a3b543db4c5fc9182abfe519
git apply /path/to/forge-again/forge-llm/patches/0001-sleep-wake-vram-hibernate.patch
```

Then configure and build just the server target. **CUDA (Linux or Windows):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j 8
```

Set `-DCMAKE_CUDA_ARCHITECTURES` to your GPU: `75` = Turing (20-series), `86` =
Ampere (30-series), `89` = Ada (40-series), `120` = Blackwell (50-series). The
shipped Windows binary was built with CUDA 12.8 and MSVC 2019 build tools.

**Apple Silicon (Metal)** — the hibernate path is written against a discrete-VRAM
model and is untested on unified memory:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j 8
```

## Pointing Forge at your build

Put the resulting `llama-server` (plus its shared libraries) somewhere on disk, then
set **Settings → AI Assistant → llama-server binary path** to it. Everything else —
model path, context size, provider — is configurable in the same section. If the
binary lacks `/sleep` and `/wake`, the assistant still works; it just falls back to
stopping and restarting the server to free VRAM, which is slower and loses context.

## Applying to a newer llama.cpp

llama.cpp moves fast and these files change often, so the patch will eventually stop
applying cleanly. Try a 3-way merge first:

```bash
git apply -3 0001-sleep-wake-vram-hibernate.patch
```

and resolve conflicts by hand. The change is small and self-contained: most of it is
the new `llama-hibernate.{cpp,h}` pair, and the rest is hooks into the context,
model, and KV-cache teardown/restore paths plus the two server routes.

## License

llama.cpp is MIT-licensed; this patch is a derivative of it and is offered under the
same MIT terms. See the upstream
[LICENSE](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE).
