# forge-onetrainer

Train **textual-inversion embeddings** from inside forge-again, using
[OneTrainer](https://github.com/Nerogar/OneTrainer) as the training engine.

## Why an external engine

Forge removed its built-in training, and it can't be added back cleanly: Forge's
`ForgeDiffusionEngine` runs the whole conditioning and VAE path under
`torch.inference_mode()`, which autograd cannot use — so there's no gradient path
to train against. (Base Forge stripped `train_embedding` for the same reason.)

Rather than reimplement a trainer against a backend that fights it — and end up
with something strictly worse than OneTrainer — this extension **drives OneTrainer
headless**. OneTrainer loads the checkpoint itself, trains, and writes an embedding
that drops straight into forge-again's `embeddings/` folder, usable in any prompt.

## First-time setup

OneTrainer is **not** bundled (it has its own multi-GB venv with its own
torch/CUDA, which must not mix with forge-again's pinned stack). On the **Train**
tab, press **Set up OneTrainer** once. The extension clones OneTrainer into
`extensions/forge-onetrainer/OneTrainer/` and runs OneTrainer's own installer to
build its venv. Both that clone and the per-run `work/` directory are gitignored.

Requires `git` on PATH (forge-again bootstraps a portable one on Windows).

## Training an embedding

1. Put your concept images in a folder (optionally with `.txt` caption sidecars).
2. On the Train tab: name the embedding, point it at that folder, pick epochs / LR.
3. The base checkpoint defaults to your currently-loaded model, so the embedding
   is trained for what you actually generate with. SD 1.5 and SDXL are supported
   (matched to your current mode).
4. Start training. Progress streams into the log. When it finishes, the embedding
   appears in `embeddings/` and the list is refreshed — use it in a prompt by name.

## Modern techniques

Exposed as toggles that map to real OneTrainer settings:

- **Min-SNR-γ loss weighting** (`loss_weight_fn = MIN_SNR_GAMMA`) — faster, steadier convergence than flat loss.
- **Cosine LR schedule with warmup** — instead of a constant rate.
- **EMA of the embedding** — a smoother final result.

Plus OneTrainer's own defaults for the rest (crop jitter, random flip, mixed
precision). For the full parameter set, edit the generated `config.json` under
`work/<run>/`, or use OneTrainer's own GUI and export a config.

## Format conversion

OneTrainer saves embeddings with its own two-key layout (`emp_params`,
`emp_params_out`), which Forge's loader doesn't accept. After training, the
extension automatically rewrites the file into a Forge-loadable form — a
single-tensor file for SD 1.5, or a `clip_l`/`clip_g` pair for SDXL — so the
result appears in `embeddings/` ready to use.

## Scope and status

- **SD 1.5 embeddings: verified end to end** — install → train → convert →
  Forge loads the embedding and generates with the token.
- **SDXL embeddings: verified end to end** — trained on the current XL
  checkpoint, converted (OneTrainer stores `clip_l`/`clip_g` + `_out` variants
  separately; we emit the `_out` pair), and Forge loads it (shape 2048, correct
  vector count) and generates with the token. Trains at 768px within ~11 GB
  thanks to latent caching.
- LoRA and full fine-tuning are out of scope (use OneTrainer's own GUI). Flux
  embedding training isn't wired up.
