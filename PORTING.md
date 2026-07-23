# forge-again — Porting notes (Gradio 4 → 6 modernization)

How this fork brought a customized Gradio-4.40 Forge install up to the current
stack — **Gradio 6.20, transformers 5.14, huggingface-hub 1.x, diffusers 0.39,
torch 2.13+cu126, Python 3.12** — in a single migration, then packaged it to boot
from one script. Kept as a record of what the port involved and the API breakages
worth knowing if you maintain something similar. **The port is complete** — every
item below is done and running.

## Why
- 3,700+ components take ~30–60 s to client-side mount on Gradio 4; Gradio 6
  renders server-side and hydrates progressively, and per-interaction change
  detection is much cheaper.
- Security: Gradio 4.40 has known CVEs; 5.x+ had a Trail of Bits audit — relevant
  since Forge runs with `--listen`.

## Strategy
- **One migration straight to Gradio 6.20 + transformers 5.14 + hub 1.x**, rather
  than an interim 5.x hop (which would have meant revalidating every DOM selector
  and component API twice). Cost accepted: the transformers-5 changes in Forge's
  text-encoder loading were fixed in the same pass.
- `gradio_rangeslider` has no Gradio-6 build → ControlNet's Timestep Range was
  rebuilt with paired `gr.Slider`s.

## What the port touched
- **Gradio compatibility layer** (`modules/gradio_extensions.py`) — restores the
  Gradio-4 tolerances extensions rely on: dropdown out-of-choices values, slider
  bounds, bare-name `_js` handlers, `visible=False` mounting, config-signature.
- **ForgeCanvas** (`modules_forge/forge_canvas`) — the inpaint canvas JS rebuilt
  for the Gradio-6 DOM and to initialize without a page-load context (lazy tabs).
  Also dropped a Gradio-4-only version gate that had silently blocked all uploads.
- **Lazy-built tabs** via `gr.render` (Settings, Extras, PNG Info, Checkpoint
  Merger, Extensions, img2img) for faster startup and lighter pages.
- **ControlNet** — Timestep Range rebuilt without `gradio_rangeslider`; each
  unit's Enable is now the always-mounted accordion-header checkbox, since Gradio
  6 unmounts a closed accordion's contents.
- **Folded extensions** brought in-repo: Replacer (Gradio-6 UI rework, per-mode
  prompt-chip presets), Segment Anything (transformers-5 fixes for its bundled
  GroundingDINO), and the forge-ai-assistant copilot.
- **transformers-5 sweep** across `backend/` and the extensions (catalog below).
- **Self-contained deploy** — one launcher per OS that bootstraps a portable
  Python 3.12, builds the venv, installs torch cu126 + deps, and launches; a
  per-mode (sd/xl/flux) model-config system; the AI assistant's `llama-server`
  bundled and its vision model auto-downloaded on first run.

## Gradio 4 → 6 API changes worth knowing (recurred across the port)
- Static files: `file=` → `gradio_api/file=` (every webui JS URL and cached image).
- Event `_js="bareName"` strings evaluate but never invoke → rewritten to calling
  arrows (`(...args) => window.NAME(...args)`).
- `visible=False` components are not mounted in the DOM → mount CSS-hidden instead
  (the `webui-hidden-mounted` pattern) wherever JS needs to find the element.
- Only the ACTIVE tab / OPEN accordion mounts its children; a container visibility
  update REMOUNTS children, resetting build-time props — guard against re-emitting.
- `Dropdown.preprocess` RAISES when a submitted value isn't in `choices` (4
  tolerated it); `Slider` silently resets out-of-bounds values to its midpoint —
  both softened back to Gradio-4 behavior in the compat layer.
- `Block.get_config` gained an optional `cls` arg; monkeypatches over Gradio
  internals must match the new signature.
- `Image(tool=, brush_color=)` and `ToolButton(label=)` args are ignored.
- Dropdowns open only via a `keydown ArrowDown`; option text carries a `✓ ` prefix.
- Unqueued events (`queue=False`) travel over bare `/run/predict` whose responses
  can't carry a `gr.render` body or ordered queue results — keep tab-build gates
  and same-trigger state refreshes QUEUED.

## transformers 5 API changes
- `pkg_resources` gone (setuptools 81+) → `importlib.metadata`. Legacy deps
  (pytorch_lightning 1.9.4) still need it, so setuptools is pinned `<81`.
- `apply_chunking_to_forward` removed from `modeling_utils` (BLIP/interrogate
  paths hard-fail when invoked) → shimmed.
- `no_init_weights` moved `modeling_utils` → `transformers.initialization` and now
  takes no args (old `_enable=True` gone).
- **CLIPTextModel is flattened** — no more `.text_model` wrapper; embeddings /
  encoder / final_layer_norm are direct children. Breaks BOTH live access AND
  state-dict key matching (checkpoints still ship `text_model.`-prefixed keys, so a
  runtime key-remap is needed; do NOT rename the checkpoint-format string literals
  in the loaders). Fixed in `backend/loader.py` + `backend/text_processing/classic_engine.py`.
- `get_head_mask` removed and `get_extended_attention_mask`'s 3rd positional is now
  a dtype, not a device — patched in GroundingDINO's BERT (Segment Anything).

## Modernization ledger
| component | was | now |
|---|---|---|
| gradio | 4.40.0 | 6.20.0 |
| huggingface-hub | 0.26.2 | 1.5+ |
| transformers | 4.46.1 | 5.14.1 |
| diffusers | 0.31.0 | 0.39.0 |
| accelerate | 0.31.0 | 1.14.0 |
| peft | 0.13.2 | 0.19.1 |
| fastapi | 0.104.1 | 0.1xx |
| torch | 2.3.1+cu121 | 2.13.0+cu126 |
| python | 3.10 | 3.12 |
