"""forge-onetrainer — train textual-inversion embeddings from inside forge-again.

Forge removed its built-in training (its ForgeDiffusionEngine runs the whole
conditioning/VAE path under torch.inference_mode, which autograd can't use), so
this does NOT try to train against Forge's backend. Instead it drives
**OneTrainer** (https://github.com/Nerogar/OneTrainer) headless: OneTrainer loads
the checkpoint itself, trains, and writes an embedding that drops straight into
forge-again's embeddings/ folder.

OneTrainer is opt-in and installs into its own venv (its torch/CUDA must not mix
with Forge's pinned stack), exactly the way the AI assistant bundles llama-server.

Design notes:
- We generate a OneTrainer config by layering our overrides over OneTrainer's own
  embedding preset, plus a concepts file (the training-images folder) and an empty
  samples file. All go in a per-run work dir under this extension.
- Modern techniques are exposed as toggles that map to real OneTrainer keys:
  min-SNR-gamma (loss_weight_fn), cosine LR + warmup, and EMA.
- Training runs as a subprocess (OneTrainer venv python scripts/train.py
  --config-path ...), its stdout streamed to the UI. Nothing blocks Forge.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import gradio as gr

from modules import script_callbacks, shared, paths, errors

EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OT_DIR = os.path.join(EXT_DIR, "OneTrainer")
WORK_DIR = os.path.join(EXT_DIR, "work")
OT_REPO = "https://github.com/Nerogar/OneTrainer.git"

# model-type + preset per forge mode
MODE_MAP = {
    "sd":  ("STABLE_DIFFUSION_15",         "512",  "sd15_embedding.json"),
    "xl":  ("STABLE_DIFFUSION_XL_10_BASE", "1024", "sdxl_embedding.json"),
}

_proc_lock = threading.Lock()
_proc = None
_log_lines = []

# Shared, thread-safe buffer for the auto-install that runs at startup, so the
# Train tab can show its progress whenever the user opens it.
_setup_log = []
_setup_state = {"running": False, "done": False}


def _enabled():
    """Training is opt-in: the --enable-training flag, or the Settings toggle."""
    if getattr(shared.cmd_opts, "enable_training", False):
        return True
    try:
        return bool(shared.opts.forge_training_enabled)
    except Exception:
        return False


# --------------------------------------------------------------- OneTrainer paths

def _ot_python():
    for rel in (os.path.join("venv", "Scripts", "python.exe"),
                os.path.join("venv", "bin", "python")):
        p = os.path.join(OT_DIR, rel)
        if os.path.exists(p):
            return p
    return None


def _ot_installed():
    return _ot_python() is not None and os.path.exists(os.path.join(OT_DIR, "scripts", "train.py"))


# --------------------------------------------------------------- forge context

def _current_mode():
    p = os.path.join(paths.script_path if hasattr(paths, "script_path") else ".", "current_mode.txt")
    try:
        return open(p, encoding="utf-8").read().strip() or "sd"
    except OSError:
        return "sd"


def _embeddings_dir():
    d = getattr(shared.cmd_opts, "embeddings_dir", None) or os.path.join(paths.data_path, "embeddings")
    os.makedirs(d, exist_ok=True)
    return d


def _current_checkpoint_path():
    """Full path to the loaded checkpoint, so OneTrainer trains against the same
    model you generate with. Falls back to empty (user must fill it in)."""
    try:
        from modules import sd_models
        info = sd_models.select_checkpoint() if hasattr(sd_models, "select_checkpoint") else None
        if info and getattr(info, "filename", None):
            return info.filename
    except Exception:
        pass
    return ""


# --------------------------------------------------------------- bootstrap

def _do_bootstrap():
    """Clone OneTrainer and build its venv via its own installer.

    Writes to the shared _setup_log so both the startup auto-install and the UI
    can watch it. Serialised behind _proc_lock. Returns True on success.
    """
    if _ot_installed():
        _setup_state["done"] = True
        return True
    if not _proc_lock.acquire(blocking=False):
        _setup_log.append("[setup] another OneTrainer process is running — skipping.")
        return False
    _setup_state["running"] = True
    try:
        if not os.path.exists(os.path.join(OT_DIR, ".git")):
            _setup_log.append("[setup] cloning OneTrainer …")
            p = subprocess.Popen(["git", "clone", "--depth", "1", OT_REPO, OT_DIR],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            for ln in p.stdout:
                _setup_log.append(ln.rstrip())
            p.wait()
            if p.returncode != 0:
                _setup_log.append("[setup] clone FAILED — check your connection and retry.")
                return False

        # OneTrainer ships its own installer that builds a correct venv with the
        # right torch/deps. Use it rather than guessing requirements.
        installer = "install.bat" if os.name == "nt" else "install.sh"
        _setup_log.append(f"[setup] running OneTrainer's {installer} (large; builds its own venv) …")
        cmd = (["cmd", "/c", os.path.join(OT_DIR, installer)] if os.name == "nt"
               else ["bash", os.path.join(OT_DIR, installer)])
        p = subprocess.Popen(cmd, cwd=OT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for ln in p.stdout:
            _setup_log.append(ln.rstrip())
        p.wait()

        ok = _ot_installed()
        _setup_log.append("[setup] ✅ OneTrainer installed. You can start training now."
                          if ok else
                          "[setup] ⚠ installer finished but the venv wasn't found — see the log.")
        _setup_state["done"] = ok
        return ok
    except Exception as e:
        _setup_log.append(f"[setup] ERROR: {e}")
        return False
    finally:
        _setup_state["running"] = False
        _proc_lock.release()


def _auto_install_if_needed():
    """Kick off the download+install in the background when training is enabled.

    Called on app start; returns immediately so it never delays the server.
    """
    if not _enabled() or _ot_installed() or _setup_state["running"]:
        return
    _setup_log.append("[setup] training is enabled — installing OneTrainer in the background …")
    threading.Thread(target=_do_bootstrap, daemon=True).start()


def bootstrap_stream():
    """UI: start the install if needed, then stream the shared setup log."""
    if _ot_installed():
        yield "<p>✅ OneTrainer is already installed.</p>"
        return
    if not _setup_state["running"]:
        threading.Thread(target=_do_bootstrap, daemon=True).start()
        time.sleep(0.5)
    # Stream the shared log until the install settles.
    while _setup_state["running"] or (not _ot_installed() and _setup_log and "FAILED" not in _setup_log[-1] and "ERROR" not in _setup_log[-1]):
        yield "<pre style='max-height:340px;overflow:auto'>" + "\n".join(_setup_log[-400:]) + "</pre>"
        if not _setup_state["running"]:
            break
        time.sleep(1.0)
    yield "<pre style='max-height:340px;overflow:auto'>" + "\n".join(_setup_log[-400:]) + "</pre>"


# --------------------------------------------------------------- config building

def _preset(mode):
    """OneTrainer's own embedding preset for this mode, as the base to override."""
    model_type, resolution, _fname = MODE_MAP[mode]
    src = os.path.join(OT_DIR, "training_presets",
                       "SD1.5" if mode == "sd" else "SDXL",
                       "#sd 1.5 embedding.json" if mode == "sd" else "#sdxl 1.0 embedding.json")
    try:
        return json.load(open(src, encoding="utf-8"))
    except Exception:
        # Minimal fallback if the preset file moved.
        return {"training_method": "EMBEDDING", "model_type": model_type,
                "output_model_format": "SAFETENSORS", "resolution": resolution}


def _build_config(run_dir, mode, name, base_model, images_dir, init_text, token_count,
                  steps, lr, batch_size, use_min_snr, min_snr_gamma, use_ema, lr_scheduler):
    cfg = _preset(mode)
    model_type, default_res, _ = MODE_MAP[mode]

    concepts_file = os.path.join(run_dir, "concepts.json")
    samples_file = os.path.join(run_dir, "samples.json")
    out_path = os.path.join(_embeddings_dir(), f"{name}.safetensors")

    # A single concept = the training-images folder, prompts read from .txt
    # sidecars if present, else the folder-level sample prompt.
    json.dump([{
        "name": name, "path": images_dir, "enabled": True,
        "include_subdirectories": False,
        "image": {"enable_crop_jitter": True, "enable_random_flip": True},
        "text": {"prompt_source": "sample", "prompt_path": ""},
    }], open(concepts_file, "w", encoding="utf-8"), indent=2)
    json.dump([], open(samples_file, "w", encoding="utf-8"))

    cfg.update({
        "model_type": model_type,
        "training_method": "EMBEDDING",
        "base_model_name": base_model,
        "output_model_destination": out_path,
        "output_model_format": "SAFETENSORS",
        "concept_file_name": concepts_file,
        "sample_definition_file_name": samples_file,
        "learning_rate": float(lr),
        "epochs": int(steps),            # OneTrainer counts epochs; concept repeats scale steps
        "batch_size": int(batch_size),
        "resolution": cfg.get("resolution", default_res),
        "debug_dir": os.path.join(run_dir, "debug"),
        "workspace_dir": os.path.join(run_dir, "workspace"),
        "cache_dir": os.path.join(run_dir, "cache"),
        # The embedding to create: placeholder token + how many vectors + init text.
        # is_output_embedding marks this as the trained result to save (vs an
        # additional read-only input embedding); train=True so it's optimized.
        "embedding": {
            "model_name": "", "token_count": int(token_count),
            "initial_embedding_text": init_text or "*",
            "placeholder": f"<{name}>",
            "train": True, "is_output_embedding": True,
        },
    })

    # ---- modern techniques -> real OneTrainer keys ----
    cfg["learning_rate_scheduler"] = lr_scheduler       # e.g. COSINE
    cfg.setdefault("learning_rate_warmup_steps", 20)
    if use_min_snr:
        cfg["loss_weight_fn"] = "MIN_SNR_GAMMA"
        cfg["loss_weight_strength"] = float(min_snr_gamma)
    cfg["ema"] = "CPU" if use_ema else "OFF"

    config_file = os.path.join(run_dir, "config.json")
    json.dump(cfg, open(config_file, "w", encoding="utf-8"), indent=2)
    return config_file, out_path


# --------------------------------------------------------------- run training

def train_stream(name, images_dir, base_model, init_text, token_count,
                 steps, lr, batch_size, use_min_snr, min_snr_gamma, use_ema, lr_scheduler):
    global _proc
    name = (name or "").strip()
    if not _ot_installed():
        yield "<p>OneTrainer isn't installed yet — press <b>Set up OneTrainer</b> first.</p>"
        return
    if not name:
        yield "<p>Give the embedding a name.</p>"
        return
    if not images_dir or not os.path.isdir(images_dir):
        yield f"<p>Training-images folder not found: {images_dir!r}</p>"
        return
    if not base_model or not os.path.exists(base_model):
        yield f"<p>Base checkpoint not found: {base_model!r}</p>"
        return
    if not _proc_lock.acquire(blocking=False):
        yield "<p>A training run is already in progress.</p>"
        return

    del _log_lines[:]
    try:
        mode = _current_mode()
        if mode not in MODE_MAP:
            yield f"<p>Mode {mode!r} isn't supported for embedding training yet (SD 1.5 and SDXL only).</p>"
            return
        run_dir = os.path.join(WORK_DIR, name + "_" + str(int(time.time())))
        os.makedirs(run_dir, exist_ok=True)
        config_file, out_path = _build_config(
            run_dir, mode, name, base_model, images_dir, init_text, token_count,
            steps, lr, batch_size, use_min_snr, min_snr_gamma, use_ema, lr_scheduler)

        _log_lines.append(f"[train] config written to {config_file}")
        _log_lines.append(f"[train] output will be {out_path}")
        _log_lines.append("[train] launching OneTrainer …")

        def render():
            body = "\n".join(_log_lines[-500:])
            return f"<pre style='max-height:420px;overflow:auto'>{body}</pre>"

        yield render()

        # Pass an empty --preset-path so OneTrainer SKIPS its legacy config
        # migrations (the arg's documented effect): our config is already in the
        # current format, and running migrations on it crashes (KeyError in an
        # old-format dtype migration). Empty preset + current-format config is
        # the intended "no migration" path, and it won't break when OneTrainer
        # adds future migration steps.
        nopreset = os.path.join(run_dir, "nopreset.json")
        json.dump({}, open(nopreset, "w", encoding="utf-8"))
        cmd = [_ot_python(), os.path.join("scripts", "train.py"),
               "--preset-path", nopreset, "--config-path", config_file]
        _proc = subprocess.Popen(cmd, cwd=OT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")

        def pump():
            for ln in _proc.stdout:
                _log_lines.append(ln.rstrip())
        t = threading.Thread(target=pump, daemon=True)
        t.start()

        while _proc.poll() is None:
            time.sleep(1.0)
            yield render()
        t.join(timeout=5)

        if _proc.returncode == 0 and os.path.exists(out_path):
            _log_lines.append(f"[train] ✅ done — embedding saved to {out_path}")
            if _convert_for_forge(out_path, name, mode):
                _log_lines.append("[train] converted to a Forge-loadable format.")
            else:
                _log_lines.append("[train] ⚠ saved in OneTrainer format but couldn't convert "
                                  "to Forge's format automatically — see the log.")
            _refresh_embeddings()
            _log_lines.append("[train] embedding list refreshed; use it in a prompt as its filename.")
        else:
            _log_lines.append(f"[train] ⚠ training exited with code {_proc.returncode}; "
                              "see the log above.")
        yield render()
    except Exception as e:
        errors.report("forge-onetrainer: training failed", exc_info=True)
        _log_lines.append(f"[train] ERROR: {e}")
        yield f"<pre>{chr(10).join(_log_lines[-500:])}</pre>"
    finally:
        _proc = None
        _proc_lock.release()


def stop_training():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        return "<p>Stopping the current training run…</p>"
    return "<p>Nothing is training right now.</p>"


def _convert_for_forge(path, name, mode):
    """Rewrite OneTrainer's embedding into a format Forge can load.

    OneTrainer saves two keys (emp_params = input vector, emp_params_out = the
    trained output vector). Forge's loader only accepts `string_to_param`, an
    SDXL `clip_g`+`clip_l` pair, or a dict with exactly ONE tensor — so the raw
    two-key file fails to load. We rewrite it in place:

      * SD 1.5  -> a single-tensor dict {name: (vectors, 768)}   (diffuser-concept branch)
      * SDXL    -> {clip_l: (vectors, 768), clip_g: (vectors, 1280)}

    Returns True on success. Best-effort: on any surprise we leave the file
    untouched and report it, rather than corrupting the output.
    """
    try:
        from safetensors.torch import load_file, save_file
        import torch
        d = load_file(path)
        vec = d.get("emp_params_out")           # the trained/EMA output vector
        if vec is None:
            vec = d.get("emp_params")
        if vec is None:
            # Already single-tensor or an unexpected layout — leave as-is.
            return len(d) == 1
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        if mode == "xl":
            # SDXL embeddings carry both encoders. OneTrainer concatenates them
            # as (vectors, 768+1280); split back into Forge's clip_l / clip_g.
            if vec.shape[-1] == 768 + 1280:
                out = {"clip_l": vec[:, :768].clone(),
                       "clip_g": vec[:, 768:].clone()}
            elif "clip_l" in d and "clip_g" in d:
                out = {"clip_l": d["clip_l"], "clip_g": d["clip_g"]}
            else:
                return False        # unknown SDXL layout; don't guess
        else:
            out = {name: vec}       # SD 1.5: single tensor, Forge accepts it

        save_file({k: v.contiguous().to(torch.float32) for k, v in out.items()}, path)
        return True
    except Exception:
        errors.report("forge-onetrainer: embedding conversion failed", exc_info=True)
        return False


def _refresh_embeddings():
    try:
        from modules import sd_hijack
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
    except Exception:
        pass


# --------------------------------------------------------------- UI

def on_ui_settings():
    section = ("training", "Training")
    shared.opts.add_option(
        "forge_training_enabled",
        shared.OptionInfo(
            False,
            "Enable the Train tab (OneTrainer). When on, OneTrainer is downloaded "
            "and installed on next launch (a few GB, its own environment). Equivalent "
            "to the --enable-training launch flag. Restart after changing.",
            section=section,
        ),
    )


def on_app_started(demo, app):
    # Enabled via flag or setting? Begin the background download+install now.
    _auto_install_if_needed()


def on_ui_tabs():
    # Opt-in: no flag and no setting -> no Train tab at all.
    if not _enabled():
        return []
    with gr.Blocks(analytics_enabled=False) as tab:
        gr.Markdown(
            "Train a **textual-inversion embedding** from a folder of images, using "
            "[OneTrainer](https://github.com/Nerogar/OneTrainer) under the hood. Forge's own "
            "backend can't train (it runs inference-only), so OneTrainer is installed once into "
            "its own environment and driven headless. The finished embedding lands in your "
            "`embeddings/` folder and can be used in any prompt by its name."
        )
        if _ot_installed():
            _init_status = "✅ OneTrainer is installed."
        elif _setup_state["running"]:
            _init_status = "⏳ OneTrainer is installing in the background — open the log with the button."
        else:
            _init_status = "⚠ OneTrainer isn't installed yet — it installs automatically, or press the button."
        with gr.Row():
            setup_btn = gr.Button("Install / show setup log", elem_id="ot_setup")
            setup_status = gr.HTML(value=_init_status)

        with gr.Row():
            with gr.Column():
                name = gr.Textbox(label="Embedding name", placeholder="myconcept",
                                  elem_id="ot_name")
                images = gr.Textbox(label="Training images folder",
                                    placeholder=r"C:\path\to\images", elem_id="ot_images")
                base_model = gr.Textbox(label="Base checkpoint (defaults to your current model)",
                                        value=_current_checkpoint_path(), elem_id="ot_base")
                init_text = gr.Textbox(label="Initialization text", value="*",
                                       elem_id="ot_init")
                token_count = gr.Slider(label="Vectors per token", minimum=1, maximum=16,
                                        step=1, value=4, elem_id="ot_tokens")
            with gr.Column():
                steps = gr.Slider(label="Epochs", minimum=1, maximum=500, step=1, value=100,
                                  elem_id="ot_steps")
                lr = gr.Number(label="Learning rate", value=0.0003, elem_id="ot_lr")
                batch = gr.Slider(label="Batch size", minimum=1, maximum=8, step=1, value=1,
                                  elem_id="ot_batch")
                lr_sched = gr.Dropdown(label="LR scheduler",
                                       choices=["COSINE", "CONSTANT", "LINEAR",
                                                "COSINE_WITH_RESTARTS"],
                                       value="COSINE", elem_id="ot_sched")
                with gr.Accordion("Modern techniques", open=True):
                    use_min_snr = gr.Checkbox(label="Min-SNR-γ loss weighting (faster convergence)",
                                              value=True, elem_id="ot_minsnr")
                    min_snr_gamma = gr.Slider(label="Min-SNR γ", minimum=1.0, maximum=20.0,
                                              step=0.5, value=5.0, elem_id="ot_gamma")
                    use_ema = gr.Checkbox(label="EMA of the embedding (steadier result)",
                                          value=True, elem_id="ot_ema")

        with gr.Row():
            train_btn = gr.Button("Start training", variant="primary", elem_id="ot_train")
            stop_btn = gr.Button("Stop", elem_id="ot_stop")
        log = gr.HTML(elem_id="ot_log")

        setup_btn.click(fn=bootstrap_stream, outputs=[setup_status], show_progress="minimal")
        train_btn.click(
            fn=train_stream,
            inputs=[name, images, base_model, init_text, token_count, steps, lr, batch,
                    use_min_snr, min_snr_gamma, use_ema, lr_sched],
            outputs=[log], show_progress="minimal")
        stop_btn.click(fn=stop_training, outputs=[log], show_progress="hidden")

    return [(tab, "Train", "train")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
