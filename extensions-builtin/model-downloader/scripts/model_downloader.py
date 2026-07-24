import html
import os
import re
import shutil
import threading
import time
import urllib.parse

import gradio as gr
import requests

from modules import script_callbacks, shared, paths, errors


CHUNK_SIZE = 1024 * 1024
_busy_lock = threading.Lock()
_cancel = threading.Event()


def _checkpoints_root():
    ckpt_dir = getattr(shared.cmd_opts, "ckpt_dir", None)
    if ckpt_dir and os.path.basename(os.path.normpath(ckpt_dir)) in ("sd", "xl", "flux"):
        return os.path.dirname(os.path.normpath(ckpt_dir))
    return None


def _category_dirs():
    """Destination folders, resolved from the same launch args the launchers set."""
    models = paths.models_path
    ckroot = _checkpoints_root()

    def ck(mode):
        if ckroot:
            return os.path.join(ckroot, mode)
        return getattr(shared.cmd_opts, "ckpt_dir", None) or os.path.join(models, "Stable-diffusion")

    return {
        "Checkpoint (SD)": ck("sd"),
        "Checkpoint (XL)": ck("xl"),
        "Checkpoint (Flux)": ck("flux"),
        "LoRA": getattr(shared.cmd_opts, "lora_dir", None) or os.path.join(models, "Lora"),
        "VAE": getattr(shared.cmd_opts, "vae_dir", None) or os.path.join(models, "VAE"),
        "Text encoder": getattr(shared.cmd_opts, "text_encoder_dir", None) or os.path.join(models, "text_encoder"),
        "ControlNet": getattr(shared.cmd_opts, "controlnet_dir", None) or os.path.join(models, "ControlNet"),
        "Upscaler (ESRGAN)": getattr(shared.cmd_opts, "esrgan_models_path", None) or os.path.join(models, "ESRGAN"),
        "Embedding": getattr(shared.cmd_opts, "embeddings_dir", None) or os.path.join(paths.data_path, "embeddings"),
    }


def _guess_category(filename):
    """Best-effort classification from a filename alone.

    Deliberately returns None rather than guessing when the name is genuinely
    ambiguous -- most SD 1.5 checkpoints are named after their style, with
    nothing to distinguish them from an SDXL one. Callers fall back to the
    current mode. Where real metadata exists (Civitai, and Hugging Face repo
    tags) it is used instead of this, because it is authoritative.
    """
    n = filename.lower()

    # ControlNet first: several are named "...inpaintXL", which the XL rule
    # below would otherwise claim as a checkpoint.
    if ("controlnet" in n or n.startswith("control_") or "xinsir" in n
            or "t2i-adapter" in n or "control-lora" in n
            or "promax" in n or re.search(r"inpaint[-_]?xl", n)):
        return "ControlNet"
    if "lora" in n or "lycoris" in n or "locon" in n:
        return "LoRA"
    # "ae.safetensors" is the Flux VAE and contains no "vae" at all.
    if ("vae" in n or "taesd" in n
            or os.path.splitext(n)[0] in ("ae", "ae_f16", "diffusion_pytorch_model_vae")):
        return "VAE"
    if re.match(r"^\d+x[-_.]", n) or "esrgan" in n or "upscal" in n or "ultrasharp" in n:
        return "Upscaler (ESRGAN)"
    if "t5" in n or "clip_l" in n or "clip_g" in n or "text_encoder" in n or "umt5" in n:
        return "Text encoder"
    if n.endswith(".pt") or "embedding" in n or "textual_inversion" in n:
        return "Embedding"
    if "flux" in n or "fill" in n:
        return "Checkpoint (Flux)"
    # A plain "xl" substring is deliberate: SDXL checkpoints are named
    # JuggernautXL, RealVisXL, epicrealismXL, anterosXXXL... Anything that is
    # actually a ControlNet or LoRA has already been claimed above, which is
    # what stops "Kataragi_inpaintXL" being mistaken for a checkpoint.
    if "xl" in n or "pony" in n or "illustrious" in n or "noobai" in n:
        return "Checkpoint (XL)"
    if "sd15" in n or "sd_15" in n or "v1-5" in n or "sd1.5" in n:
        return "Checkpoint (SD)"
    return None


def _resolve_huggingface_meta(url, hf_token):
    """Ask the HF API what a repo actually contains.

    Filenames on Hugging Face say even less than on Civitai -- the repo tags
    carry the base model, so this is what makes an SD 1.5 checkpoint land in
    sd/ rather than in whichever mode happens to be active.

    Returns a category or None; never raises.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            return None
        repo = f"{parts[0]}/{parts[1]}"
        headers = {"User-Agent": "forge-again-model-downloader/1.0"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token.strip()}"
        r = requests.get(f"https://huggingface.co/api/models/{repo}", headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        tags = [str(t).lower() for t in (data.get("tags") or [])]
        blob = " ".join(tags)
        card = data.get("cardData") or {}
        base = str(card.get("base_model") or "").lower()

        # The repo id matters as much as the tags: a base model carries no
        # base_model tag (it *is* the base), so stabilityai/stable-diffusion-xl
        # and lllyasviel/ControlNet-v1-1 are only identifiable by name.
        both = blob + " " + base + " " + repo.lower()

        if "controlnet" in both:
            return "ControlNet"
        if "lora" in tags:
            return "LoRA"
        if "textual_inversion" in blob:
            return "Embedding"
        if "flux" in both:
            return "Checkpoint (Flux)"
        if "stable-diffusion-xl" in both or "sdxl" in both:
            return "Checkpoint (XL)"
        if "stable-diffusion-v1-5" in both or "runwayml/stable-diffusion" in both:
            return "Checkpoint (SD)"
    except Exception:
        pass
    return None


def _current_mode_checkpoint_category():
    ckpt_dir = getattr(shared.cmd_opts, "ckpt_dir", None)
    mode = os.path.basename(os.path.normpath(ckpt_dir)) if ckpt_dir else ""
    return {"sd": "Checkpoint (SD)", "xl": "Checkpoint (XL)", "flux": "Checkpoint (Flux)"}.get(mode, "Checkpoint (SD)")


def _normalize_url(url):
    """huggingface /blob/ page links become direct /resolve/ links; strip ?download=."""
    url = url.strip().strip('"')
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("only http(s) links are supported")
    if parsed.netloc.endswith("huggingface.co") and "/blob/" in parsed.path:
        parsed = parsed._replace(path=parsed.path.replace("/blob/", "/resolve/", 1), query="")
    elif parsed.netloc.endswith("huggingface.co"):
        q = [(k, v) for k, v in urllib.parse.parse_qsl(parsed.query) if k != "download"]
        parsed = parsed._replace(query=urllib.parse.urlencode(q))
    return urllib.parse.urlunparse(parsed)


def _request_headers_and_url(url, hf_token):
    headers = {"User-Agent": "forge-again-model-downloader/1.0"}
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.endswith("huggingface.co") and hf_token:
        headers["Authorization"] = f"Bearer {hf_token.strip()}"
    if parsed.netloc.endswith("civitai.com"):
        token_file = os.path.join(paths.data_path, ".civitai_token")
        try:
            token = open(token_file, encoding="utf-8").read().strip()
            if token and "token=" not in (parsed.query or ""):
                q = urllib.parse.parse_qsl(parsed.query)
                q.append(("token", token))
                url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(q)))
        except OSError:
            pass
    return headers, url


def _filename_from_response(url, response):
    cd = response.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    if m:
        name = urllib.parse.unquote(m.group(1))
    else:
        name = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])
    name = os.path.basename(name.strip())
    if not name or name in (".", ".."):
        raise ValueError("could not derive a filename from the link")
    return name


def _fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def _download_one(url, dest_dir, hf_token, log_progress):
    headers, url = _request_headers_and_url(url, hf_token)

    with requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True) as probe:
        probe.raise_for_status()
        name = _filename_from_response(url, probe)
        total = int(probe.headers.get("Content-Length") or 0)
        accepts_ranges = probe.headers.get("Accept-Ranges") == "bytes"

        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest):
            return name, dest, "already exists — skipped"

        part = dest + ".part"
        have = os.path.getsize(part) if os.path.exists(part) else 0

        response, mode = probe, "wb"
        if have and accepts_ranges and total and have < total:
            probe.close()
            headers["Range"] = f"bytes={have}-"
            response = requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)
            if response.status_code == 206:
                mode = "ab"
            else:
                have = 0
        else:
            have = 0

        done = have
        started = time.time()
        last_report = 0.0
        try:
            with open(part, mode) as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if _cancel.is_set():
                        return name, dest, "cancelled — partial file kept for resume"
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_report >= 1.0:
                        last_report = now
                        speed = (done - have) / max(now - started, 0.001)
                        pct = f"{done * 100 // total}%" if total else _fmt_size(done)
                        log_progress(f"{pct} of {_fmt_size(total) if total else '?'} — {_fmt_size(speed)}/s")
        finally:
            if response is not probe:
                response.close()

        if total and done < total:
            return name, dest, "connection ended early — partial file kept, run again to resume"
        os.replace(part, dest)
        return name, dest, f"done ({_fmt_size(done)})"


def _refresh_lists(categories_used):
    try:
        if any(c.startswith("Checkpoint") for c in categories_used):
            from modules import sd_models
            sd_models.list_models()
        if "VAE" in categories_used:
            from modules import sd_vae
            sd_vae.refresh_vae_list()
        if "LoRA" in categories_used:
            import networks
            networks.list_available_networks()
    except Exception:
        errors.report("model-downloader: refreshing model lists failed", exc_info=True)


# ------------------------------------------------------------------ civitai
#
# Civitai page links can't be downloaded directly, and the API knows things a
# filename never will -- the model type and its base model -- so a checkpoint
# lands in sd/, xl/ or flux/ correctly instead of being guessed at from a name.

CIVITAI_API = "https://civitai.com/api/v1"

_CIVITAI_TYPE_MAP = {
    "lora": "LoRA",
    "locon": "LoRA",
    "lycoris": "LoRA",
    "doraversion": "LoRA",
    "textualinversion": "Embedding",
    "vae": "VAE",
    "controlnet": "ControlNet",
    "upscaler": "Upscaler (ESRGAN)",
}


def _civitai_token(explicit=None):
    if explicit and explicit.strip():
        return explicit.strip()
    try:
        return open(os.path.join(paths.data_path, ".civitai_token"), encoding="utf-8").read().strip()
    except OSError:
        return ""


def _civitai_category(model_type, base_model):
    cat = _CIVITAI_TYPE_MAP.get((model_type or "").strip().lower().replace(" ", ""))
    if cat:
        return cat
    b = (base_model or "").lower()
    if "flux" in b:
        return "Checkpoint (Flux)"
    if any(k in b for k in ("xl", "pony", "illustrious", "noobai")):
        return "Checkpoint (XL)"
    if b:
        return "Checkpoint (SD)"
    return None


def _civitai_get(path, token):
    headers = {"User-Agent": "forge-again-model-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{CIVITAI_API}/{path}", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def _civitai_pick_file(version):
    files = version.get("files") or []
    if not files:
        raise ValueError("this Civitai version has no downloadable files")
    # Prefer the primary weights; some versions also ship configs/previews.
    for f in files:
        if f.get("primary"):
            return f
    for f in files:
        if (f.get("type") or "").lower() == "model":
            return f
    return files[0]


def _resolve_civitai(url, token):
    """Turn any civitai.com link into (download_url, filename, category).

    Raises on anything it can't resolve, so the queue shows a real reason
    instead of silently downloading an HTML error page.
    """
    parsed = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(parsed.query))
    version_id = q.get("modelVersionId")

    m = re.search(r"/api/download/models/(\d+)", parsed.path)
    if m:
        version_id = version_id or m.group(1)
    model_id = None
    m = re.search(r"/models/(\d+)", parsed.path)
    if m and "/api/download/" not in parsed.path:
        model_id = m.group(1)

    if version_id:
        version = _civitai_get(f"model-versions/{version_id}", token)
        model_type = ((version.get("model") or {}).get("type"))
    elif model_id:
        model = _civitai_get(f"models/{model_id}", token)
        versions = model.get("modelVersions") or []
        if not versions:
            raise ValueError("this Civitai model has no versions")
        version = versions[0]                       # newest first
        model_type = model.get("type")
        if not version.get("files"):
            version = _civitai_get(f"model-versions/{version['id']}", token)
    else:
        raise ValueError("could not find a model or version id in that Civitai link")

    f = _civitai_pick_file(version)
    dl = f.get("downloadUrl") or version.get("downloadUrl")
    if not dl:
        raise ValueError("Civitai did not return a download URL (model may require sign-in)")
    name = f.get("name") or ""
    cat = _civitai_category(model_type, version.get("baseModel"))
    return dl, name, cat


# -------------------------------------------------------------------- queue
#
# A single background worker drains a shared queue, so links can be added at
# any time -- including while a download is running.

_queue = []
_qlock = threading.RLock()
_worker_thread = None
_cancel_current = threading.Event()
_next_id = [0]


def _queue_add(url, name, category, note=""):
    with _qlock:
        _next_id[0] += 1
        item = {
            "id": _next_id[0], "url": url, "name": name or url,
            "category": category, "status": "queued", "detail": note,
            "dest": "",
        }
        _queue.append(item)
        return item


def _worker_loop():
    while True:
        with _qlock:
            item = next((i for i in _queue if i["status"] == "queued"), None)
            if item is None:
                return                              # drained; thread exits
            item["status"] = "downloading"
            item["detail"] = "starting"
        _cancel_current.clear()
        used = set()
        try:
            dirs = _category_dirs()
            dest_dir = dirs[item["category"]]

            def log_progress(msg, it=item):
                it["detail"] = msg

            name, dest, status = _download_one(item["url"], dest_dir, item.get("token", ""), log_progress)
            with _qlock:
                item["name"] = name
                item["dest"] = dest
                item["detail"] = status
                if status.startswith("done"):
                    item["status"] = "done"
                    used.add(item["category"])
                elif status.startswith("already"):
                    item["status"] = "skipped"
                elif status.startswith("cancelled"):
                    item["status"] = "cancelled"
                else:
                    item["status"] = "failed"
        except Exception as e:
            with _qlock:
                item["status"] = "failed"
                item["detail"] = str(e)[:300]
        if used:
            _refresh_lists(used)


def _ensure_worker():
    global _worker_thread
    with _qlock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
            _worker_thread.start()


_ICON = {"queued": "🕘", "downloading": "⬇", "done": "✅", "skipped": "⏭",
         "failed": "❌", "cancelled": "🛑"}


def _render_queue():
    with _qlock:
        items = list(_queue)
    if not items:
        return "<p>Queue is empty. Paste links and press <b>Add to queue</b>.</p>"
    pending = sum(1 for i in items if i["status"] in ("queued", "downloading"))
    rows = []
    for i in items:
        icon = _ICON.get(i["status"], "•")
        detail = html.escape(i["detail"] or "")
        cat = html.escape(i["category"])
        nm = html.escape(str(i["name"])[:90])
        colour = {"failed": "#e06c75", "done": "#98c379", "downloading": "#61afef"}.get(i["status"], "#9aa5b1")
        rows.append(
            f"<div style='padding:2px 0'><span style='color:{colour}'>{icon}</span> "
            f"<b>{nm}</b> <span style='opacity:.7'>→ {cat}</span>"
            + (f" <span style='opacity:.8'>— {detail}</span>" if detail else "") + "</div>"
        )
    head = f"<div style='opacity:.7;margin-bottom:4px'>{len(items)} item(s), {pending} pending</div>"
    return f"<div style='font-family:monospace;line-height:1.5'>{head}{''.join(rows)}</div>"


def add_to_queue(urls_text, category, hf_token, civitai_token):
    """Resolve links and append them. Never blocks on the download itself."""
    raw_urls = [u for u in (line.strip() for line in (urls_text or "").splitlines()) if u]
    raw_urls = list(dict.fromkeys(raw_urls))
    if not raw_urls:
        return _render_queue(), gr.update()

    ctoken = _civitai_token(civitai_token)
    for raw in raw_urls:
        try:
            url = _normalize_url(raw)
            parsed = urllib.parse.urlparse(url)
            name, cat = "", None

            if parsed.netloc.endswith("civitai.com"):
                url, name, cat = _resolve_civitai(url, ctoken)
                if ctoken and "token=" not in url:
                    sep = "&" if "?" in url else "?"
                    url = f"{url}{sep}token={urllib.parse.quote(ctoken)}"
            else:
                name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])

            if category != "Auto-detect":
                cat = category
            if not cat:
                # Filename first (it's free and often decisive), then repo
                # metadata, and only then the current mode as a last resort.
                cat = _guess_category(name)
                if not cat and parsed.netloc.endswith("huggingface.co"):
                    cat = _resolve_huggingface_meta(url, hf_token)
                cat = cat or _current_mode_checkpoint_category()

            item = _queue_add(url, name, cat, "waiting")
            item["token"] = hf_token or ""
        except Exception as e:
            it = _queue_add(raw, raw[:90], _current_mode_checkpoint_category(), str(e)[:200])
            it["status"] = "failed"

    _ensure_worker()
    return _render_queue(), ""          # clear the input box


def queue_stream():
    """Stream queue state while anything is pending, then settle."""
    yield _render_queue()
    idle_rounds = 0
    while idle_rounds < 2:
        time.sleep(1.0)
        with _qlock:
            busy = any(i["status"] in ("queued", "downloading") for i in _queue)
        idle_rounds = 0 if busy else idle_rounds + 1
        yield _render_queue()


def cancel_download():
    """Cancel the item being downloaded; the rest of the queue continues."""
    _cancel.set()
    _cancel_current.set()
    time.sleep(0.2)
    _cancel.clear()
    return _render_queue()


def clear_finished():
    with _qlock:
        _queue[:] = [i for i in _queue if i["status"] in ("queued", "downloading")]
    return _render_queue()


def clear_queue():
    """Drop everything not already in flight."""
    with _qlock:
        _queue[:] = [i for i in _queue if i["status"] == "downloading"]
    return _render_queue()


# --------------------------------------------------------- installed models
#
# Auto-sorting can only guess when a filename is ambiguous, so being able to
# move a model to the right folder afterwards matters as much as getting the
# download right.

MODEL_EXTS = (".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".bin", ".sft")


def _installed_models():
    """{category: [(filename, size, fullpath)]} for everything on disk."""
    out = {}
    for cat, d in _category_dirs().items():
        if not d or not os.path.isdir(d):
            continue
        files = []
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if not os.path.isfile(p) or fn.startswith("."):
                continue
            if not fn.lower().endswith(MODEL_EXTS):
                continue
            try:
                files.append((fn, os.path.getsize(p), p))
            except OSError:
                pass
        if files:
            out[cat] = files
    return out


def _model_choices():
    """(label, value) pairs; value is "category::filename"."""
    choices = []
    for cat, files in _installed_models().items():
        for fn, size, _p in files:
            choices.append((f"[{cat}] {fn}  ({_fmt_size(size)})", f"{cat}::{fn}"))
    return choices


def _resolve_selection(value):
    """"category::filename" -> (category, filename, verified path).

    Rejects anything that would resolve outside its category directory, so a
    crafted value can't reach arbitrary files.
    """
    cat, _, fn = (value or "").partition("::")
    dirs = _category_dirs()
    if cat not in dirs or not fn:
        raise ValueError(f"unknown selection: {value!r}")
    base = os.path.abspath(dirs[cat])
    path = os.path.abspath(os.path.join(base, fn))
    if os.path.dirname(path) != base or not os.path.isfile(path):
        raise ValueError(f"not a file in {cat}: {fn}")
    return cat, os.path.basename(path), path


def load_installed(filter_cat="All"):
    """Rows for the table: [{value, name, category, size}], newest folders first."""
    rows = []
    for cat, files in _installed_models().items():
        if filter_cat not in ("All", None) and cat != filter_cat:
            continue
        for fn, size, _p in files:
            rows.append({"value": f"{cat}::{fn}", "name": fn,
                         "category": cat, "size": _fmt_size(size), "bytes": size})
    rows.sort(key=lambda r: (r["category"], r["name"].lower()))
    return rows


def installed_summary(rows):
    total = sum(r["bytes"] for r in rows)
    return f"**{len(rows)} model file(s)** — {_fmt_size(total)}"


def move_one(value, target_category):
    """Move a single file; returns (status_html, refreshed_rows)."""
    try:
        cat, fn, src = _resolve_selection(value)
    except Exception as e:
        return f"<span style='color:#e06c75'>{html.escape(str(e))}</span>", None
    if not target_category or target_category == cat:
        return "<span style='opacity:.7'>Pick a different destination first.</span>", None

    dirs = _category_dirs()
    dest_dir = dirs.get(target_category)
    if not dest_dir:
        return f"<span style='color:#e06c75'>unknown destination</span>", None
    os.makedirs(dest_dir, exist_ok=True)
    dst = os.path.join(dest_dir, fn)
    if os.path.exists(dst):
        return (f"<span style='color:#e5c07b'>{html.escape(fn)} already exists in "
                f"{html.escape(target_category)} — nothing moved</span>", None)
    try:
        shutil.move(src, dst)
    except Exception as e:
        return f"<span style='color:#e06c75'>{html.escape(str(e))}</span>", None
    _refresh_lists({cat, target_category})
    return (f"<span style='color:#98c379'>moved {html.escape(fn)} → "
            f"{html.escape(target_category)}</span>", True)


def delete_one(value):
    """Delete a single file; the UI arms this behind a confirm click."""
    try:
        cat, fn, path = _resolve_selection(value)
        size = os.path.getsize(path)
        os.remove(path)
    except Exception as e:
        return f"<span style='color:#e06c75'>{html.escape(str(e))}</span>", None
    _refresh_lists({cat})
    return (f"<span style='color:#98c379'>deleted {html.escape(fn)} "
            f"({_fmt_size(size)} freed)</span>", True)


def move_models(selected, target_category):
    """Move files into another category folder (e.g. an XL checkpoint into sd/)."""
    if not selected:
        return "<p>Select at least one model first.</p>", *refresh_installed()
    if not target_category:
        return "<p>Pick a destination first.</p>", *refresh_installed()

    dirs = _category_dirs()
    dest_dir = dirs.get(target_category)
    if not dest_dir:
        return f"<p>Unknown destination: {html.escape(str(target_category))}</p>", *refresh_installed()
    os.makedirs(dest_dir, exist_ok=True)

    lines, touched = [], set()
    for value in selected:
        try:
            cat, fn, src = _resolve_selection(value)
            if cat == target_category:
                lines.append(f"⏭ {html.escape(fn)} — already in {html.escape(cat)}")
                continue
            dst = os.path.join(dest_dir, fn)
            if os.path.exists(dst):
                lines.append(f"⚠ {html.escape(fn)} — a file with that name is already in {html.escape(target_category)}; left alone")
                continue
            shutil.move(src, dst)
            lines.append(f"✅ {html.escape(fn)} — {html.escape(cat)} → {html.escape(target_category)}")
            touched.update((cat, target_category))
        except Exception as e:
            lines.append(f"❌ {html.escape(str(value))} — {html.escape(str(e))}")

    if touched:
        _refresh_lists(touched)
    body = "".join(f"<div>{l}</div>" for l in lines)
    return f"<div style='font-family:monospace;line-height:1.6'>{body}</div>", *refresh_installed()


def delete_models(selected, confirm):
    """Permanently delete the selected files. Gated behind an explicit tick."""
    if not selected:
        return "<p>Select at least one model first.</p>", *refresh_installed()
    if not confirm:
        return ("<p>⚠ Deleting is permanent. Tick <b>Yes, delete permanently</b> "
                "next to the button to confirm.</p>", *refresh_installed())

    lines, touched, freed = [], set(), 0
    for value in selected:
        try:
            cat, fn, path = _resolve_selection(value)
            size = os.path.getsize(path)
            os.remove(path)
            freed += size
            touched.add(cat)
            lines.append(f"🗑 {html.escape(fn)} — deleted from {html.escape(cat)} ({_fmt_size(size)})")
        except Exception as e:
            lines.append(f"❌ {html.escape(str(value))} — {html.escape(str(e))}")

    if touched:
        _refresh_lists(touched)
    if freed:
        lines.append(f"<b>{_fmt_size(freed)} freed</b>")
    body = "".join(f"<div>{l}</div>" for l in lines)
    return f"<div style='font-family:monospace;line-height:1.6'>{body}</div>", *refresh_installed()


# ------------------------------------------------- top models per mode

MODE_FILTERS = {
    "sd": ("SD 1.5", "base_model:finetune:stable-diffusion-v1-5/stable-diffusion-v1-5"),
    "xl": ("SDXL", "base_model:finetune:stabilityai/stable-diffusion-xl-base-1.0"),
    "flux": ("Flux", "base_model:finetune:black-forest-labs/FLUX.1-dev"),
}
_top_cache = {}
TOP_CACHE_SECONDS = 6 * 3600


def _current_mode():
    ckpt_dir = getattr(shared.cmd_opts, "ckpt_dir", None)
    mode = os.path.basename(os.path.normpath(ckpt_dir)) if ckpt_dir else ""
    return mode if mode in MODE_FILTERS else "sd"


def _pick_checkpoint_file(repo_id):
    """Largest root-level .safetensors >= 1.5 GB, or None if the repo has no single-file checkpoint."""
    r = requests.get(f"https://huggingface.co/api/models/{repo_id}", params={"blobs": "true"}, timeout=10)
    r.raise_for_status()
    d = r.json()
    files = [s for s in d.get("siblings", [])
             if "/" not in s["rfilename"] and s["rfilename"].endswith(".safetensors")
             and (s.get("size") or 0) >= 1.5 * 1024 ** 3]
    if not files:
        return None, False
    best = max(files, key=lambda s: s.get("size") or 0)
    gated = bool(d.get("gated"))
    return best, gated


def _fetch_top_models(mode):
    cached = _top_cache.get(mode)
    if cached and time.time() - cached[0] < TOP_CACHE_SECONDS:
        return cached[1]

    label, tag = MODE_FILTERS[mode]
    r = requests.get(
        "https://huggingface.co/api/models",
        params={"filter": tag, "pipeline_tag": "text-to-image", "sort": "downloads", "direction": "-1", "limit": "12"},
        timeout=10,
    )
    r.raise_for_status()
    results = []
    for m in r.json():
        repo = m["id"]
        joined_tags = " ".join(m.get("tags", [])).lower()
        if "controlnet" in repo.lower() or "controlnet" in joined_tags or "lora" in joined_tags:
            continue
        try:
            best, gated = _pick_checkpoint_file(repo)
        except Exception:
            continue
        if not best:
            continue
        results.append({
            "repo": repo,
            "downloads": m.get("downloads") or 0,
            "likes": m.get("likes") or 0,
            "size": best.get("size") or 0,
            "gated": gated,
            "url": f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(best['rfilename'])}",
            "file": best["rfilename"],
        })
        if len(results) >= 3:
            break
    _top_cache[mode] = (time.time(), results)
    return results


def refresh_top_models():
    mode = _current_mode()
    label = MODE_FILTERS[mode][0]
    try:
        tops = _fetch_top_models(mode)
    except Exception as e:
        return (f"**Top {label} checkpoints on Hugging Face** — couldn't reach huggingface.co ({e})",
                "", "", "", [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

    rows, urls = [], []
    for t in tops:
        gated = " · 🔒 gated (needs token)" if t["gated"] else ""
        rows.append(
            f"**[{t['repo']}](https://huggingface.co/{t['repo']})** — "
            f"{t['downloads']:,} downloads · {t['likes']:,} likes · {_fmt_size(t['size'])}{gated}<br>"
            f"<span style='opacity:.7'>{html.escape(t['file'])}</span>"
        )
        urls.append(t["url"])
    while len(rows) < 3:
        rows.append("")
    buttons = [gr.update(visible=i < len(urls)) for i in range(3)]
    title = f"**Top {MODE_FILTERS[mode][0]} checkpoints on Hugging Face** (for your current mode — switch modes and refresh to see others)"
    return (title, rows[0], rows[1], rows[2], urls, *buttons)


def _add_top_url(index):
    def fn(urls_text, top_urls):
        if not top_urls or index >= len(top_urls):
            return gr.update()
        link = top_urls[index]
        existing = (urls_text or "").rstrip()
        if link in existing:
            return gr.update()
        return (existing + "\n" + link).lstrip()
    return fn


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tab:
        gr.Markdown(
            "Paste **Hugging Face** or **Civitai** links, one per line. A Civitai model page URL works "
            "(`civitai.com/models/12345`, with or without `?modelVersionId=`) — the API is asked for the "
            "actual file, and the model type and base model decide the folder, so a checkpoint lands in "
            "`sd/`, `xl/` or `flux/` correctly. Hugging Face `/blob/` page links become direct downloads. "
            "Links can be added **at any time**, including while something is downloading."
        )
        with gr.Row():
            with gr.Column(scale=3):
                urls = gr.Textbox(
                    label="Model links (one per line)",
                    lines=4,
                    placeholder="https://civitai.com/models/12345\nhttps://huggingface.co/owner/repo/blob/main/model.safetensors",
                    elem_id="model_downloader_urls",
                )
            with gr.Column(scale=1):
                category = gr.Dropdown(
                    label="Put into",
                    choices=["Auto-detect"] + list(_category_dirs().keys()),
                    value="Auto-detect",
                    elem_id="model_downloader_category",
                )
                hf_token = gr.Textbox(
                    label="Hugging Face token (gated models)",
                    type="password",
                    elem_id="model_downloader_token",
                )
                civitai_token = gr.Textbox(
                    label="Civitai token (some models require one)",
                    type="password",
                    elem_id="model_downloader_civitai_token",
                )
        with gr.Row():
            start = gr.Button("Add to queue", variant="primary", elem_id="model_downloader_start")
            stop = gr.Button("Cancel current", elem_id="model_downloader_cancel")
            clear_done = gr.Button("Clear finished", elem_id="model_downloader_clear_done")
            refresh_q = gr.Button("🔄 Refresh", elem_id="model_downloader_refresh_queue")
        status = gr.HTML(value=_render_queue(), elem_id="model_downloader_status")

        top_title = gr.Markdown("**Top checkpoints on Hugging Face** — press refresh to load", elem_id="model_downloader_top_title")
        top_urls_state = gr.State([])
        top_rows, top_buttons = [], []
        for i in range(3):
            with gr.Row():
                with gr.Column(scale=8):
                    top_rows.append(gr.Markdown("", elem_id=f"model_downloader_top_{i}"))
                with gr.Column(scale=1, min_width=120):
                    top_buttons.append(gr.Button("Add to list", visible=False, elem_id=f"model_downloader_top_add_{i}"))
        refresh_top = gr.Button("🔄 Refresh top models", elem_id="model_downloader_refresh_top")

        # Adding returns immediately (the worker runs in the background), then a
        # second event streams queue state until it settles -- so the box is
        # free for more links while downloads continue.
        start.click(
            fn=add_to_queue,
            inputs=[urls, category, hf_token, civitai_token],
            outputs=[status, urls],
            show_progress="hidden",
        ).then(fn=queue_stream, outputs=[status], show_progress="hidden")
        stop.click(fn=cancel_download, outputs=[status], show_progress="hidden")
        clear_done.click(fn=clear_finished, outputs=[status], show_progress="hidden")
        refresh_q.click(fn=queue_stream, outputs=[status], show_progress="hidden")

        refresh_top.click(
            fn=refresh_top_models,
            outputs=[top_title, *top_rows, top_urls_state, *top_buttons],
            show_progress="hidden",
        )
        for i, btn in enumerate(top_buttons):
            btn.click(fn=_add_top_url(i), inputs=[urls, top_urls_state], outputs=[urls], show_progress="hidden")

        with gr.Accordion("Installed models — move or delete", open=False, elem_id="model_downloader_manage"):
            all_cats = list(_category_dirs().keys())
            rows_state = gr.State(load_installed("All"))
            armed_state = gr.State("")        # row awaiting delete confirmation

            with gr.Row():
                filter_cat = gr.Dropdown(
                    label="Show", choices=["All"] + all_cats, value="All",
                    scale=2, elem_id="model_downloader_filter",
                )
                manage_summary = gr.Markdown(installed_summary(load_installed("All")))
                refresh_installed_btn = gr.Button("🔄 Refresh", scale=0, elem_id="model_downloader_refresh_installed")
            manage_status = gr.HTML(elem_id="model_downloader_manage_status")

            @gr.render(inputs=[rows_state, armed_state, filter_cat])
            def _render_installed(rows, armed, current_filter):
                if not rows:
                    gr.Markdown("_Nothing here yet._")
                    return
                with gr.Row():
                    gr.Markdown("**Model**", scale=5)
                    gr.Markdown("**Folder**", scale=2)
                    gr.Markdown("**Size**", scale=1)
                    gr.Markdown("**Move to**", scale=3)
                    gr.Markdown("", scale=1)
                for row in rows:
                    value = row["value"]
                    with gr.Row(equal_height=True):
                        gr.Markdown(row["name"], scale=5)
                        gr.Markdown(f"`{row['category']}`", scale=2)
                        gr.Markdown(row["size"], scale=1)
                        target = gr.Dropdown(
                            choices=[c for c in all_cats if c != row["category"]],
                            value=None, show_label=False, container=False, scale=3,
                        )
                        if armed == value:
                            confirm_btn = gr.Button("Confirm?", variant="stop", scale=1, min_width=90)
                            confirm_btn.click(
                                fn=lambda v=value, f=current_filter: (
                                    *delete_one(v)[:1], load_installed(f), ""),
                                outputs=[manage_status, rows_state, armed_state],
                            ).then(fn=installed_summary, inputs=[rows_state], outputs=[manage_summary])
                        else:
                            with gr.Row(scale=1):
                                move_btn = gr.Button("Move", scale=1, min_width=70)
                                del_btn = gr.Button("🗑", variant="stop", scale=0, min_width=44)
                                move_btn.click(
                                    fn=lambda t, v=value, f=current_filter: (
                                        move_one(v, t)[0], load_installed(f), ""),
                                    inputs=[target],
                                    outputs=[manage_status, rows_state, armed_state],
                                ).then(fn=installed_summary, inputs=[rows_state], outputs=[manage_summary])
                                # First click arms the row; the button becomes
                                # "Confirm?" so nothing is ever one click from gone.
                                del_btn.click(
                                    fn=lambda v=value: (
                                        v, "<span style='opacity:.8'>Click <b>Confirm?</b> to delete "
                                           "permanently, or Refresh to cancel.</span>"),
                                    outputs=[armed_state, manage_status],
                                )

        # Kept queued: gr.render bodies cannot travel over an unqueued
        # /run/predict response.
        refresh_installed_btn.click(
            fn=lambda f: (load_installed(f), ""),
            inputs=[filter_cat], outputs=[rows_state, armed_state],
        ).then(fn=installed_summary, inputs=[rows_state], outputs=[manage_summary])
        filter_cat.change(
            fn=lambda f: (load_installed(f), ""),
            inputs=[filter_cat], outputs=[rows_state, armed_state],
        ).then(fn=installed_summary, inputs=[rows_state], outputs=[manage_summary])


    return [(tab, "Model Downloader", "model_downloader")]


script_callbacks.on_ui_tabs(on_ui_tabs)
