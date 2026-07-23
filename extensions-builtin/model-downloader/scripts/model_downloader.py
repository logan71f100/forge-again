import html
import os
import re
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
    n = filename.lower()
    if "controlnet" in n or n.startswith("control_") or "xinsir" in n:
        return "ControlNet"
    if "lora" in n or "lycoris" in n or "locon" in n:
        return "LoRA"
    if "vae" in n or "taesd" in n:
        return "VAE"
    if re.match(r"^\d+x[-_.]", n) or "esrgan" in n or "upscal" in n or "ultrasharp" in n:
        return "Upscaler (ESRGAN)"
    if "t5" in n or "clip_l" in n or "clip_g" in n or "text_encoder" in n or "umt5" in n:
        return "Text encoder"
    if n.endswith(".pt") or "embedding" in n or "textual_inversion" in n:
        return "Embedding"
    if "flux" in n or "fill" in n:
        return "Checkpoint (Flux)"
    if "xl" in n or "sdxl" in n or "pony" in n or "illustrious" in n:
        return "Checkpoint (XL)"
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


def download(urls_text, category, hf_token):
    urls = [u for u in (line.strip() for line in (urls_text or "").splitlines()) if u]
    urls = list(dict.fromkeys(urls))
    if not urls:
        yield "<p>Paste at least one link first.</p>"
        return
    if not _busy_lock.acquire(blocking=False):
        yield "<p>A download is already running — wait for it to finish (or cancel it).</p>"
        return

    _cancel.clear()
    lines = []
    current = {"text": ""}

    def render():
        rows = "".join(f"<div>{line}</div>" for line in lines)
        live = f"<div>{current['text']}</div>" if current["text"] else ""
        return f"<div style='font-family:monospace; line-height:1.6'>{rows}{live}</div>"

    try:
        dirs = _category_dirs()
        used = set()
        for i, raw in enumerate(urls, start=1):
            try:
                url = _normalize_url(raw)
                display = html.escape(urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]) or url)
                cat = category
                if cat == "Auto-detect":
                    cat = _guess_category(display) or _current_mode_checkpoint_category()
                dest_dir = dirs[cat]
                current["text"] = f"⬇ [{i}/{len(urls)}] {display} → {html.escape(cat)} … starting"
                yield render()

                def log_progress(msg, display=display, i=i):
                    current["text"] = f"⬇ [{i}/{len(urls)}] {display} … {html.escape(msg)}"

                last_yield = time.time()
                gen_done = {}

                def worker():
                    try:
                        gen_done["result"] = _download_one(url, dest_dir, hf_token, log_progress)
                    except Exception as e:
                        gen_done["error"] = e

                t = threading.Thread(target=worker, daemon=True)
                t.start()
                while t.is_alive():
                    t.join(timeout=0.5)
                    if time.time() - last_yield >= 1.0:
                        last_yield = time.time()
                        yield render()

                if "error" in gen_done:
                    raise gen_done["error"]
                name, dest, status = gen_done["result"]
                ok = status.startswith("done") or status.startswith("already")
                icon = "✅" if status.startswith("done") else ("⏭" if status.startswith("already") else "⚠")
                lines.append(f"{icon} {html.escape(name)} → {html.escape(dest)} — {html.escape(status)}")
                if status.startswith("done"):
                    used.add(cat)
                if _cancel.is_set():
                    lines.append("🛑 cancelled — remaining links were not downloaded")
                    current["text"] = ""
                    yield render()
                    return
            except Exception as e:
                lines.append(f"❌ {html.escape(raw[:120])} — {html.escape(str(e))}")
            current["text"] = ""
            yield render()

        if used:
            _refresh_lists(used)
            lines.append("🔄 model lists refreshed — new files show up in the dropdowns (hit the refresh icon if one doesn't)")
        yield render()
    finally:
        _busy_lock.release()


def cancel_download():
    _cancel.set()
    return gr.update()


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
            "Paste one or more **Hugging Face** file links (the page URL of a `.safetensors` file works — "
            "it is converted to a direct download automatically). Direct links from other sites work too. "
            "Files are placed into the right models folder for you."
        )
        with gr.Row():
            with gr.Column(scale=3):
                urls = gr.Textbox(
                    label="Model links (one per line)",
                    lines=4,
                    placeholder="https://huggingface.co/owner/repo/blob/main/model.safetensors",
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
                    label="Hugging Face token (only for gated models)",
                    type="password",
                    elem_id="model_downloader_token",
                )
        with gr.Row():
            start = gr.Button("Download", variant="primary", elem_id="model_downloader_start")
            stop = gr.Button("Cancel", elem_id="model_downloader_cancel")
        status = gr.HTML(elem_id="model_downloader_status")

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

        start.click(fn=download, inputs=[urls, category, hf_token], outputs=[status], show_progress="hidden")
        stop.click(fn=cancel_download, outputs=[status], show_progress="hidden")
        refresh_top.click(
            fn=refresh_top_models,
            outputs=[top_title, *top_rows, top_urls_state, *top_buttons],
            show_progress="hidden",
        )
        for i, btn in enumerate(top_buttons):
            btn.click(fn=_add_top_url(i), inputs=[urls, top_urls_state], outputs=[urls], show_progress="hidden")

    return [(tab, "Model Downloader", "model_downloader")]


script_callbacks.on_ui_tabs(on_ui_tabs)
