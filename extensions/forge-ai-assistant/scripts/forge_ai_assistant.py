"""Forge AI Assistant — local LLM copilot for dialing in inpainting.

Adds API endpoints under /forge-ai/* that:
  - start/stop a patched llama-server directly (freeing Forge VRAM first)
  - hibernate/wake the LLM's VRAM around Forge generations (/sleep, /wake —
    weights + KV cache move to pinned host RAM in ~1.7s, restore in ~1.5s,
    with the encoded conversation incl. image embeddings fully preserved)
  - proxy chat requests to llama-server's OpenAI-compatible API
  - report status (server health, VRAM)

The server binary is a llama.cpp llama-server; the VRAM hibernate feature
expects a build patched with /sleep and /wake endpoints (optional — without
it the assistant still works but shares VRAM with generations). Binary and
GGUF model paths are configured under Settings > AI Assistant.

The chat UI itself is injected by javascript/forge_ai_assistant.js.
"""

import os
import glob
import json
import time
import base64
import shlex
import ctypes
import threading
import subprocess
from ctypes import wintypes

import gradio as gr
import requests
from fastapi import Body

import modules.scripts as scripts_mod
from modules import script_callbacks, shared

# Self-contained defaults so the assistant works out of the box in a fresh copy:
#   - the patched llama-server binary is bundled at <project>/forge-llm/
#   - the vision GGUF model auto-downloads to <models>/llm/ on first launch (see the start scripts)
# Both are overridable in Settings > AI Assistant. <project> is this repo's root
# (…/extensions/forge-ai-assistant/scripts/this_file → up 4).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_LLAMA_EXE = "llama-server.exe" if os.name == "nt" else "llama-server"
DEFAULT_SERVER_BIN = os.path.join(_PROJECT_ROOT, "forge-llm", _LLAMA_EXE)
# follow FORGE_MODELS_DIR (set by the launchers) so the LLM lives beside the SD models
DEFAULT_MODELS_DIR = os.path.join(os.environ.get("FORGE_MODELS_DIR", os.path.join(_PROJECT_ROOT, "models")), "llm")
DEFAULT_API_URL = "http://127.0.0.1:5000"

_proc = {"popen": None}
_auto = {"stopped_for_gen": False, "restoring": False}
_restore_lock = threading.Lock()
_start_lock = threading.Lock()   # serializes ALL launch paths (button, auto-start, restore thread)


# ---------------------------------------------------------------- settings

def on_ui_settings():
    section = ("forge_ai", "AI Assistant")
    shared.opts.add_option(
        "forge_ai_task_guidance",
        shared.OptionInfo(
            "Typical task: replacing an object or garment in a photo (e.g. swapping a shirt's "
            "color, removing sunglasses, changing a hairstyle). Detection prompt names what "
            "EXISTS in the source photo (\"sunglasses\", \"red shirt\"); the positive prompt "
            "describes what should appear instead; the negative prompt lists what must not "
            "come back. For full object replacement use high denoising (0.85-1.0); for "
            "restyling the same object use 0.4-0.6.",
            "Task guidance injected into the assistant's system prompt — describe YOUR typical "
            "workflow, with worked example prompts. The assistant follows this when deciding "
            "what to write into the prompt fields.",
            gr.Textbox,
            {"lines": 8},
            section=section,
        ),
    )
    shared.opts.add_option(
        "forge_ai_checkpoint_notes",
        shared.OptionInfo(
            "",
            "Your own per-checkpoint tuning notes, as a JSON array: "
            "[{\"match\": \"modelfilename\", \"notes\": \"sampler/steps/CFG/prompt style...\"}]. "
            "\"match\" is matched case-insensitively as a substring of the loaded checkpoint's "
            "filename; the notes are injected into the assistant's system prompt so it tunes for "
            "that model. Merged with the built-in notes for common public models — yours take "
            "priority on a conflict. Leave blank to use only the built-ins.",
            gr.Textbox,
            {"lines": 8},
            section=section,
        ),
    )
    shared.opts.add_option(
        "forge_ai_server_bin",
        shared.OptionInfo(DEFAULT_SERVER_BIN, "llama-server binary (patched build with /sleep + /wake)", section=section),
    )
    shared.opts.add_option(
        "forge_ai_models_dir",
        shared.OptionInfo(DEFAULT_MODELS_DIR, "GGUF models folder", section=section),
    )
    shared.opts.add_option(
        "forge_ai_api_url",
        shared.OptionInfo(DEFAULT_API_URL, "llama-server API base URL", section=section),
    )
    shared.opts.add_option(
        "forge_ai_model",
        shared.OptionInfo(
            "Qwen3-VL-30B-A3B-Thinking/Qwen3-VL-30B-A3B-Thinking-UD-Q4_K_XL.gguf",
            "Default GGUF model to load (path relative to models dir; blank = pick in chat panel)",
            section=section,
        ),
    )
    shared.opts.add_option(
        "forge_ai_extra_args",
        shared.OptionInfo(
            # native llama-server flags — no more text-gen translation layer.
            # --parallel 2: exactly one slot for the conversation + one for the
            # judge; 4 idle slots sharing the unified KV caused eviction fights
            # (symptom: prompt processing looping with progress > 1.0)
            '--ctx-size 65536 --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 '
            '--flash-attn on --image-min-tokens 1024 --parallel 2',
            "Extra llama-server launch args (--model/--mmproj/--host/--port are added automatically)",
            section=section,
        ),
    )
    shared.opts.add_option(
        "forge_ai_auto_start",
        shared.OptionInfo(True, "Start the LLM together with Forge (it always shuts down with Forge)", section=section),
    )
    shared.opts.add_option(
        "forge_ai_auto_unload",
        shared.OptionInfo(True, "Unload Forge model weights before starting the LLM", section=section),
    )
    shared.opts.add_option(
        "forge_ai_auto_restore",
        shared.OptionInfo(True, "Auto-restart the LLM after a generation that stopped it to free VRAM", section=section),
    )
    shared.opts.add_option(
        "forge_ai_max_tokens",
        shared.OptionInfo(16000, "Max tokens per assistant reply (Thinking models need room for <think>)", section=section),
    )
    shared.opts.add_option(
        "forge_ai_temperature",
        shared.OptionInfo(0.7, "LLM temperature", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.05}, section=section),
    )
    shared.opts.add_option(
        "forge_ai_provider",
        shared.OptionInfo("local", "AI provider: 'local' (text-gen) or 'claude' (Anthropic API)", section=section),
    )
    shared.opts.add_option(
        "forge_ai_claude_model",
        shared.OptionInfo("claude-sonnet-4-6", "Claude model id (when provider=claude)", section=section),
    )


def _opt(name, default):
    val = getattr(shared.opts, name, None)
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    return val


# ------------------------------------------------------------- providers

def _provider():
    return str(_opt("forge_ai_provider", "local")).strip().lower()


def _anthropic_key():
    """Read the Anthropic key from env or a key file — never from Forge config
    (which is world-readable). The user supplies their own key."""
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return k
    keyfile = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "anthropic_key.txt")
    try:
        with open(keyfile, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _to_anthropic(messages):
    """Convert OpenAI-style messages (as the JS sends) to Anthropic format:
    pull out the system prompt, turn image_url data-URLs into image blocks."""
    system = ""
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            system += (_content_text_full(content) + "\n")
            continue
        blocks = []
        if isinstance(content, str):
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    blocks.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    if url.startswith("data:"):
                        try:
                            header, b64 = url.split(",", 1)
                            media = header.split(";")[0].split(":")[1] if ":" in header else "image/jpeg"
                            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}})
                        except Exception:
                            pass
        if blocks:
            out.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
    return system.strip(), out


def _content_text_full(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


BRIDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge")


def _takeover_active():
    return os.path.exists(os.path.join(BRIDGE_DIR, "takeover_active"))


def _bridge_chat(messages, max_tokens):
    """Route a chat turn to a live Claude Code session via files. The extension
    writes the flattened conversation + image files, then waits for the Claude
    Code operator to write response.json. Lets a human-driven Claude Code agent
    'take over' the assistant with no API key."""
    os.makedirs(BRIDGE_DIR, exist_ok=True)
    if not _takeover_active():
        return {"error": "Claude Code takeover is not active. In your Claude Code session, tell it to take over the Forge session (it starts the bridge loop)."}

    for f in glob.glob(os.path.join(BRIDGE_DIR, "img*.jpg")):
        try:
            os.remove(f)
        except Exception:
            pass

    rid = str(int(time.time() * 1000))
    img_idx = 0
    lines = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = m.get("content")
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
        elif isinstance(content, list):
            parts = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        try:
                            with open(os.path.join(BRIDGE_DIR, f"img{img_idx}.jpg"), "wb") as fh:
                                fh.write(base64.b64decode(url.split(",", 1)[1]))
                            parts.append(f"[IMAGE {img_idx} — file img{img_idx}.jpg]")
                            img_idx += 1
                        except Exception:
                            pass
            lines.append(f"{role}: " + "\n".join(parts))

    full_prompt = "\n\n".join(lines)
    req = {"id": rid, "ts": time.strftime("%H:%M:%S"), "n_images": img_idx, "prompt": full_prompt}
    resp_path = os.path.join(BRIDGE_DIR, "response.json")
    try:
        os.remove(resp_path)
    except Exception:
        pass
    with open(os.path.join(BRIDGE_DIR, "request.json"), "w", encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False, indent=1)
    # also write the operator-friendly plain-text view: the request id header,
    # then the conversation. Lets the Claude Code operator just Read one file.
    with open(os.path.join(BRIDGE_DIR, "request.txt"), "w", encoding="utf-8") as f:
        f.write(f"REQUEST id={rid}  images={img_idx}  ts={req['ts']}\n")
        f.write("To respond: write response.json = {\"id\":\"" + rid + "\",\"reply\":\"...tool blocks...\"}\n")
        f.write("=" * 70 + "\n")
        f.write(full_prompt)

    deadline = time.time() + 1800
    while time.time() < deadline:
        if os.path.exists(resp_path):
            try:
                with open(resp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if str(data.get("id")) == rid:
                    os.remove(resp_path)
                    reply = data.get("reply", "")
                    _log_chat(messages, reply)
                    return {"reply": reply, "finish_reason": "stop"}
            except Exception:
                pass
        if not _takeover_active():
            return {"error": "Claude Code takeover ended before responding."}
        time.sleep(0.5)
    return {"error": "Claude Code operator did not respond in 30 min."}


def _claude_chat(messages, max_tokens, temperature):
    key = _anthropic_key()
    if not key:
        return {"error": "no Anthropic API key. Set ANTHROPIC_API_KEY in the environment, or put your key in extensions/forge-ai-assistant/anthropic_key.txt, then switch provider to claude."}
    system, conv = _to_anthropic(messages)
    body = {
        "model": str(_opt("forge_ai_claude_model", "claude-sonnet-4-6")),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system": system,
        "messages": conv,
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=340)
        r.raise_for_status()
        data = r.json()
        reply = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        _log_chat(messages, reply)
        return {"reply": reply, "finish_reason": data.get("stop_reason")}
    except Exception as e:
        detail = ""
        try:
            detail = r.text[:300]
        except Exception:
            pass
        _log_chat(messages, f"<CLAUDE ERROR: {e} {detail}>")
        return {"error": f"Claude request failed: {e} {detail}"}


# ---------------------------------------------------------------- helpers

def _api_base():
    return str(_opt("forge_ai_api_url", DEFAULT_API_URL)).rstrip("/")


def _api_port():
    base = _api_base()
    try:
        return int(base.rsplit(":", 1)[1].split("/")[0])
    except Exception:
        return 5000


def _api_ready():
    # /health responds even while the server is hibernated
    try:
        r = requests.get(_api_base() + "/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _proc_alive():
    p = _proc["popen"]
    return p is not None and p.poll() is None


def _pid_on_port(port):
    """Find the PID listening on a TCP port (covers a text-gen we didn't spawn)."""
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"], text=True, creationflags=subprocess.CREATE_NO_WINDOW
        )
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in line and parts[1].endswith(f":{port}"):
                return int(parts[-1])
    except Exception:
        pass
    return None


def _unload_forge_model():
    try:
        from modules import sd_models
        sd_models.unload_model_weights()
        return True
    except Exception as e:
        print(f"[forge-ai] sd_models.unload_model_weights failed: {e}")
    try:
        from backend import memory_management
        memory_management.unload_all_models()
        memory_management.soft_empty_cache()
        return True
    except Exception as e:
        print(f"[forge-ai] backend memory unload failed: {e}")
        return False


def _vram():
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return {"free_gb": round(free / 2**30, 2), "total_gb": round(total / 2**30, 2)}
    except Exception:
        return None


# ---------------------------------------------------------------- actions

def _llm_env():
    """Environment for the llama-server subprocess with PyTorch's CUDA runtime
    libs on PATH. llama-server needs cublas/cublasLt (CUDA 12); rather than bundle
    those ~750 MB NVIDIA redistributables in the repo (they also blow past GitHub's
    100 MB file limit), we reuse the identical ones the torch cu126 install already
    ships in torch/lib. Verified drop-in for the patched /sleep+/wake build."""
    env = os.environ.copy()
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            env["PATH"] = torch_lib + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    return env


def _models_dir():
    return str(_opt("forge_ai_models_dir", DEFAULT_MODELS_DIR))


def _is_mmproj(name):
    return "mmproj" in name.lower() and name.lower().endswith(".gguf")


def _list_models():
    d = _models_dir()
    files = sorted(glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True))
    return [os.path.relpath(f, d).replace("\\", "/") for f in files if not _is_mmproj(os.path.basename(f))]


def _find_mmproj(model):
    """If exactly one mmproj file sits next to the model, return its path."""
    if not model:
        return None
    folder = os.path.dirname(os.path.join(_models_dir(), model))
    try:
        candidates = [f for f in os.listdir(folder) if _is_mmproj(f)]
    except OSError:
        return None
    if len(candidates) == 1:
        return os.path.join(folder, candidates[0])
    return None


def _start_textgen(model=None):
    # One launch at a time: the panel button, message auto-start, and the
    # server-side restore thread can all fire together — without this lock two
    # server instances boot and the second fails on the taken port.
    with _start_lock:
        return _start_textgen_locked(model)


def _server_log_path():
    return os.path.join(os.path.dirname(str(_opt("forge_ai_server_bin", DEFAULT_SERVER_BIN))), "server.log")


_tail = {"thread": None}


def _start_log_tail():
    """Mirror server.log into the Forge console (prefixed [llama]).

    The server writes the log file directly (inherited handle — keeps logging
    even if Forge restarts), so tailing the FILE works for servers this Forge
    spawned AND for one adopted from a previous Forge process."""
    if _tail["thread"] is not None and _tail["thread"].is_alive():
        return

    def run():
        path = _server_log_path()
        f = None
        pos = 0
        while True:
            try:
                if f is None:
                    if not os.path.exists(path):
                        time.sleep(1.0)
                        continue
                    f = open(path, "rb")
                    f.seek(0, 2)          # attach at end — don't replay history
                    pos = f.tell()
                if os.path.getsize(path) < pos:   # file truncated (fresh boot)
                    f.close()
                    f = open(path, "rb")
                    pos = 0
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                pos = f.tell()
                txt = line.decode("utf-8", "replace").rstrip()
                if not txt:
                    continue
                try:
                    print("[llama] " + txt)
                except UnicodeEncodeError:
                    # Forge console may be cp1252 — degrade rather than die
                    print(("[llama] " + txt).encode("ascii", "replace").decode())
            except Exception:
                try:
                    if f:
                        f.close()
                except Exception:
                    pass
                f = None
                time.sleep(2.0)

    t = threading.Thread(target=run, daemon=True)
    _tail["thread"] = t
    t.start()
    print("[forge-ai] relaying llama-server log to this console ([llama] lines)")


# --- tie llama-server's lifetime to Forge's (Windows job object) -----------
# The child is assigned to a kill-on-close job owned by this process: when
# Forge exits — cleanly, crashed, or console closed with X — Windows kills the
# server too. No more orphaned llama-servers holding VRAM/binaries.

_job = {"handle": None}


def _make_kill_on_close_job():
    k32 = ctypes.windll.kernel32

    class _BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC),
                    ("IoInfo", _IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(job)
        return None
    return job


def _assign_to_job(popen):
    try:
        if _job["handle"] is None:
            _job["handle"] = _make_kill_on_close_job()
        if _job["handle"]:
            ctypes.windll.kernel32.AssignProcessToJobObject(_job["handle"], int(popen._handle))
            return True
    except Exception as e:
        print(f"[forge-ai] job-object attach failed ({e}) — server will not auto-die with Forge")
    return False


def _sanitize_extra_args(raw):
    """Translate legacy text-gen-era args into native llama-server flags.

    The extra_args setting persisted in config.json may predate the switch off
    text-gen (2026-07-07); unknown flags make llama-server exit instantly, so
    convert what has an equivalent and drop what doesn't.
    """
    tokens = shlex.split(str(raw))
    out = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if t in ("--nowebui", "--api", "--listen"):
            i += 1
            continue
        if t in ("--api-port", "--listen-port"):
            i += 2
            continue
        if t == "--cache-type" and nxt:
            out += ["--cache-type-k", nxt, "--cache-type-v", nxt]
            i += 2
            continue
        if t == "--extra-flags" and nxt:
            # text-gen's comma format: flag=value,flag2
            for part in nxt.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" in part:
                    k, v = part.split("=", 1)
                    out += ["--" + k.lstrip("-"), v]
                else:
                    out.append("--" + part.lstrip("-"))
            i += 2
            continue
        out.append(t)
        i += 1
    return out


def _start_textgen_locked(model=None):
    want = model or str(_opt("forge_ai_model", "")) or None

    # fast path: server already up — wake it (or restart for a model switch)
    if _api_ready():
        current = _model_name_loaded()
        if current and (not want or _same_model(current, want)):
            _llm_wake()   # no-op if awake; reclaims VRAM if hibernated
            # the hibernate path never respawns, so the park flag must be
            # cleared HERE too — otherwise the restore watchdog loops forever
            _auto["stopped_for_gen"] = False
            return {"ok": True, "already_running": True}
        # different model requested: llama-server loads one model per process,
        # so a switch is a restart
        print(f"[forge-ai] switching model {current} -> {want} (server restart)")
        _stop_textgen("kill")
        _wait_textgen_gone(8.0)

    # a boot we launched is still coming up — don't start a second instance
    if _proc_alive():
        return {"ok": True, "already_starting": True}

    # something unresponsive is squatting on the API port (half-dead boot)
    zombie = _pid_on_port(_api_port())
    if zombie:
        print(f"[forge-ai] killing unresponsive process on port {_api_port()} (pid {zombie})")
        subprocess.call(["taskkill", "/PID", str(zombie), "/T", "/F"], creationflags=subprocess.CREATE_NO_WINDOW)
        _wait_textgen_gone(8.0)

    if bool(_opt("forge_ai_auto_unload", True)):
        _unload_forge_model()

    exe = str(_opt("forge_ai_server_bin", DEFAULT_SERVER_BIN))
    if not os.path.isfile(exe):
        return {"ok": False, "error": f"llama-server binary not found: {exe}"}
    if not want:
        return {"ok": False, "error": "no model selected"}
    model_path = os.path.join(_models_dir(), want)
    if not os.path.isfile(model_path):
        return {"ok": False, "error": f"model file not found: {model_path}"}

    args = [exe, "--model", model_path, "--host", "127.0.0.1", "--port", str(_api_port())]
    args += _sanitize_extra_args(_opt("forge_ai_extra_args", ""))
    if "--parallel" not in args:
        # 2 slots (conversation + judge) even with a stale saved setting —
        # more slots = unified-KV eviction fights and prompt-processing thrash
        args += ["--parallel", "2"]
    mmproj = _find_mmproj(want)
    if mmproj:
        args += ["--mmproj", mmproj]
        print(f"[forge-ai] vision: using mmproj {os.path.basename(mmproj)}")

    # server writes server.log directly via an inherited handle (keeps working
    # across Forge restarts); the tail thread mirrors it into this console
    log_path = os.path.join(os.path.dirname(exe), "server.log")
    _start_log_tail()
    with open(log_path, "ab") as log_f:
        _proc["popen"] = subprocess.Popen(
            args, cwd=os.path.dirname(exe), stdout=log_f, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW, env=_llm_env()
        )
    _assign_to_job(_proc["popen"])   # dies with Forge, no matter how Forge exits
    _auto["stopped_for_gen"] = False
    print(f"[forge-ai] launched llama-server (pid {_proc['popen'].pid}): {' '.join(args[1:])}")

    # instant-exit means bad args / missing DLLs — surface the log right away
    time.sleep(1.5)
    if _proc["popen"].poll() is not None:
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-8:]).strip()
        except Exception:
            pass
        _proc["popen"] = None
        print(f"[forge-ai] llama-server exited immediately:\n{tail}")
        return {"ok": False, "error": f"llama-server exited immediately — see {log_path}", "log_tail": tail}
    return {"ok": True, "pid": _proc["popen"].pid}


def _same_model(loaded_path, want):
    """Compare the server's reported model path against a models-dir-relative name."""
    if not loaded_path or not want:
        return False
    return os.path.normcase(os.path.basename(str(loaded_path))) == \
           os.path.normcase(os.path.basename(str(want)))


def _model_name_loaded():
    """Basename of the model the server is running (survives hibernate), or None."""
    try:
        r = requests.get(_api_base() + "/props", timeout=2)
        if r.status_code == 200:
            path = r.json().get("model_path")
            if path:
                return os.path.basename(path)
    except Exception:
        pass
    return None


def _llm_sleeping():
    try:
        r = requests.get(_api_base() + "/props", timeout=2)
        return r.status_code == 200 and r.json().get("is_sleeping") is True
    except Exception:
        return False


def _llm_sleep():
    """Fast VRAM hibernate (patched llama-server): weights + KV cache move to
    pinned host RAM and ALL VRAM is freed (~1.7s). The encoded conversation,
    including image embeddings, survives — wake needs no reload or re-encode."""
    try:
        r = requests.post(_api_base() + "/sleep", timeout=120)
        return r.status_code == 200 and r.json().get("is_sleeping") is True
    except Exception:
        return False


def _llm_wake():
    """Reclaim VRAM after a hibernate (~1.5s). No-op if not sleeping. Any chat
    request also auto-wakes the server; calling this early just hides latency."""
    try:
        r = requests.post(_api_base() + "/wake", timeout=120)
        return r.status_code == 200 and r.json().get("is_sleeping") is False
    except Exception:
        return False


def _wait_textgen_gone(timeout=8.0):
    """Block until the server process is dead (VRAM released)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _proc_alive() and _pid_on_port(_api_port()) is None:
            return True
        time.sleep(0.3)
    return False


def _stop_textgen(mode="soft"):
    # soft: hibernate — ALL VRAM freed in ~1.7s, server stays up, KV cache
    # (incl. encoded images) kept in pinned host RAM for a ~1.5s wake.
    if mode == "soft" and _api_ready():
        if _llm_sleep():
            print("[forge-ai] LLM hibernated (VRAM freed, KV cache kept in RAM)")
            return {"ok": True, "soft": True}
    killed = []
    if _proc_alive():
        pid = _proc["popen"].pid
        subprocess.call(
            ["taskkill", "/PID", str(pid), "/T", "/F"], creationflags=subprocess.CREATE_NO_WINDOW
        )
        killed.append(pid)
        _proc["popen"] = None
    else:
        # maybe the user started the server themselves — find it by port
        pid = _pid_on_port(_api_port())
        if pid:
            subprocess.call(
                ["taskkill", "/PID", str(pid), "/T", "/F"], creationflags=subprocess.CREATE_NO_WINDOW
            )
            killed.append(pid)
    return {"ok": True, "killed": killed}


# ------------------------------------------------------------- AI memory

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_memory.json")

# Forge UI snapshot (settings & prompts per tab) — restored by the ↺ button.
SESSION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "last_session.json")

# The bot's own state (chat, run log, best/reference images) — autosaved and
# auto-restored every session until the user clears it with the 🗑 button.
BOT_STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot_state.json")

# Named settings profiles ({name: {ts, uiSnapshots, uiActiveTab}}) — saved via
# the 💾 button next to the profile dropdown in Forge's top bar.
PROFILES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui_profiles.json")


def _profiles_load():
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _profiles_store(p):
    tmp = PROFILES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(p, f)
    os.replace(tmp, PROFILES_FILE)


def _session_save(state):
    tmp = SESSION_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, SESSION_FILE)   # atomic — a crash mid-write can't corrupt the file


def _session_load():
    if not os.path.exists(SESSION_FILE):
        return None
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
CHAT_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_log.jsonl")
GUIDANCE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live_guidance.txt")
_memory_lock = threading.Lock()


def _read_guidance():
    try:
        with open(GUIDANCE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _write_guidance(text):
    try:
        with open(GUIDANCE_FILE, "w", encoding="utf-8") as f:
            f.write(str(text))
    except Exception:
        pass


def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _log_chat(messages, reply):
    """Append the exchange to a reviewable transcript (no images)."""
    try:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = _content_text(m.get("content"))
                break
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "user": last_user[:2000],
                "reply": str(reply)[:4000],
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            notes = json.load(f)
            return notes if isinstance(notes, list) else []
    except Exception:
        return []


def _add_memory(note, checkpoint=None, tab=None):
    with _memory_lock:
        notes = _load_memory()
        notes.append({
            "ts": time.strftime("%Y-%m-%d"),
            "checkpoint": (checkpoint or "")[:80],
            "tab": (tab or "")[:40],
            "note": str(note)[:400],
        })
        notes = notes[-200:]   # keep the most recent 200
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=1)
        return len(notes)


# ---------------------------------------------------- server-side restore

def _restore_worker():
    """Wait for the generation queue to go idle, then reload the LLM.

    Runs server-side so restore works even if no browser tab is open/awake.
    """
    try:
        idle = 0
        deadline = time.time() + 3600
        while idle < 2 and time.time() < deadline:
            time.sleep(3)
            try:
                busy = (getattr(shared.state, "job_count", 0) or 0) > 0
            except Exception:
                busy = False
            idle = 0 if busy else idle + 1
        if _auto["stopped_for_gen"] and bool(_opt("forge_ai_auto_restore", True)):
            print("[forge-ai] generations finished — auto-restoring the LLM")
            result = _start_textgen(None)
            print(f"[forge-ai] auto-restore: {result}")
    except Exception as e:
        print(f"[forge-ai] auto-restore failed: {e}")
    finally:
        _auto["restoring"] = False


def _spawn_restore_thread():
    if not bool(_opt("forge_ai_auto_restore", True)):
        return
    with _restore_lock:
        if _auto["restoring"]:
            return
        _auto["restoring"] = True
    threading.Thread(target=_restore_worker, daemon=True).start()


# ------------------------------------------------- generation VRAM guard

class ForgeAIAssistantScript(scripts_mod.Script):
    """Runs at the start of every generation (UI, Replacer, API — anything).

    If the LLM is holding the VRAM, kill it and wait for the release so the
    Forge model can load. The browser-side watchdog restarts the LLM after
    the job finishes (if 'auto restore' is enabled).
    """

    def title(self):
        return "Forge AI Assistant"

    def show(self, is_img2img):
        return scripts_mod.AlwaysVisible

    def process(self, p, *args):
        if _api_ready():
            if _model_name_loaded():
                print("[forge-ai] generation starting — soft-unloading LLM to free VRAM")
                result = _stop_textgen("soft")
                if not result.get("soft"):
                    _wait_textgen_gone(8.0)
                _auto["stopped_for_gen"] = True
                _spawn_restore_thread()
        elif _proc_alive():
            # process exists but API not up (mid-boot) — can't soft-unload
            print("[forge-ai] generation starting — killing booting llama-server to free VRAM")
            _stop_textgen("kill")
            _wait_textgen_gone(8.0)
            _auto["stopped_for_gen"] = True
            _spawn_restore_thread()


# ---------------------------------------------------------------- routes

def _auto_start_worker():
    """Start the LLM with Forge: replace any leftover server with one whose
    lifetime is tied to THIS Forge, then hibernate it so Forge keeps all VRAM
    until the first chat message wakes it (~1.5s)."""
    try:
        time.sleep(4.0)   # let Forge finish wiring its UI/queues
        if _provider() != "local":
            return
        # a server from a previous Forge run isn't in our job object and would
        # outlive us — replace it with one that stops and starts with Forge
        pid = _pid_on_port(_api_port())
        if pid:
            print(f"[forge-ai] replacing leftover llama-server (pid {pid}) with one tied to this Forge")
            subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], creationflags=subprocess.CREATE_NO_WINDOW)
            _wait_textgen_gone(8.0)
        result = _start_textgen(None)
        print(f"[forge-ai] auto-start with Forge: {result}")
        if not result.get("ok"):
            return
        for _ in range(120):   # wait for the model to finish loading
            if _api_ready() and _model_name_loaded():
                break
            time.sleep(2)
        # park it: VRAM goes back to Forge, KV stays warm in pinned RAM
        if _llm_sleep():
            print("[forge-ai] LLM ready and hibernated — first chat message wakes it in ~1.5s")
    except Exception as e:
        print(f"[forge-ai] auto-start failed: {e}")


def on_app_started(demo, app):

    # if a llama-server from a previous Forge run is still alive (hibernate
    # keeps it up across restarts), start relaying its log right away
    _start_log_tail()

    # one-time fix: SAM's GroundingDINO pip-build always fails in this venv
    # (no pip module inside build isolation) — the extension's own fallback
    # then kicks in anyway. Setting this skips the doomed 30s install attempt.
    try:
        if not shared.opts.data.get("sam_use_local_groundingdino", False):
            shared.opts.set("sam_use_local_groundingdino", True)
            shared.opts.save(shared.config_filename)
            print("[forge-ai] enabled sam_use_local_groundingdino (skips the failing pip build)")
    except Exception as e:
        print(f"[forge-ai] could not set sam_use_local_groundingdino: {e}")

    # start the LLM alongside Forge (it dies with Forge via the job object)
    if bool(_opt("forge_ai_auto_start", True)):
        threading.Thread(target=_auto_start_worker, daemon=True).start()

    @app.get("/forge-ai/status")
    def status():
        prov = _provider()
        if prov == "claude_code":
            return {
                "provider": "claude_code",
                "claude_ready": _takeover_active(),
                "model_loaded": "claude-code (you)" if _takeover_active() else None,
                "textgen_proc": False,
                "textgen_api_ready": _takeover_active(),
                "stopped_for_gen": False,
                "auto_restore": False,
                "vram": _vram(),
            }
        if prov == "claude":
            return {
                "provider": "claude",
                "claude_model": str(_opt("forge_ai_claude_model", "claude-sonnet-4-6")),
                "claude_ready": bool(_anthropic_key()),
                # claude runs in the cloud — no local model / no VRAM juggle needed
                "textgen_proc": False,
                "textgen_api_ready": True,
                "model_loaded": "claude:" + str(_opt("forge_ai_claude_model", "claude-sonnet-4-6")),
                "stopped_for_gen": False,
                "auto_restore": False,
                "vram": _vram(),
            }
        api_ready = _api_ready()
        return {
            "provider": "local",
            "textgen_proc": _proc_alive(),
            "textgen_api_ready": api_ready,
            "model_loaded": _model_name_loaded() if api_ready else None,
            "sleeping": _llm_sleeping() if api_ready else False,
            "stopped_for_gen": _auto["stopped_for_gen"],
            "auto_restore": bool(_opt("forge_ai_auto_restore", True)),
            "vram": _vram(),
        }

    @app.post("/forge-ai/provider")
    def set_provider(payload: dict = Body(...)):
        p = str(payload.get("provider", "")).strip().lower()
        if p not in ("local", "claude", "claude_code"):
            return {"ok": False, "error": "provider must be 'local', 'claude', or 'claude_code'"}
        shared.opts.set("forge_ai_provider", p)
        shared.opts.save(shared.config_filename)
        ready = None
        if p == "claude":
            ready = bool(_anthropic_key())
        elif p == "claude_code":
            ready = _takeover_active()
        return {"ok": True, "provider": p, "claude_ready": ready}

    # operator convenience: check whether a request is waiting, and respond
    @app.get("/forge-ai/bridge/pending")
    def bridge_pending():
        req_path = os.path.join(BRIDGE_DIR, "request.json")
        if not os.path.exists(req_path):
            return {"pending": False, "active": _takeover_active()}
        try:
            with open(req_path, "r", encoding="utf-8") as f:
                req = json.load(f)
            resp_path = os.path.join(BRIDGE_DIR, "response.json")
            answered = os.path.exists(resp_path)
            return {"pending": not answered, "active": _takeover_active(),
                    "id": req.get("id"), "n_images": req.get("n_images"), "ts": req.get("ts")}
        except Exception as e:
            return {"pending": False, "error": str(e)}

    @app.post("/forge-ai/bridge/respond")
    def bridge_respond(payload: dict = Body(...)):
        rid = str(payload.get("id", ""))
        reply = str(payload.get("reply", ""))
        if not rid:
            return {"ok": False, "error": "id required"}
        with open(os.path.join(BRIDGE_DIR, "response.json"), "w", encoding="utf-8") as f:
            json.dump({"id": rid, "reply": reply}, f, ensure_ascii=False)
        return {"ok": True}

    # let the Claude Code operator arm/disarm takeover and poll the bridge
    @app.post("/forge-ai/takeover")
    def takeover(payload: dict = Body(default={})):
        os.makedirs(BRIDGE_DIR, exist_ok=True)
        on = bool(payload.get("active", True))
        flag = os.path.join(BRIDGE_DIR, "takeover_active")
        if on:
            open(flag, "w").close()
            shared.opts.set("forge_ai_provider", "claude_code")
            shared.opts.save(shared.config_filename)
        else:
            try:
                os.remove(flag)
            except Exception:
                pass
            shared.opts.set("forge_ai_provider", "local")
            shared.opts.save(shared.config_filename)
        return {"ok": True, "active": on}

    @app.get("/forge-ai/models")
    def models():
        return {"models": _list_models()}

    @app.post("/forge-ai/textgen/start")
    def start(payload: dict = Body(default={})):
        return _start_textgen(payload.get("model") or None)

    @app.post("/forge-ai/textgen/stop")
    def stop(payload: dict = Body(default={})):
        result = _stop_textgen(payload.get("mode", "soft"))
        _auto["stopped_for_gen"] = False   # deliberate stop — don't auto-restore
        return result

    @app.post("/forge-ai/forge/unload")
    def unload():
        return {"ok": _unload_forge_model(), "vram": _vram()}

    @app.get("/forge-ai/guidance")
    def guidance_get():
        return {"guidance": _read_guidance()}

    @app.post("/forge-ai/guidance")
    def guidance_set(payload: dict = Body(...)):
        _write_guidance(str(payload.get("guidance", "")))
        return {"ok": True}

    @app.post("/forge-ai/session/save")
    def session_save(payload: dict = Body(...)):
        try:
            # defense in depth: a page whose capture produced nothing (stale tab from
            # an old build, mid-load unload) must never clobber a real snapshot
            if not payload.get("uiSnapshots"):
                existing = _session_load()
                if existing and existing.get("uiSnapshots"):
                    return {"ok": True, "skipped": "empty snapshot ignored"}
            _session_save(payload)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/forge-ai/profiles")
    def profiles_list():
        try:
            return {"profiles": sorted(_profiles_load().keys())}
        except Exception as e:
            return {"profiles": [], "error": str(e)}

    @app.post("/forge-ai/profiles/save")
    def profiles_save(payload: dict = Body(...)):
        name = str(payload.get("name", "")).strip()
        if not name:
            return {"ok": False, "error": "empty name"}
        try:
            p = _profiles_load()
            p[name] = payload.get("state") or {}
            _profiles_store(p)
            return {"ok": True, "profiles": sorted(p.keys())}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/forge-ai/profiles/get")
    def profiles_get(name: str = ""):
        try:
            p = _profiles_load()
            if name not in p:
                return {"ok": False, "error": f"no profile named '{name}'"}
            return {"ok": True, "state": p[name]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/forge-ai/botstate/save")
    def botstate_save(payload: dict = Body(...)):
        try:
            tmp = BOT_STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, BOT_STATE_FILE)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/forge-ai/botstate/latest")
    def botstate_latest():
        try:
            if not os.path.exists(BOT_STATE_FILE):
                return {"exists": False}
            with open(BOT_STATE_FILE, "r", encoding="utf-8") as f:
                return {"exists": True, "state": json.load(f)}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    @app.post("/forge-ai/botstate/clear")
    def botstate_clear():
        try:
            if os.path.exists(BOT_STATE_FILE):
                os.remove(BOT_STATE_FILE)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/forge-ai/session/info")
    def session_info():
        # lightweight existence/timestamp check — used for the startup hint
        try:
            state = _session_load()
            if state is None:
                return {"exists": False}
            return {"exists": True, "ts": state.get("ts"), "tabs": len(state.get("uiSnapshots", {}))}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    @app.get("/forge-ai/session/latest")
    def session_latest():
        try:
            state = _session_load()
            if state is None:
                return {"exists": False}
            return {"exists": True, "state": state}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    @app.get("/forge-ai/memory")
    def memory():
        return {"notes": _load_memory()}

    @app.post("/forge-ai/memory")
    def memory_add(payload: dict = Body(...)):
        note = str(payload.get("note", "")).strip()
        if not note:
            return {"ok": False, "error": "empty note"}
        count = _add_memory(note, payload.get("checkpoint"), payload.get("tab"))
        return {"ok": True, "count": count}

    @app.post("/forge-ai/chat")
    def chat(payload: dict = Body(...)):
        max_tokens = int(payload.get("max_tokens", _opt("forge_ai_max_tokens", 800)))
        temperature = float(payload.get("temperature", _opt("forge_ai_temperature", 0.7)))

        # route to a live Claude Code session (file bridge) — no API key
        if _provider() == "claude_code":
            return _bridge_chat(payload.get("messages", []), max_tokens)
        # route to Claude (cloud API) when selected
        if _provider() == "claude":
            return _claude_chat(payload.get("messages", []), max_tokens, temperature)

        if not _api_ready():
            return {"error": "llama-server is not running. Start the LLM first."}
        if not _model_name_loaded():
            return {"error": "llama-server is up but no model is loaded. Wait for the auto-restore or press Start."}
        body = {
            "messages": payload.get("messages", []),
            "max_tokens": max_tokens,
            "temperature": temperature,
            # keep the longest shared prefix (system prompt + encoded images) in
            # the KV cache across turns — critical for skipping image re-encodes
            "cache_prompt": True,
            # PIN slots: the main conversation always lives in slot 0, blind-judge
            # calls in slot 1. Without this, llama-server's LCP/LRU routing bounces
            # the two prompt families across slots, evicting the conversation's
            # cached images and forcing a full ~70s re-read (seen in server.log).
            "id_slot": 1 if payload.get("judge") else 0,
        }
        # a freshly (re)loaded model can 5xx for a few seconds while warming up;
        # retry a couple of times before surfacing the error
        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(_api_base() + "/v1/chat/completions", json=body, timeout=890)
                if r.status_code >= 500:
                    last_err = f"{r.status_code} from llama-server"
                    time.sleep(4)
                    continue
                r.raise_for_status()
                data = r.json()
                reply = data["choices"][0]["message"]["content"]
                _log_chat(body["messages"], reply)
                return {"reply": reply, "finish_reason": data["choices"][0].get("finish_reason")}
            except Exception as e:
                last_err = str(e)
                time.sleep(4)
        _log_chat(body["messages"], f"<ERROR: {last_err}>")
        return {"error": f"chat request failed after retries: {last_err}"}


script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_settings(on_ui_settings)
