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
import re
import glob
import json
import time
import base64
import shlex
import ctypes
import signal
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


def _default_server_bin():
    """Locate the llama-server binary.

    Windows ships a prebuilt one at forge-llm/. On Linux/macOS start.sh compiles
    it per backend into forge-llm/<backend>/, so look there too -- otherwise the
    freshly built server is invisible and the assistant reports "binary not
    found" pointing at the bare forge-llm/ path. Ordered by preference: a
    discrete-GPU backend beats the CPU-ish fallbacks.
    """
    root = os.path.join(_PROJECT_ROOT, "forge-llm")
    for sub in ("", "cuda", "rocm", "vulkan", "metal"):
        cand = os.path.join(root, sub, _LLAMA_EXE) if sub else os.path.join(root, _LLAMA_EXE)
        if os.path.isfile(cand):
            return cand
    return os.path.join(root, _LLAMA_EXE)   # nothing built yet: report the canonical path


DEFAULT_SERVER_BIN = _default_server_bin()
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


_api_ready_cache = {"t": 0.0, "v": False}


def _api_ready(max_age=5.0):
    # /health responds even while the server is hibernated.
    # CACHED + short-circuited: the UI watchdog polls status on a timer, and
    # with no llama-server running every probe used to burn a full 2s timeout
    # on a threadpool worker. If we never launched a process and nothing is
    # listening, the answer is no -- cheaply.
    now = time.time()
    if now - _api_ready_cache["t"] < max_age:
        return _api_ready_cache["v"]
    ok = False
    try:
        if _proc_alive() or _pid_on_port(_api_port()):
            r = requests.get(_api_base() + "/health", timeout=1)
            ok = r.status_code == 200
    except Exception:
        ok = False
    _api_ready_cache.update(t=now, v=ok)
    return ok


def _proc_alive():
    p = _proc["popen"]
    return p is not None and p.poll() is None


_atexit_registered = False


def _register_atexit_kill():
    """POSIX counterpart to the Windows job object: kill the server when Forge exits.

    The job object handles every Windows exit path, but POSIX has no equivalent,
    so without this a clean shutdown leaves llama-server running and holding
    several GB of VRAM. Registered once, on first launch.
    """
    global _atexit_registered
    if _atexit_registered or os.name == "nt":
        return
    _atexit_registered = True
    import atexit

    def _kill_llm_atexit():
        p = _proc.get("popen")
        if p is None or p.poll() is not None:
            return
        try:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        except Exception:
            pass

    atexit.register(_kill_llm_atexit)


def _kill_process_or_pid(pid):
    """Kill a process by PID (cross-platform)."""
    try:
        if os.name == "nt":
            subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception as e:
        print(f"[forge-ai] kill process {pid} failed: {e}")


def _pid_on_port(port):
    """Find the PID listening on a TCP port (covers a text-gen we didn't spawn)."""
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "tcp"], text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line and parts[1].endswith(f":{port}"):
                    return int(parts[-1])
        else:
            # psutil, not `fuser`: fuser lives in psmisc and is absent on plenty
            # of distros (and in slim containers), where it would fail silently
            # and we'd never reap a stale server. psutil is already a hard
            # dependency of this project.
            import psutil
            for c in psutil.net_connections(kind="tcp"):
                if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                    return int(c.pid)
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
    if os.name != "nt":
        # Where our per-backend llama.cpp .so files live. PREPEND to any existing
        # LD_LIBRARY_PATH rather than replacing it, and never fold PATH into it:
        # PATH holds executable dirs, and mixing it in makes the dynamic linker
        # stat every one of them looking for shared objects.
        _lib_dirs = [
            os.path.join(_PROJECT_ROOT, "forge-llm", "vulkan"),
            os.path.join(_PROJECT_ROOT, "forge-llm", "rocm"),
            os.path.join(_PROJECT_ROOT, "forge-llm", "cuda"),
            os.path.join(_PROJECT_ROOT, "forge-llm"),
        ]
        _existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [d for d in _lib_dirs if os.path.isdir(d)] + ([_existing] if _existing else [])
        )
        env["LLAMA_VK_EXCLUSIVE_FILL"] = "1"
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

if os.name == "nt":
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
else:
    def _make_kill_on_close_job(): pass
    def _assign_to_job(popen): pass  # no-op on Linux (process groups handle this)


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
        _kill_process_or_pid(zombie)
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
        popen_kwargs = dict(
            args=args, cwd=os.path.dirname(exe), stdout=log_f, stderr=subprocess.STDOUT,
            env=_llm_env(),
        )
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        # POSIX: do NOT pass start_new_session=True here. setsid() detaches the
        # server from our process group, which is the opposite of what we want --
        # it would survive Forge exiting and keep holding its VRAM. Staying in
        # the group means a Ctrl-C / terminal hangup reaches it too, and
        # _kill_llm_atexit below covers the clean-exit path.
        _proc["popen"] = subprocess.Popen(**popen_kwargs)
    _assign_to_job(_proc["popen"])   # Windows: dies with Forge however Forge exits
    _register_atexit_kill()          # POSIX equivalent
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


def _wait_llm_parked(timeout=30.0):
    """Block until the LLM has ACTUALLY released its VRAM.

    /sleep answering 200 does not mean the memory is back. llama-server will not
    hibernate while a completion is in flight, and a long vision reply runs for
    tens of seconds: the console shows the sleep request, then twenty more
    seconds of `n_decoded` climbing. Returning immediately on that 200 let Forge
    start a generation into a card that was still 94% full -- which is how a run
    sat at progress 0.0, and how another died mid-sampling with "weight is on
    cpu, different from other tensors on cuda:0" after the memory manager
    evicted weights under the pressure.

    Poll /props, which reports the real hibernation state, rather than trusting
    the acknowledgement.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _llm_sleeping():
            return True
        time.sleep(0.4)
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
        _kill_process_or_pid(pid)
        killed.append(pid)
        _proc["popen"] = None
    else:
        # maybe the user started the server themselves — find it by port
        pid = _pid_on_port(_api_port())
        if pid:
            _kill_process_or_pid(pid)
            killed.append(pid)
    return {"ok": True, "killed": killed}


# ------------------------------------------------------------- AI memory

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai_memory.json")

# Forge UI snapshot (settings & prompts per tab) — restored by the ↺ button.
# The test harness copies config.json to a temp file so a run can never touch
# personal settings, but this lived at a fixed path and was written by every
# UI-tier run -- which opens img2img, clicks around inside it and generates at
# 128x128. That is how a tab nobody had configured got into a real session, and
# from there into a real restore. Honour an override so the harness can isolate
# it the same way.
SESSION_FILE = os.environ.get("FORGE_AI_SESSION_FILE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "last_session.json")

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


UI_CONFIG_FILE = os.path.join(_PROJECT_ROOT, "ui-config.json")


def _norm_label(label):
    """Our capture labels carry decorations ui-config.json does not.

    We prefix a control with its section ("Hires. fix > Denoising strength") and
    suffix duplicates ("Save mask #2"); ui-config keys end in "<tab>/<label>".
    Strip both so the two can be compared.
    """
    label = re.sub(r"\s*#\d+$", "", label)
    if " > " in label:
        label = label.rsplit(" > ", 1)[-1]
    return label.strip()


# Display tab name -> the token ui-config actually uses for it.
_UI_CONFIG_TAB_ALIAS = {
    "checkpoint merger": "modelmerger",
    "png info": "pnginfo",
    "spaces": "space",
    "model downloader": "model_downloader",
}

_ui_config_cache = {"mtime": None, "exact": None, "labels": None, "steps": None,
                    "plain": None, "plain_steps": None}


def _ui_config_index():
    """(tab, label) -> {default values} for EVERY ui-config key shape.

    The first cut of this only looked at "<tab>/<label>/value" and pruned 29 of
    403 entries. That is 741 of the file's 1974 keys: script and extension
    controls are stored as "customscript/<script>.py/<tab>/<label>/value" (1233
    keys), and some tabs key off their id rather than their display name
    ("modelmerger", "pnginfo") or a capitalised name ("Replacer"). Indexing on
    the LAST TWO path segments catches all of them at once.
    """
    empty = ({}, {}, {}, {}, {})
    try:
        mtime = os.path.getmtime(UI_CONFIG_FILE)
    except OSError:
        return empty
    if _ui_config_cache["mtime"] == mtime:
        return (_ui_config_cache["exact"], _ui_config_cache["labels"],
                _ui_config_cache["steps"], _ui_config_cache["plain"],
                _ui_config_cache["plain_steps"])
    exact, labels, steps, plain, plain_steps = {}, {}, {}, {}, {}
    try:
        with open(UI_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return empty
    for key, val in (cfg or {}).items():
        for suffix, sink in (("/value", exact), ("/step", steps)):
            if not key.endswith(suffix):
                continue
            parts = key[: -len(suffix)].split("/")
            if len(parts) < 2:
                continue
            tab, label = parts[-2].strip().lower(), parts[-1].strip()
            sink.setdefault((tab, label), set()).add(str(val))   # list defaults are unhashable
            # A PLAIN "<tab>/<label>" key is the tab's own control. Indexing on
            # the last two segments alone lumps it together with every script
            # that happens to reuse the name -- img2img's Width (1024) with
            # Ultimate SD Upscale's tile Width (64), Sampling steps 25 with 20,
            # Mask blur 4 with 8 -- and "every recorded default must agree" then
            # kept all of them forever. That single collision was most of what
            # survived pruning in a tab nobody had touched.
            if len(parts) == 2:
                (plain if sink is exact else plain_steps).setdefault(
                    (tab, label), set()).add(str(val))
            if sink is exact:
                labels.setdefault(tab, set()).add(label)
    _ui_config_cache.update({"mtime": mtime, "exact": exact, "labels": labels,
                             "steps": steps, "plain": plain,
                             "plain_steps": plain_steps})
    return exact, labels, steps, plain, plain_steps


def _match_key(exact, labels, tabs, label):
    """The (tab, label) ui-config records this captured control under, or None."""
    base = _norm_label(label)
    for tab in tabs:
        for cand in (label, base):
            if (tab, cand) in exact:
                return (tab, cand)
    # Some captured labels append the control's description ("Schedule bias
    # Shifts when preservation of original content occurs..."); match the
    # longest recorded label our label starts with.
    for tab in tabs:
        best = None
        for recorded in labels.get(tab, ()):
            if len(recorded) >= 5 and base.startswith(recorded):
                if best is None or len(recorded) > len(best):
                    best = recorded
        if best is not None:
            return (tab, best)
    return None


def _as_number(val):
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return None


def _is_default_value(val, known, step):
    """Is this captured value the recorded default?

    Compared NUMERICALLY, which is the whole point. The browser hands us what
    the DOM holds -- "1", "7", "0" -- while ui-config holds what python built
    the control with -- 1.0, 7.0, 0.0. String equality said those differed, so
    every untouched slider in the lazily-built img2img tab survived pruning:
    160 of the 216 entries a "pruned" session still carried were ControlNet,
    FreeU, PAG and LatentModifier sliders sitting at exactly their default.

    A slider also QUANTIZES to its step, so what the DOM reports is not always
    what python passed -- Auto SAM's crop_overlap_ratio is built as
    0.3413333333333333 and reads back 0.34. Half a step of tolerance closes
    that gap without letting a real neighbouring position through.

    Requires agreement from EVERY recorded default under this (tab, label). Two
    scripts in one tab can share a short label with different defaults, and
    when they disagree we cannot tell which control we are looking at -- so we
    keep the entry rather than guess.
    """
    if not known:
        return False
    num = _as_number(val)
    tol = 0.0
    if step:
        widest = max((_as_number(s) or 0.0) for s in step)
        tol = abs(widest) / 2.0
    for recorded in known:
        if str(recorded) == str(val):
            continue
        # "nothing selected" is written two ways: a multiselect records [] and
        # reads back as an empty string (Styles).
        if str(recorded).strip() in ("", "[]") and str(val).strip() in ("", "[]"):
            continue
        if isinstance(recorded, bool) or str(recorded).lower() in ("true", "false"):
            if str(recorded).lower() == str(val).strip().lower():
                continue
            return False
        rnum = _as_number(recorded)
        if num is None or rnum is None:
            return False
        if abs(rnum - num) > tol + 1e-9:
            return False
    return True


def _shape(val):
    s = str(val).strip().lower()
    if s in ("true", "false"):
        return "bool"
    return "num" if _as_number(s) is not None else "text"


def _shapes_agree(val, known):
    """Does the record we matched even describe a control of this type?

    Labels are matched by text, so collisions happen: the "Hires. fix"
    InputAccordion toggle (false) lands on a "Hires. fix" DROPDOWN recorded as
    "Follow txt2img", and the PNG-info checkboxes land on the prompt/seed fields
    they are named after. A bool against a string is not a value that differs
    from its default -- it is the wrong record, and pretending otherwise kept
    those entries in the session forever. Treat it as no record at all and let
    the self-evidently-unset checks below decide.
    """
    return any(_shape(r) == _shape(val) for r in (known or ()))


def _is_unidentifiable(label):
    """A capture label that does not name any control.

    A double-ended slider with no <label> of its own -- ControlNet's timestep
    ranges -- is captured as a bare "(start)" / "(end)", and an unlabelled block
    can arrive as "#2" or "Dropdown". Nothing can be done with these. ui-config
    has no such key, so pruning can never judge them and always keeps them; that
    is what kept an img2img tab in the session after everything real had been
    removed from it. RESTORE is worse than useless: it matches the first control
    carrying the same decoration, which in a tab holding three ControlNet units
    is a coin toss into the wrong unit.

    A section prefix rescues the label ("ControlNet Unit 1 > (start)" does name
    something), so only bare decorations are dropped.
    """
    if " > " in label:
        return False
    base = re.sub(r"\s*#\d+$", "", label).strip()
    return base in ("", "(start)", "(end)", "Dropdown")


def _is_first_choice(label, val):
    """DISABLED -- kept only so old callers do not break. See below.

    The idea was that an unlabelled radio captured under its first option's text
    ("Balanced" = "Balanced") is sitting on its build default, so the entry can
    go. That assumption is false whenever a control's recorded default is NOT
    its first option, and img2img has exactly such a control:

        img2img/Inpaint area/value = 'Only masked'      <- default
        first option, and so the capture label            = 'Whole picture'

    Choosing "Whole picture" is therefore a REAL change that this rule deleted,
    and restore then left the page on 'Only masked'. That is a silently lost
    setting, which is the one failure this whole subsystem exists to prevent --
    and it cost a user their inpaint area after a restore. The entries it used
    to remove were harmless noise: restoring a radio to the option it is already
    on is a no-op. Noise is recoverable; a lost setting is not.
    """
    return False


def _unused_is_first_choice(label, val):
    """An unlabeled radio still sitting on its first option.

    Gradio 6 gives a radio GROUP no title element of its own, so the capture
    falls back to the text of the first option: ControlNet's control mode is
    captured as "Balanced" = "Balanced", img2img's resize mode as "Just resize"
    = "Just resize", the top bar's "Queue" = "Queue". Label equal to value means
    the control is on the option its label was taken from -- the first, and the
    build default for every one of these that Forge creates. Some 45 entries per
    session, none of which restore anything.

    When the value has moved off that first option ("fill" = "original") we keep
    it: without a ui-config record there is no way to tell a user's choice from
    a build default, and the cost of being wrong is a lost setting.
    """
    return bool(label) and _norm_label(label) == str(val).strip()


def _is_untouched_value(val):
    """True when a value is self-evidently the un-set state of its control.

    Blank text, an unchecked box, and a numeric zero. The first two cannot be
    anything but unset. Zero is a judgement call, taken deliberately: this path
    runs ONLY for controls ui-config has no record of, and in practice those are
    the integrated extras -- ControlNet's Start Timestep, LatentModifier's
    Rescale Cfg Phi, DynamicThresholding's Cfg Scale Min, Auto SAM's
    min_mask_region_area -- where zero IS the off position. Anything ui-config
    does record is compared against its real default above and never reaches
    here, so the only exposure is a control with an unrecorded non-zero default
    that the user deliberately set to 0; it would come back as its own default.
    """
    if val is None:
        return True
    s = str(val).strip()
    if s == "" or s.lower() == "false":
        return True
    try:
        return float(s) == 0.0
    except (TypeError, ValueError):
        return False


def _prune_default_values(snapshots):
    """Drop entries the user has not changed.

    A session used to carry a copy of the whole page -- 400+ entries, nearly all
    defaults -- which made restore a bulk rewrite and dragged lazily-built tabs
    open for values nobody had touched. ui-config.json is the authoritative
    answer to "what does a fresh page show" (it is what create_ui applies at
    build time), and consulting it HERE, at save time, avoids what defeated two
    browser-side attempts: gradio builds tabs lazily, so a baseline taken in the
    page races the mount and can record the user's own edit as the default.

    Conservative by construction: an entry goes only when its default is known
    and equal, or when it is self-evidently unset (blank text / unchecked box).
    Everything else is kept, so the worst case is a file that is still too big
    -- never a setting that silently disappears.
    """
    exact, labels, steps, plain, plain_steps = _ui_config_index()
    if not exact:
        return snapshots

    out = {}
    for tab, snap in (snapshots or {}).items():
        if not isinstance(snap, dict) or tab.startswith("__"):
            # quicksettings has no ui-config counterpart -- keep it whole, minus
            # the first-choice radios described below ("Queue" = "Queue").
            out[tab] = ({k: v for k, v in snap.items() if not _is_first_choice(k, v)}
                        if isinstance(snap, dict) else snap)
            continue
        tab_token = _UI_CONFIG_TAB_ALIAS.get(tab.lower(), tab.lower())
        kept = {}
        for label, val in snap.items():
            if label.startswith("__"):
                kept[label] = val
                continue
            if _is_unidentifiable(label):
                continue
            # An empty value is not a choice, it is a control that has not been
            # populated: a dropdown whose options load asynchronously reads as
            # "" until they arrive, which is how "ControlNet Unit 0 > Model",
            # "SAM Model", "Script" and eight more got into the session for a
            # tab nobody had configured. Restoring "" into a dropdown does
            # nothing useful, and a control genuinely sitting at an empty
            # default would be dropped by the comparison below anyway.
            if str(val).strip() == "":
                continue
            # a section prefix ("Replacer > ...") is often a tab of its own
            tabs = [tab_token]
            if " > " in label:
                tabs.append(label.split(" > ")[0].strip().lower())
            hit = _match_key(exact, labels, tabs, label)
            known = exact.get(hit) if hit is not None else None
            step_set = steps.get(hit)
            # A BARE label means the tab's own control, so the plain
            # "<tab>/<label>" key wins outright. Falling back to the union of
            # every script that reuses the name is what kept Width (1024 vs
            # Ultimate SD Upscale's tile 64), Sampling steps, Sampling method,
            # Mask blur and Schedule type in an untouched img2img tab: the
            # "all recorded defaults must agree" rule can never be satisfied
            # when two different controls share a short name.
            if hit is not None and " > " not in label and hit in plain:
                known = plain[hit]
                step_set = plain_steps.get(hit, step_set)
            if known:
                if _is_default_value(val, known, step_set):
                    continue         # recorded default, and we are sitting on it
                if _shapes_agree(val, known):
                    kept[label] = val
                    continue         # genuinely different from its default
                # otherwise the label matched a different control entirely —
                # fall through and judge the value on its own
            if _is_first_choice(label, val):
                continue             # unlabeled radio, still on its first option
            if _is_untouched_value(val):
                continue             # no record, but unmistakably unset
            kept[label] = val
        # A tab whose only survivors are bookkeeping ("__subtab") holds nothing
        # the user changed. Keeping it made restore VISIT that tab -- which in
        # gradio 6 means BUILDING it -- so a restore opened Extras, PNG Info and
        # every extension tab that had merely been glanced at once.
        if any(not k.startswith("__") for k in kept):
            out[tab] = kept
    return out


def _session_save(state):
    # Keep one generation of backup: a save from a fresh/near-default page must
    # never be able to DESTROY the previous real session. Only rotate when the
    # incoming state is not obviously poorer than what it replaces (fewer tabs
    # or dramatically fewer captured controls == suspicious downgrade; keep the
    # old file as .prev either way so recovery is always possible).
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                old = json.load(f)
            old_tabs = old.get("uiSnapshots") or {}
            new_tabs = state.get("uiSnapshots") or {}
            old_n = sum(len(v) for v in old_tabs.values())
            new_n = sum(len(v) for v in new_tabs.values())
            if len(old_tabs) > len(new_tabs) or new_n < old_n // 2:
                # downgrade-looking save: preserve the richer state as .prev
                with open(SESSION_FILE + ".prev", "w", encoding="utf-8") as f:
                    json.dump(old, f)
            # MERGE per tab rather than replace. The page can only ever capture
            # the tab that is mounted, so a save legitimately carries a SUBSET
            # of what the user has configured; writing it verbatim quietly threw
            # away every other tab. Incoming values win; anything the payload
            # does not mention is carried forward. (Stale labels are harmless --
            # restore matches by label and skips what it cannot find.)
            merged = dict(old_tabs)
            for tab, snap in new_tabs.items():
                merged[tab] = {**(old_tabs.get(tab) or {}), **(snap or {})}
            state = dict(state)
            state["uiSnapshots"] = merged
            # The edited-tab list accumulates the same way, and for the same
            # reason: one save carries only what this page knows about.
            state["uiEditedTabs"] = sorted(set(old.get("uiEditedTabs") or [])
                                           | set(state.get("uiEditedTabs") or []))
    except Exception:
        pass
    try:
        if state.get("uiSnapshots"):
            state = dict(state)
            snaps = _prune_default_values(state["uiSnapshots"])
            # Second gate, independent of the value comparison: a tab nobody
            # ever typed in does not belong in the session. Value pruning alone
            # cannot clear a tab like Checkpoint Merger, whose model dropdowns
            # legitimately read back the loaded checkpoint and whose defaults
            # ui-config never recorded -- so merely OPENING that tab once left
            # it in the session, and every restore afterwards re-opened (and in
            # gradio 6, rebuilt) it. An edit is a trusted input event inside the
            # tab, or the assistant applying a value there; the list persists in
            # the file, so it survives restarts and multi-client saves. The
            # active tab is always kept: it is where the user is.
            # ABSENT means "this client is too old to know" -> do not filter.
            # EMPTY means "nothing has been edited yet" -> filter everything but
            # the active tab. Conflating the two made the reset button useless: it
            # left the list empty, which switched the gate off, so the very next
            # capture wrote back the mounted-but-untouched img2img tab it had
            # just removed.
            edited_raw = state.get("uiEditedTabs")
            if edited_raw is not None:
                edited = set(edited_raw)
                keep_too = {state.get("uiActiveTab")}
                snaps = {t: v for t, v in snaps.items()
                         if t.startswith("__") or t in edited or t in keep_too}
                state["uiEditedTabs"] = sorted(e for e in edited if e in snaps)
            state["uiSnapshots"] = snaps
    except Exception:
        pass                       # pruning is an optimisation, never a gate
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
                # Already hibernated (idle since boot or a previous run) — Forge
                # already owns the VRAM. Arming the restore worker here would
                # wake the LLM after EVERY generation, evicting the Forge model
                # and making every back-to-back run pay a full weight re-upload.
                # Leave it parked; the first chat message auto-wakes it.
                if _llm_sleeping():
                    return
                print("[forge-ai] generation starting — soft-unloading LLM to free VRAM")
                result = _stop_textgen("soft")
                if not result.get("soft"):
                    _wait_textgen_gone(8.0)
                elif not _wait_llm_parked(30.0):
                    # Still holding VRAM after 30s: it is mid-completion and
                    # will not park. Generating alongside it means paging at
                    # best and an evicted-weights crash at worst, so take the
                    # memory back the blunt way -- the KV cache is worth less
                    # than the run the user is waiting on.
                    print("[forge-ai] LLM did not hibernate (busy generating) — killing it to free VRAM")
                    _stop_textgen("kill")
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
            _kill_process_or_pid(pid)
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

    @app.post("/forge-ai/session/clear")
    def session_clear():
        """Throw the saved session away.

        Editing the file by hand does not work while a page is open, and that
        is not a quirk -- it is the design. The client seeds uiSnapshots once at
        boot and never re-syncs, and _session_save MERGES per tab, so a stale
        page puts back anything removed on the next autosave. Reloading does not
        help either: the beforeunload beacon writes the stale copy out as the
        page goes away, before the fresh one can seed. The only reset that holds
        is one that clears BOTH sides, which is what the button calling this
        does -- the client empties its own copy before asking.
        """
        try:
            for p in (SESSION_FILE, SESSION_FILE + ".prev"):
                if os.path.exists(p):
                    os.remove(p)
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
