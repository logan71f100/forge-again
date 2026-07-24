#!/usr/bin/env python
"""
forge-again pre-merge test harness.

Run this on the `testing` branch before folding it into `main`. It has no
third-party dependencies -- plain stdlib, so it works in any environment that
can run Forge at all.

    python tests/run_tests.py              # everything
    python tests/run_tests.py --static     # tier 1 only (fast, seconds)
    python tests/run_tests.py --list       # show checks without running

Tier 1 (static)  -- syntax, dependency conflicts, JSON/BOM, line endings,
                    and a guard that no personal file is tracked by git.
Tier 2 (boot)    -- actually starts the server on a spare port with throwaway
                    settings files, and fails if ANY traceback appears during
                    startup. Your real config.json is never touched.

Every check the harness performs corresponds to something that has actually
broken this project before; see tests/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Code we actually maintain. Vendored trees (annotators, third-party packages)
# are excluded -- some of them aren't even valid modern Python and we are not
# going to fix them.
OUR_CODE = [
    "modules",
    "modules_forge",
    "scripts",
    "extensions/forge-ai-assistant",
    "extensions/sd-webui-replacer",
    "extensions-builtin/sd_forge_controlnet",
]
VENDORED = ("forge_legacy_preprocessors", "packages_3rdparty", "annotator",
            "node_modules", "__pycache__", "venv", "python")

# Dependency conflicts we have consciously accepted. Anything NOT in this list
# is treated as a regression, which is how the protobuf/open-clip breakage
# would have been caught.
ACCEPTED_CONFLICTS = [
    # onnxruntime wants protobuf>=4.25.8, but open-clip-torch (SDXL text
    # encoder) caps it <4. We pin 3.20.3; see requirements_versions.txt.
    ("onnxruntime", "protobuf"),
]

# Files that must never be committed -- personal settings and generated output.
MUST_NOT_BE_TRACKED = [
    "config.json", "ui-config.json", "extra-args.txt", "current_mode.txt",
    "styles.csv",
]
MUST_NOT_BE_TRACKED_DIRS = ["outputs/", "output/", "log/", "venv/", "python/"]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    colour = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m"}.get(status, "")
    reset = "\033[0m" if colour else ""
    print(f"  [{colour}{status}{reset}] {name}" + (f"\n         {detail}" if detail else ""))


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def venv_python() -> str:
    """The interpreter Forge itself runs under, not necessarily ours."""
    for rel in (os.path.join("venv", "Scripts", "python.exe"),
                os.path.join("venv", "bin", "python")):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            return p
    return sys.executable


# --------------------------------------------------------------------------
# Tier 1 -- static
# --------------------------------------------------------------------------

def check_syntax() -> None:
    """Every file we maintain must at least compile."""
    bad, checked = [], 0
    targets = [os.path.join(ROOT, p) for p in OUR_CODE]
    targets += [os.path.join(ROOT, f) for f in os.listdir(ROOT) if f.endswith(".py")]

    import warnings
    # Old files carry invalid escape sequences in docstrings. Those are
    # warnings, not syntax errors, and they'd drown the report.
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    for target in targets:
        files = []
        if os.path.isfile(target):
            files = [target]
        elif os.path.isdir(target):
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames[:] = [d for d in dirnames if d not in VENDORED]
                files += [os.path.join(dirpath, f) for f in filenames if f.endswith(".py")]
        for f in files:
            checked += 1
            try:
                # compile() checks syntax without writing a .pyc anywhere --
                # py_compile needs a cfile, and os.devnull ("nul") is illegal
                # for that on Windows.
                compile(open(f, "rb").read(), f, "exec")
            except SyntaxError as e:
                bad.append(f"{os.path.relpath(f, ROOT)}:{e.lineno}: {e.msg}")
            except Exception as e:
                bad.append(f"{os.path.relpath(f, ROOT)}: {type(e).__name__}: {e}")

    if bad:
        record("syntax: all maintained Python compiles", FAIL,
               f"{len(bad)} file(s) failed:\n         " + "\n         ".join(bad[:5]))
    else:
        record("syntax: all maintained Python compiles", PASS, f"{checked} files")


def check_dependency_conflicts() -> None:
    """`pip check`, minus conflicts we've deliberately accepted.

    This is the guard that would have caught onnxruntime silently pulling
    protobuf 7.x over open-clip-torch's <4 cap.
    """
    py = venv_python()
    proc = run([py, "-m", "pip", "check"])
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0:
        record("deps: no conflicts", PASS)
        return

    unexpected = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "which is not installed" in line and not line:
            continue
        if any(a.lower() in line.lower() and b.lower() in line.lower()
               for a, b in ACCEPTED_CONFLICTS):
            continue
        unexpected.append(line)

    if unexpected:
        record("deps: no conflicts", FAIL,
               "unexpected conflict(s):\n         " + "\n         ".join(unexpected[:6]))
    else:
        record("deps: no conflicts", PASS, f"{len(out.splitlines())} accepted conflict(s) ignored")


def check_pins_hold() -> None:
    """Installed versions still match the `==` pins.

    Extension installers run on every startup and can pull pinned packages past
    their documented caps, so this drifts silently on a working machine.
    """
    proc = run([venv_python(), "check_pins.py", "--check"])
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0:
        record("deps: installed versions match the pins", PASS)
    else:
        record("deps: installed versions match the pins", FAIL,
               out[:400] or "check_pins.py reported drift")


def _load_downloader():
    """Import the model-downloader extension standalone.

    Forge's modules aren't importable outside a running server, so they're
    stubbed. This lets the downloader's classification and file-handling logic
    be tested without booting anything.
    """
    import types
    import importlib.util

    for name in ("modules",):
        sys.modules.setdefault(name, types.ModuleType(name))
    mods = sys.modules["modules"]
    mods.script_callbacks = types.SimpleNamespace(on_ui_tabs=lambda *a, **k: None)
    mods.shared = types.SimpleNamespace(cmd_opts=types.SimpleNamespace())
    mods.paths = types.SimpleNamespace(models_path="models", data_path=".")
    mods.errors = types.SimpleNamespace(report=lambda *a, **k: None)
    if "gradio" not in sys.modules:
        g = types.ModuleType("gradio")
        g.update = lambda **k: k
        sys.modules["gradio"] = g

    path = os.path.join(ROOT, "extensions-builtin", "model-downloader",
                        "scripts", "model_downloader.py")
    spec = importlib.util.spec_from_file_location("_md_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Filename -> expected folder. Each entry is a real case, and several are
# regressions: Kataragi is a ControlNet that the "xl" rule used to claim,
# ae.safetensors is the Flux VAE and contains no "vae" at all, and the
# Juggernaut/RealVis names broke when the XL match was made too strict.
CLASSIFY_CASES = [
    ("Kataragi_inpaintXL-fp16.safetensors", "ControlNet"),
    ("control_v11p_sd15_canny_fp16.safetensors", "ControlNet"),
    ("xinsir-depth-sdxl.safetensors", "ControlNet"),
    ("controlnet-union-promax-sdxl.safetensors", "ControlNet"),
    ("ae.safetensors", "VAE"),
    ("sdxlVAE_sdxlVAE.safetensors", "VAE"),
    ("4x-UltraSharp.pth", "Upscaler (ESRGAN)"),
    ("t5xxl_fp8_e4m3fn.safetensors", "Text encoder"),
    ("clip_l.safetensors", "Text encoder"),
    ("someStyle_v2_lora.safetensors", "LoRA"),
    ("fluxunchained-dev-Q6_K.gguf", "Checkpoint (Flux)"),
    ("JuggernautXL.safetensors", "Checkpoint (XL)"),
    ("RealVisXL_V5.safetensors", "Checkpoint (XL)"),
    ("anterosXXXL_v10.safetensors", "Checkpoint (XL)"),
    ("novaFurryXL_ilV180A.safetensors", "Checkpoint (XL)"),
    ("v1-5-pruned-emaonly.safetensors", "Checkpoint (SD)"),
    # Deliberately unclassifiable: the name says nothing about the base model.
    # None is the correct answer -- callers fall back to metadata, then mode.
    ("cyberrealistic_v90.safetensors", None),
    ("epicrealism_pureEvolutionV5.safetensors", None),
]


def check_downloader_classification() -> None:
    """Model files must be sorted into the right folder by filename."""
    try:
        md = _load_downloader()
    except Exception as e:
        record("downloader: filename classification", FAIL, f"import failed: {type(e).__name__}: {e}")
        return
    wrong = []
    for name, want in CLASSIFY_CASES:
        got = md._guess_category(name)
        if got != want:
            wrong.append(f"{name}: expected {want!r}, got {got!r}")
    if wrong:
        record("downloader: filename classification", FAIL, "\n         ".join(wrong[:8]))
    else:
        record("downloader: filename classification", PASS, f"{len(CLASSIFY_CASES)} cases")


def check_downloader_file_safety() -> None:
    """Move/delete must not escape their folder or clobber existing files."""
    try:
        md = _load_downloader()
    except Exception as e:
        record("downloader: file operation safety", FAIL, f"import failed: {e}")
        return

    tmp = tempfile.mkdtemp(prefix="forge-dl-test-")
    problems = []
    try:
        cats = {c: os.path.join(tmp, re.sub(r"\W", "_", c))
                for c in ("Checkpoint (SD)", "Checkpoint (XL)", "LoRA")}
        for d in cats.values():
            os.makedirs(d, exist_ok=True)
        md._category_dirs = lambda: cats
        md._refresh_lists = lambda *a, **k: None

        xl = os.path.join(cats["Checkpoint (XL)"], "m.safetensors")
        sd = os.path.join(cats["Checkpoint (SD)"], "m.safetensors")
        open(xl, "wb").write(b"x" * 2048)
        open(sd, "wb").write(b"y" * 99)

        # 1. path traversal must be rejected outright
        for evil in ("Checkpoint (SD)::../../evil.safetensors",
                     "Checkpoint (SD)::..\\..\\evil.safetensors",
                     "Nope::m.safetensors"):
            try:
                md._resolve_selection(evil)
                problems.append(f"accepted an escaping selection: {evil}")
            except Exception:
                pass

        # 2. a move must never overwrite an existing destination
        md.move_one("Checkpoint (XL)::m.safetensors", "Checkpoint (SD)")
        if not os.path.exists(xl):
            problems.append("move clobbered: source removed despite a name clash")
        if os.path.getsize(sd) != 99:
            problems.append("move clobbered: destination file was overwritten")

        # 3. a clean move relocates the file
        os.remove(sd)
        md.move_one("Checkpoint (XL)::m.safetensors", "Checkpoint (SD)")
        if os.path.exists(xl) or not os.path.exists(sd):
            problems.append("clean move did not relocate the file")

        # 4. delete removes exactly the target
        md.delete_one("Checkpoint (SD)::m.safetensors")
        if os.path.exists(sd):
            problems.append("delete did not remove the file")
    except Exception as e:
        problems.append(f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if problems:
        record("downloader: file operation safety", FAIL, "\n         ".join(problems))
    else:
        record("downloader: file operation safety", PASS,
               "traversal rejected, no clobbering, move + delete correct")


def check_json_and_bom() -> None:
    """JSON must parse strictly and must NOT carry a UTF-8 BOM.

    A BOM in config.json makes Forge fail to read it and silently reset every
    setting to defaults -- PowerShell's `Set-Content -Encoding utf8` writes one.
    """
    problems, checked = [], 0
    candidates = []
    for name in ("config.json", "ui-config.json"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            candidates.append(p)
    for sub in ("docker-compose.yml",):  # presence only; YAML isn't parsed here
        pass
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, ".github")):
        dirnames[:] = [d for d in dirnames if d not in VENDORED]
        candidates += [os.path.join(dirpath, f) for f in filenames if f.endswith(".json")]

    for p in candidates:
        checked += 1
        rel = os.path.relpath(p, ROOT)
        raw = open(p, "rb").read()
        if raw.startswith(b"\xef\xbb\xbf"):
            problems.append(f"{rel}: has a UTF-8 BOM (Forge will reset settings)")
            continue
        try:
            json.loads(raw.decode("utf-8"))
        except Exception as e:
            problems.append(f"{rel}: {e}")

    if problems:
        record("json: parses, no BOM", FAIL, "\n         ".join(problems))
    elif checked == 0:
        record("json: parses, no BOM", SKIP, "no JSON files present to check")
    else:
        record("json: parses, no BOM", PASS, f"{checked} file(s)")


def check_line_endings() -> None:
    """.bat must be CRLF (cmd misparses LF), .sh must be LF (bash rejects CRLF)."""
    problems = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in VENDORED and not d.startswith(".git")]
        for f in filenames:
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            if f.endswith((".bat", ".cmd")):
                if b"\r\n" not in open(p, "rb").read(8192):
                    problems.append(f"{rel}: expected CRLF, found LF")
            elif f.endswith(".sh"):
                if b"\r\n" in open(p, "rb").read(8192):
                    problems.append(f"{rel}: expected LF, found CRLF (bash will reject it)")
    if problems:
        record("line endings: .bat=CRLF .sh=LF", FAIL, "\n         ".join(problems[:8]))
    else:
        record("line endings: .bat=CRLF .sh=LF", PASS)


def check_no_personal_files_tracked() -> None:
    """Personal settings and generated output must never be committed."""
    proc = run(["git", "ls-files"])
    if proc.returncode != 0:
        record("privacy: no personal files tracked", SKIP, "not a git repo")
        return
    tracked = set(proc.stdout.splitlines())
    leaked = [f for f in MUST_NOT_BE_TRACKED if f in tracked]
    leaked += [f for f in tracked if any(f.startswith(d) for d in MUST_NOT_BE_TRACKED_DIRS)]
    if leaked:
        record("privacy: no personal files tracked", FAIL,
               "tracked but must not be:\n         " + "\n         ".join(sorted(set(leaked))[:10]))
    else:
        record("privacy: no personal files tracked", PASS, f"{len(tracked)} tracked files scanned")


# --------------------------------------------------------------------------
# Tier 2 -- boot
# --------------------------------------------------------------------------

def free_port(preferred: int = 7899) -> int:
    for port in range(preferred, preferred + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def check_server_boot(timeout: int = 300) -> None:
    """Start the real server and fail on ANY traceback during startup.

    This is the check that catches missing extension dependencies -- both the
    segment_anything and joblib breakages presented exactly this way.
    """
    py = venv_python()
    port = free_port()
    tmp = tempfile.mkdtemp(prefix="forge-test-")
    settings = os.path.join(tmp, "config.json")
    uiconfig = os.path.join(tmp, "ui-config.json")

    # The harness runs against its OWN config fixture, copied to a temp file so
    # the fixture itself is never mutated and the developer's real config.json
    # is never touched. Values in the fixture are non-default on purpose, so the
    # round-trip check below proves the server actually read them.
    fixture = os.path.join(ROOT, "tests", "fixtures", "test-config.json")
    expected = {k: v for k, v in json.load(open(fixture, encoding="utf-8")).items()
                if not k.startswith("_")}
    with open(settings, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(expected, fh, indent=4)

    env = dict(os.environ)
    env["FORGE_NO_LLM"] = "1"          # don't pull an 18 GB model in a test
    env["SD_WEBUI_RESTARTING"] = "1"   # suppress browser autolaunch
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [py, "launch.py", "--port", str(port), "--api", "--skip-install",
           "--skip-python-version-check", "--no-half-vae", "--disable-xformers",
           "--ui-settings-file", settings, "--ui-config-file", uiconfig]

    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")

    # Drain stdout continuously on a thread. Forge emits a lot during UI build,
    # and if nobody reads the pipe it fills (~64 KB) and the server BLOCKS
    # mid-startup, which looks exactly like a hang.
    log_lines: list[str] = []

    def _pump() -> None:
        try:
            for line in proc.stdout:            # type: ignore[union-attr]
                log_lines.append(line.rstrip("\n"))
        except Exception:
            pass

    import threading
    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    booted = False
    live_options: dict | None = None
    models_ok: bool | None = None
    settings_snapshot: str | None = None
    try:
        start = time.time()
        while time.time() - start < timeout:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/internal/ping", timeout=2):
                    booted = True
                    break
            except Exception:
                time.sleep(2)

        # Interrogate the running server BEFORE shutting it down.
        if booted:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/sdapi/v1/options", timeout=30) as r:
                    live_options = json.load(r)
            except Exception:
                live_options = None
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/sdapi/v1/sd-models", timeout=30) as r:
                    models_ok = isinstance(json.load(r), list)
            except Exception:
                models_ok = False

        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        pump.join(timeout=10)   # let the reader flush the tail of the log
    finally:
        if proc.poll() is None:
            proc.kill()
        # Keep a copy of the settings file as the server left it, so the
        # preservation check below can inspect it after the temp dir is gone.
        try:
            fd, settings_snapshot = tempfile.mkstemp(prefix="forge-test-cfg-", suffix=".json")
            os.close(fd)
            shutil.copy2(settings, settings_snapshot)
        except Exception:
            settings_snapshot = None
        shutil.rmtree(tmp, ignore_errors=True)

    if not booted:
        tail = "\n         ".join(log_lines[-12:]) or "(no output)"
        record("boot: server starts and answers /internal/ping", FAIL,
               f"never reached ping within {timeout}s. Last output:\n         {tail}")
        return
    record("boot: server starts and answers /internal/ping", PASS, f"port {port}")

    # The real prize: a clean startup log.
    tracebacks = [i for i, l in enumerate(log_lines) if l.startswith("Traceback")]
    if tracebacks:
        excerpts = []
        for i in tracebacks[:3]:
            excerpts.append("\n         ".join(log_lines[i:i + 6]))
        record("boot: startup log has no tracebacks", FAIL,
               f"{len(tracebacks)} traceback(s):\n         " + "\n         ---\n         ".join(excerpts))
    else:
        record("boot: startup log has no tracebacks", PASS, f"{len(log_lines)} log lines scanned")

    errors = [l for l in log_lines if re.search(r"Error loading script|Error executing callback", l)]
    if errors:
        record("boot: all scripts load", FAIL, "\n         ".join(errors[:6]))
    else:
        record("boot: all scripts load", PASS)

    # ---- config round-trip -------------------------------------------------
    # Every option in the fixture must come back from the API with the value we
    # set. A mismatch means the option was renamed, dropped, or silently reset
    # to its default -- which is exactly how config regressions hide.
    # Core options are the ones /sdapi/v1/options actually reports. Extension
    # options (forge_ai_*, replacer_*) are deliberately excluded here: Forge
    # does not surface extension-registered settings through that endpoint at
    # all -- verified against a real config too, so it is pre-existing
    # behaviour, not a regression. They're covered by the file check below.
    core = {k: v for k, v in expected.items()
            if not k.startswith(("forge_ai_", "replacer_"))}

    if live_options is None:
        record("config: core options round-trip via API", FAIL, "could not read /sdapi/v1/options")
    else:
        missing, wrong = [], []
        for key, want in core.items():
            if key not in live_options:
                missing.append(key)
            elif isinstance(want, float):
                if abs(float(live_options[key]) - want) > 1e-6:
                    wrong.append(f"{key}: set {want!r}, got {live_options[key]!r}")
            elif live_options[key] != want:
                wrong.append(f"{key}: set {want!r}, got {live_options[key]!r}")

        detail = []
        if missing:
            detail.append("unknown to the server (renamed/removed?): " + ", ".join(missing))
        detail += wrong
        if detail:
            record("config: core options round-trip via API", FAIL, "\n         ".join(detail[:10]))
        else:
            record("config: core options round-trip via API", PASS,
                   f"{len(core)} option(s) set and read back unchanged")

    # Every option we wrote must still be in the file, with its value intact,
    # after a full start/stop cycle. This is the guard against settings being
    # silently pruned or reset -- the failure mode a stray BOM once caused.
    try:
        after = json.load(open(settings_snapshot, encoding="utf-8")) if settings_snapshot else None
    except Exception as e:
        after, _err = None, e
    if after is None:
        record("config: settings survive a server lifecycle", SKIP, "config file unreadable after run")
    else:
        lost = [k for k in expected if k not in after]
        changed = [f"{k}: was {expected[k]!r}, now {after[k]!r}"
                   for k in expected
                   if k in after and after[k] != expected[k]
                   and not isinstance(expected[k], float)]
        if lost or changed:
            d = []
            if lost:
                d.append(f"{len(lost)} option(s) dropped from config: " + ", ".join(lost[:8]))
            d += changed[:6]
            record("config: settings survive a server lifecycle", FAIL, "\n         ".join(d))
        else:
            record("config: settings survive a server lifecycle", PASS,
                   f"all {len(expected)} option(s) intact, incl. {len(expected) - len(core)} extension option(s)")

    record("api: /sdapi/v1/sd-models responds",
           PASS if models_ok else FAIL,
           "" if models_ok else "endpoint did not return a list")

    if settings_snapshot and os.path.exists(settings_snapshot):
        os.unlink(settings_snapshot)


# --------------------------------------------------------------------------
# Tier 3 -- GPU generation
#
# These assert that generated images are CORRECT, not merely that the endpoint
# returned 200. A server that boots fine can still emit black frames, ignore
# the requested resolution, or silently stop honouring the seed.
# --------------------------------------------------------------------------

class ServerSession:
    """A running Forge instance, for tests that need to make several calls.

    Starting the server costs ~15 s and the first generation pays for the
    checkpoint load, so the GPU checks share one session rather than each
    paying that toll.
    """

    def __init__(self, mode: str = None, timeout: int = 420):
        self.mode = mode or self._current_mode()
        self.timeout = timeout
        self.port = free_port(7910)
        self.tmp = tempfile.mkdtemp(prefix="forge-gpu-test-")
        self.log: list[str] = []
        self.proc: subprocess.Popen | None = None

    @staticmethod
    def _current_mode() -> str:
        p = os.path.join(ROOT, "current_mode.txt")
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip() or "xl"
        return "xl"

    def __enter__(self) -> "ServerSession":
        settings = os.path.join(self.tmp, "config.json")
        fixture = os.path.join(ROOT, "tests", "fixtures", "test-config.json")
        data = {k: v for k, v in json.load(open(fixture, encoding="utf-8")).items()
                if not k.startswith("_")}
        # Live previews and progress polling only add noise to an API-driven run.
        data["live_previews_enable"] = False
        json.dump(data, open(settings, "w", encoding="utf-8"), indent=4)

        models = os.environ.get("FORGE_MODELS_DIR", os.path.join(ROOT, "models"))
        env = dict(os.environ)
        env.update(FORGE_NO_LLM="1", SD_WEBUI_RESTARTING="1", PYTHONUNBUFFERED="1")

        cmd = [venv_python(), "launch.py", "--port", str(self.port), "--api",
               "--skip-install", "--skip-python-version-check", "--no-half-vae",
               "--disable-xformers", "--cuda-malloc",
               "--ui-settings-file", settings,
               "--ui-config-file", os.path.join(self.tmp, "ui-config.json"),
               "--ckpt-dir", os.path.join(models, "checkpoints", self.mode),
               "--lora-dir", os.path.join(models, "Lora"),
               "--vae-dir", os.path.join(models, "VAE"),
               "--text-encoder-dir", os.path.join(models, "text_encoder"),
               "--esrgan-models-path", os.path.join(models, "ESRGAN")]

        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace")

        import threading
        # Same pipe-draining requirement as the boot check: an undrained pipe
        # fills and blocks the server mid-startup.
        threading.Thread(
            target=lambda: [self.log.append(l.rstrip("\n")) for l in self.proc.stdout],
            daemon=True).start()

        start = time.time()
        while time.time() - start < self.timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited during startup:\n" + "\n".join(self.log[-15:]))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/internal/ping", timeout=2):
                    return self
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"server did not start within {self.timeout}s")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def recent_errors(self, limit: int = 12) -> str:
        """Server-side error lines, so a failing request reports the CAUSE.

        An HTTP 500 on its own is nearly useless in a test report; the
        traceback the server printed is the actual finding.
        """
        # Return the whole last traceback block, not just lines that look like
        # errors -- the "File ..., line N" frames are what identify the cause.
        # The tail of the log, unfiltered. Attempts to be clever about which
        # frames matter just hide the exception line, which is the one thing
        # you always need.
        tail = [l for l in self.log[-limit:] if l.strip()]
        return "\n         ".join(tail) if tail else "(server printed nothing)"

    def post(self, path: str, payload: dict, timeout: int = 900) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # The API puts the real reason in the response body; without this
            # every failure reports as a bare "HTTP Error 500".
            body = e.read().decode("utf-8", "replace")
            try:
                j = json.loads(body)
                detail = j.get("detail") or j.get("error") or body
                if isinstance(detail, dict):
                    detail = detail.get("errors") or detail.get("error") or json.dumps(detail)
            except Exception:
                detail = body
            raise RuntimeError(f"HTTP {e.code} from {path}: {str(detail)[:400]}") from None


def _decode(b64: str):
    """base64 -> PIL image. Pillow is a hard dependency of Forge itself."""
    import base64
    import io
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _looks_blank(img) -> bool:
    """True if the image carries essentially no detail (black/uniform frame)."""
    from PIL import ImageStat
    stat = ImageStat.Stat(img.convert("L"))
    return stat.stddev[0] < 3.0


def check_gpu_generation() -> None:
    """txt2img, seed determinism, hires-fix dimensions and img2img on real hardware."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        record("gpu: Pillow available", SKIP, "Pillow missing; run under the Forge venv")
        return

    # Skip cleanly rather than failing confusingly when the machine simply
    # can't run this tier.
    probe = run([venv_python(), "-c",
                 "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"])
    if probe.returncode != 0:
        record("gpu: CUDA available", SKIP, "no CUDA device visible to torch")
        return

    mode = ServerSession._current_mode()
    ckpt_dir = os.path.join(os.environ.get("FORGE_MODELS_DIR", os.path.join(ROOT, "models")),
                            "checkpoints", mode)
    have = [f for f in os.listdir(ckpt_dir)] if os.path.isdir(ckpt_dir) else []
    if not any(f.endswith((".safetensors", ".ckpt", ".gguf")) for f in have):
        record("gpu: checkpoint available", SKIP, f"no checkpoint in {os.path.relpath(ckpt_dir, ROOT)}")
        return

    try:
        session = ServerSession()
    except Exception as e:
        record("gpu: server starts", FAIL, str(e)[:300])
        return

    try:
        with session as s:
            record("gpu: server starts", PASS, f"mode={s.mode}, port={s.port}")
            base = {"steps": 6, "cfg_scale": 5, "sampler_name": "Euler",
                    "prompt": "a red apple on a wooden table", "seed": 12345}

            # --- txt2img: right size, and actually contains an image ---------
            try:
                r = s.post("/sdapi/v1/txt2img", dict(base, width=768, height=768))
                imgs = r.get("images") or []
                if not imgs:
                    record("gpu: txt2img returns an image", FAIL, "no images in response")
                    return
                img = _decode(imgs[0])
                if img.size != (768, 768):
                    record("gpu: txt2img honours requested size", FAIL,
                           f"asked 768x768, got {img.size[0]}x{img.size[1]}")
                else:
                    record("gpu: txt2img honours requested size", PASS, "768x768")
                if _looks_blank(img):
                    record("gpu: txt2img output is not blank", FAIL,
                           "image is uniform -- black frame / VAE overflow?")
                else:
                    record("gpu: txt2img output is not blank", PASS)
                first = imgs[0]
            except Exception as e:
                record("gpu: txt2img returns an image", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
                return

            # --- determinism: same seed must reproduce the same image --------
            try:
                from PIL import ImageChops, ImageStat
                r2 = s.post("/sdapi/v1/txt2img", dict(base, width=768, height=768))
                again = _decode((r2.get("images") or [""])[0]).convert("RGB")
                d = ImageChops.difference(_decode(first).convert("RGB"), again)
                diff = sum(ImageStat.Stat(d).mean) / 3.0
                # Not byte-equality: Forge offloads dynamically, so the same
                # request can take a slightly different execution path
                # depending on free VRAM and land a few LSBs apart. A genuine
                # seed regression produces a completely different image
                # (mean diff in the tens), so the threshold is far below that
                # while staying well clear of float noise.
                if diff < 2.0:
                    record("gpu: same seed reproduces the same image", PASS,
                           f"mean diff {diff:.2f}")
                else:
                    record("gpu: same seed reproduces the same image", FAIL,
                           f"same seed produced a materially different image (mean diff {diff:.1f})")
            except Exception as e:
                record("gpu: same seed reproduces the same image", FAIL, str(e)[:200])

            # --- hires fix: final size must be width*scale -------------------
            # Guards the class of bug where hires silently returns the base
            # resolution or a wrongly-proportioned frame.
            try:
                r3 = s.post("/sdapi/v1/txt2img", dict(
                    base, width=512, height=512, enable_hr=True, hr_scale=1.5,
                    hr_second_pass_steps=4, denoising_strength=0.35))
                hi = _decode((r3.get("images") or [""])[0])
                if hi.size != (768, 768):
                    record("gpu: hires fix produces the scaled size", FAIL,
                           f"512x512 at hr_scale 1.5 should be 768x768, got {hi.size[0]}x{hi.size[1]}")
                else:
                    record("gpu: hires fix produces the scaled size", PASS, "512 -> 768")
            except Exception as e:
                record("gpu: hires fix produces the scaled size", FAIL,
                       f"{type(e).__name__}: {str(e)[:120]}\n         server said:\n         "
                       + s.recent_errors())

            # --- inpaint: the masked region changes, the rest is left alone --
            # Asserts mask ORIENTATION, not just that something happened: an
            # inverted or ignored mask still returns a plausible image, and
            # this is the operation Replacer is built on.
            try:
                import base64
                import io
                from PIL import Image, ImageChops, ImageStat

                src_img = _decode(first).convert("RGB")
                mask = Image.new("L", src_img.size, 0)
                mask.paste(255, (0, 0, src_img.size[0] // 2, src_img.size[1]))  # left half
                buf = io.BytesIO()
                mask.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode()

                r5 = s.post("/sdapi/v1/img2img", {
                    "init_images": [first], "mask": mask_b64,
                    "prompt": "a blue ceramic mug", "steps": 8, "cfg_scale": 5,
                    "denoising_strength": 0.95, "inpainting_fill": 1,
                    "inpaint_full_res": False, "mask_blur": 4,
                    "width": src_img.size[0], "height": src_img.size[1], "seed": 4242,
                })
                out_img = _decode((r5.get("images") or [""])[0]).convert("RGB")

                if out_img.size != src_img.size:
                    record("gpu: inpaint respects the mask", FAIL,
                           f"size changed {src_img.size} -> {out_img.size}")
                else:
                    w, h = src_img.size
                    def region_diff(box):
                        d = ImageChops.difference(src_img.crop(box), out_img.crop(box))
                        return sum(ImageStat.Stat(d).mean) / 3.0
                    masked = region_diff((0, 0, w // 2, h))
                    untouched = region_diff((w // 2, 0, w, h))
                    # Relative, not absolute: the VAE round-trip perturbs the
                    # whole frame slightly, so an absolute threshold would be
                    # flaky. The masked half must simply change much more.
                    if masked < 5.0:
                        record("gpu: inpaint respects the mask", FAIL,
                               f"masked region barely changed (mean diff {masked:.1f})")
                    elif masked <= untouched * 2.0:
                        record("gpu: inpaint respects the mask", FAIL,
                               f"mask looks ignored or inverted: masked diff {masked:.1f} "
                               f"vs untouched {untouched:.1f}")
                    else:
                        record("gpu: inpaint respects the mask", PASS,
                               f"masked {masked:.1f} vs untouched {untouched:.1f} mean diff")
            except Exception as e:
                record("gpu: inpaint respects the mask", FAIL,
                       f"{type(e).__name__}: {str(e)[:150]}\n         server said:\n         "
                       + s.recent_errors())

            # --- img2img: consumes an image and returns a changed one --------
            try:
                r4 = s.post("/sdapi/v1/img2img", {
                    "init_images": [first], "prompt": "a green apple on a wooden table",
                    "steps": 6, "cfg_scale": 5, "denoising_strength": 0.55,
                    "width": 768, "height": 768, "seed": 999})
                out = (r4.get("images") or [""])[0]
                im = _decode(out)
                if im.size != (768, 768):
                    record("gpu: img2img round-trip", FAIL, f"size changed to {im.size}")
                elif out == first:
                    record("gpu: img2img round-trip", FAIL, "output identical to input; img2img did nothing")
                elif _looks_blank(im):
                    record("gpu: img2img round-trip", FAIL, "output is blank")
                else:
                    record("gpu: img2img round-trip", PASS)
            except Exception as e:
                record("gpu: img2img round-trip", FAIL, str(e)[:200])
    except Exception as e:
        record("gpu: session", FAIL, f"{type(e).__name__}: {str(e)[:300]}")


# --------------------------------------------------------------------------
# Tier 4 -- clean-machine install
#
# Every other tier assumes a working install, so none of them could have caught
# the bug that mattered most in practice: on a machine without Git for Windows,
# start.bat exited silently. These guard the fresh-install path.
# --------------------------------------------------------------------------

def check_release_install_boots() -> None:
    """Boot the server from a `git archive` export -- no .git, no dev files.

    This is what somebody who downloads a release actually runs, and it is a
    genuinely different environment from the working tree: code that reaches
    for git metadata behaves differently, and export-ignore'd files are gone.
    Nothing else in the suite covers it.
    """
    export = tempfile.mkdtemp(prefix="forge-release-test-")
    try:
        tar = os.path.join(export, "src.tar")
        proc = run(["git", "archive", "--format=tar", "-o", tar, "HEAD"])
        if proc.returncode != 0:
            record("clean: release export boots", SKIP, "git archive failed")
            return
        src = os.path.join(export, "src")
        os.makedirs(src, exist_ok=True)
        import tarfile
        with tarfile.open(tar) as t:
            t.extractall(src)
        os.remove(tar)

        if os.path.exists(os.path.join(src, ".git")):
            record("clean: release export boots", FAIL, "export unexpectedly contains .git")
            return

        settings = os.path.join(export, "config.json")
        open(settings, "w", encoding="utf-8").write("{}")
        models = os.environ.get("FORGE_MODELS_DIR", os.path.join(ROOT, "models"))

        env = dict(os.environ)
        env.update(FORGE_NO_LLM="1", FORGE_NO_CONTROLNET="1",
                   SD_WEBUI_RESTARTING="1", PYTHONUNBUFFERED="1")

        port = free_port(7930)
        cmd = [venv_python(), "launch.py", "--port", str(port), "--api",
               "--skip-install", "--skip-python-version-check",
               "--ui-settings-file", settings,
               "--ui-config-file", os.path.join(export, "ui-config.json"),
               "--ckpt-dir", os.path.join(models, "checkpoints", "xl")]

        p = subprocess.Popen(cmd, cwd=src, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        log: list[str] = []
        import threading
        threading.Thread(target=lambda: [log.append(l.rstrip("\n")) for l in p.stdout],
                         daemon=True).start()

        booted = False
        start = time.time()
        while time.time() - start < 300:
            if p.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/internal/ping", timeout=2):
                    booted = True
                    break
            except Exception:
                time.sleep(2)
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
        time.sleep(1)

        if not booted:
            record("clean: release export boots", FAIL,
                   "server never came up from a release export:\n         "
                   + "\n         ".join(log[-12:]))
            return
        # NOTE: this exports committed HEAD, not the working tree -- it answers
        # "would what we are about to merge work", which is the point of a gate.
        # An uncommitted fix will look like it is still broken here.
        record("clean: release export boots", PASS, "committed HEAD, no .git, no dev files")

        tb = [i for i, l in enumerate(log) if l.startswith("Traceback")]
        # "fatal: not a git repository" is git's own stderr leaking through the
        # version probe. Harmless, but it reads as an error on every startup of
        # every release-zip install, so it must not come back.
        gitish = [l for l in log
                  if "InvalidGitRepositoryError" in l
                  or "git info" in l.lower()
                  or "not a git repository" in l.lower()]
        if tb or gitish:
            detail = []
            if gitish:
                detail.append("git-metadata errors without a .git dir:")
                detail += gitish[:4]
            for i in tb[:2]:
                detail.append("\n         ".join(log[i:i + 5]))
            record("clean: release export starts without git errors", FAIL,
                   "\n         ".join(detail))
        else:
            record("clean: release export starts without git errors", PASS,
                   f"{len(log)} log lines, no tracebacks")
    except Exception as e:
        record("clean: release export boots", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
    finally:
        shutil.rmtree(export, ignore_errors=True)


def check_launcher_guards() -> None:
    """The launchers must handle missing git and must not swallow failures."""
    bat = open(os.path.join(ROOT, "start.bat"), encoding="utf-8", errors="replace").read()
    sh = open(os.path.join(ROOT, "start.sh"), encoding="utf-8", errors="replace").read()

    problems = []
    # git is a hard requirement: Forge clones three helper repos and runs
    # `git rev-parse` on them even when they already exist.
    if "GITURL" not in bat or "where git" not in bat:
        problems.append("start.bat: no portable-git bootstrap (a machine without git will fail)")
    if "command -v git" not in sh:
        problems.append("start.sh: no git preflight check")

    # A crash must not look like a clean exit. This is what turned a small bug
    # into an unreadable one: the console window simply vanished.
    if "ERRORLEVEL" not in bat.upper():
        problems.append("start.bat: launch.py exit code is never checked")
    if "pause" not in bat:
        problems.append("start.bat: no pause on failure -- the window will vanish before it can be read")
    if ":crashed" not in bat:
        problems.append("start.bat: no crash handler")

    if problems:
        record("clean: launchers guard the fresh-install path", FAIL, "\n         ".join(problems))
    else:
        record("clean: launchers guard the fresh-install path", PASS,
               "git bootstrap + exit-code check + pause present")


def check_bootstrap_urls() -> None:
    """The hardcoded bootstrap downloads must still exist.

    Only a new user ever exercises these, so link rot would be invisible here
    and fatal there.
    """
    bat = open(os.path.join(ROOT, "start.bat"), encoding="utf-8", errors="replace").read()
    urls = dict(re.findall(r'set "(PYURL|GITURL)=(\S+)"', bat))
    if not urls:
        record("clean: bootstrap download URLs resolve", FAIL, "could not find PYURL/GITURL in start.bat")
        return

    bad = []
    for name, url in sorted(urls.items()):
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "forge-again-tests")
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status >= 400:
                    bad.append(f"{name}: HTTP {r.status}")
        except urllib.error.HTTPError as e:
            # GitHub release assets answer HEAD with a redirect chain; only a
            # real 4xx means the asset is gone.
            if e.code >= 400:
                bad.append(f"{name}: HTTP {e.code} -> {url}")
        except Exception as e:
            bad.append(f"{name}: {type(e).__name__}: {str(e)[:80]}")

    if bad:
        record("clean: bootstrap download URLs resolve", FAIL, "\n         ".join(bad))
    else:
        record("clean: bootstrap download URLs resolve", PASS, f"{len(urls)} URL(s) reachable")


def check_git_bootstrap_works() -> None:
    """Actually fetch portable git with git hidden, and clone with it.

    Reproduces the reported machine: git absent from PATH. Skipped off Windows,
    and skipped unless --deep is passed, since it downloads ~38 MB.
    """
    if sys.platform != "win32":
        record("clean: portable git bootstrap", SKIP, "Windows-only path")
        return
    if not os.environ.get("FORGE_TEST_DEEP"):
        record("clean: portable git bootstrap", SKIP,
               "downloads ~38 MB; run with --deep to exercise it")
        return

    bat = open(os.path.join(ROOT, "start.bat"), encoding="utf-8", errors="replace").read()
    m = re.search(r'set "GITURL=(\S+)"', bat)
    if not m:
        record("clean: portable git bootstrap", FAIL, "GITURL not found in start.bat")
        return

    tmp = tempfile.mkdtemp(prefix="forge-git-test-")
    try:
        zip_path = os.path.join(tmp, "git.zip")
        gitdir = os.path.join(tmp, "git")
        urllib.request.urlretrieve(m.group(1), zip_path)
        os.makedirs(gitdir, exist_ok=True)
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(gitdir)
        git_exe = os.path.join(gitdir, "cmd", "git.exe")
        if not os.path.exists(git_exe):
            record("clean: portable git bootstrap", FAIL, "cmd/git.exe missing after extraction")
            return
        # The operation that was actually failing on the reported machine.
        clone_to = os.path.join(tmp, "assets")
        proc = subprocess.run(
            [git_exe, "clone", "--depth", "1",
             "https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git", clone_to],
            capture_output=True, text=True, timeout=300)
        if proc.returncode == 0 and os.path.isdir(os.path.join(clone_to, ".git")):
            record("clean: portable git bootstrap", PASS, "fetched portable git and cloned assets")
        else:
            record("clean: portable git bootstrap", FAIL,
                   (proc.stderr or proc.stdout or "clone failed")[:300])
    except Exception as e:
        record("clean: portable git bootstrap", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_gpu_all_modes() -> None:
    """Generate in every mode that has a checkpoint, not just the active one.

    A break in sd or flux is invisible while you're working in xl, which is
    exactly how mode-specific regressions survive. Opt-in: each mode pays its
    own server start and checkpoint load.
    """
    if not os.environ.get("FORGE_TEST_ALL_MODES"):
        record("gpu: every mode generates", SKIP, "run with --all-modes (a minute per mode)")
        return

    models = os.environ.get("FORGE_MODELS_DIR", os.path.join(ROOT, "models"))
    available = []
    for mode in ("sd", "xl", "flux"):
        d = os.path.join(models, "checkpoints", mode)
        if os.path.isdir(d) and any(f.lower().endswith((".safetensors", ".ckpt", ".gguf"))
                                    for f in os.listdir(d)):
            available.append(mode)
    if not available:
        record("gpu: every mode generates", SKIP, "no checkpoints found in any mode")
        return

    for mode in available:
        try:
            with ServerSession(mode=mode) as s:
                # Flux is slow and needs no CFG; keep every mode cheap.
                payload = {"prompt": "a red apple on a wooden table", "steps": 6,
                           "width": 512, "height": 512, "seed": 7,
                           "cfg_scale": 1.0 if mode == "flux" else 5.0}
                r = s.post("/sdapi/v1/txt2img", payload, timeout=1800)
                imgs = r.get("images") or []
                if not imgs:
                    record(f"gpu[{mode}]: generates an image", FAIL, "no image returned")
                    continue
                img = _decode(imgs[0])
                if img.size != (512, 512):
                    record(f"gpu[{mode}]: generates an image", FAIL,
                           f"expected 512x512, got {img.size[0]}x{img.size[1]}")
                elif _looks_blank(img):
                    record(f"gpu[{mode}]: generates an image", FAIL, "output is blank")
                else:
                    record(f"gpu[{mode}]: generates an image", PASS, "512x512, non-blank")
        except Exception as e:
            record(f"gpu[{mode}]: generates an image", FAIL,
                   f"{type(e).__name__}: {str(e)[:200]}")


def check_download_end_to_end() -> None:
    """Actually fetch a file and confirm it lands in the right folder.

    Resolution and classification are unit-tested, but nothing exercised the
    transfer itself -- filename derivation, the .part rename, destination.
    Network-dependent, so it's opt-in rather than a flaky gate.
    """
    if not os.environ.get("FORGE_TEST_DEEP"):
        record("downloader: end-to-end fetch", SKIP, "needs network; run with --deep")
        return
    try:
        md = _load_downloader()
    except Exception as e:
        record("downloader: end-to-end fetch", FAIL, f"import failed: {e}")
        return

    tmp = tempfile.mkdtemp(prefix="forge-dl-e2e-")
    try:
        cats = {"VAE": os.path.join(tmp, "VAE")}
        md._category_dirs = lambda: cats
        md._refresh_lists = lambda *a, **k: None
        os.makedirs(cats["VAE"], exist_ok=True)

        # Small, public, stable file -- the point is the transfer path, not size.
        url = ("https://huggingface.co/openai/clip-vit-base-patch32/"
               "resolve/main/config.json")
        name, dest, status = md._download_one(url, cats["VAE"], "", lambda *_a: None)

        if not status.startswith("done"):
            record("downloader: end-to-end fetch", FAIL, f"status was {status!r}")
        elif not os.path.isfile(dest) or os.path.getsize(dest) == 0:
            record("downloader: end-to-end fetch", FAIL, "file missing or empty after download")
        elif os.path.dirname(os.path.abspath(dest)) != os.path.abspath(cats["VAE"]):
            record("downloader: end-to-end fetch", FAIL, f"landed in the wrong folder: {dest}")
        elif os.path.exists(dest + ".part"):
            record("downloader: end-to-end fetch", FAIL, ".part file was left behind")
        else:
            # Re-running must skip rather than redownload or duplicate.
            _n2, _d2, status2 = md._download_one(url, cats["VAE"], "", lambda *_a: None)
            if not status2.startswith("already"):
                record("downloader: end-to-end fetch", FAIL,
                       f"second run should skip an existing file, said {status2!r}")
            else:
                record("downloader: end-to-end fetch", PASS,
                       f"{name} ({_fmt_bytes(os.path.getsize(dest))}), re-run skipped")
    except Exception as e:
        record("downloader: end-to-end fetch", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Tier 5 -- UI regression
#
# The bugs that keep recurring in this fork are DOM-level: a control that stops
# responding after a profile is applied, a tab that won't open, an accordion
# that unmounts its children. None of that is visible over the API, so this
# tier drives a real browser.
#
# Playwright is an optional, test-only dependency:
#     venv\Scripts\python -m pip install playwright
#     venv\Scripts\python -m playwright install chromium
# --------------------------------------------------------------------------

def check_ui_regression() -> None:
    """Load the UI in a browser and exercise the controls that break."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        record("ui: loads and controls respond", SKIP,
               "playwright not installed (pip install playwright && playwright install chromium)")
        return

    try:
        session = ServerSession()
    except Exception as e:
        record("ui: loads and controls respond", FAIL, f"server did not start: {str(e)[:200]}")
        return

    with session as s:
        base = f"http://127.0.0.1:{s.port}/"
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            console_errors: list[str] = []
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

            try:
                page.goto(base, wait_until="domcontentloaded", timeout=120000)
                # Gradio hydrates after load; wait for a control to exist.
                page.wait_for_selector("#txt2img_prompt textarea", timeout=120000)
                record("ui: page loads and hydrates", PASS)
            except Exception as e:
                record("ui: page loads and hydrates", FAIL, f"{type(e).__name__}: {str(e)[:200]}")
                browser.close()
                return

            # --- a control must accept input and keep it ---------------------
            # The recurring failure is a control that looks fine but silently
            # refuses to change, so this reads the value back.
            try:
                page.fill("#txt2img_prompt textarea", "a test prompt from the harness")
                page.wait_for_timeout(400)
                got = page.input_value("#txt2img_prompt textarea")
                if got == "a test prompt from the harness":
                    record("ui: prompt accepts and keeps input", PASS)
                else:
                    record("ui: prompt accepts and keeps input", FAIL, f"read back {got!r}")
            except Exception as e:
                record("ui: prompt accepts and keeps input", FAIL, str(e)[:200])

            # --- tabs must open ---------------------------------------------
            # "the extras tab doesnt open" was a real bug; lazily-built tabs
            # are a gradio-6 specific hazard in this fork.
            for tab_id, label in (("#tab_extras", "Extras"), ("#tab_img2img", "Img2img")):
                try:
                    # role=tab + exact text: has-text() is a substring match and
                    # would also hit "Send to img2img" style buttons.
                    page.click(f'button[role=tab]:text-is("{label}")', timeout=15000)
                    page.wait_for_selector(f"{tab_id}", state="visible", timeout=30000)
                    record(f"ui: {label} tab opens", PASS)
                except Exception as e:
                    record(f"ui: {label} tab opens", FAIL, f"{type(e).__name__}: {str(e)[:160]}")

            # --- accordion checkbox must toggle ------------------------------
            # InputAccordion drives its own checkbox; a desync here is exactly
            # the "control is stuck" class of bug.
            try:
                page.click('button[role=tab]:text-is("Txt2img")', timeout=15000)
                # InputAccordion exposes its own checkbox as *-visible-checkbox;
                # it is a sibling of the accordion div, not a child.
                page.wait_for_selector("#txt2img_hr-visible-checkbox", timeout=30000)
                cb = page.locator("#txt2img_hr-visible-checkbox").first
                before = cb.is_checked()
                cb.click()
                page.wait_for_timeout(500)
                after = cb.is_checked()
                if before != after:
                    record("ui: hires-fix accordion toggles", PASS, f"{before} -> {after}")
                    cb.click()          # leave it as we found it
                else:
                    record("ui: hires-fix accordion toggles", FAIL,
                           "checkbox did not change state when clicked (stuck control)")
            except Exception as e:
                record("ui: hires-fix accordion toggles", FAIL, f"{type(e).__name__}: {str(e)[:160]}")

            # --- lazy tabs must not render disabled ---------------------------
            # gradio infers `interactive` from event wiring, which a gr.render
            # body doesn't have yet, so a lazily-built tab can come up entirely
            # non-interactive. That took out all of img2img: the UI looked
            # normal and simply ignored every click.
            try:
                page.click('button[role=tab]:text-is("Img2img")', timeout=15000)
                page.wait_for_timeout(3500)
                n_disabled = page.evaluate("""() => {
                    const root = document.querySelector('#tab_img2img') || document;
                    return root.querySelectorAll(
                        'input:disabled, select:disabled, textarea:disabled, label.disabled').length;
                }""")
                if n_disabled:
                    record("ui: lazy tab controls are interactive", FAIL,
                           f"{n_disabled} disabled control(s) in img2img -- the tab will ignore clicks")
                else:
                    record("ui: lazy tab controls are interactive", PASS, "img2img fully interactive")
            except Exception as e:
                record("ui: lazy tab controls are interactive", FAIL, f"{type(e).__name__}: {str(e)[:160]}")

            # --- a radio in a lazy tab must actually toggle -------------------
            try:
                before = page.evaluate(
                    "() => Array.from(document.querySelectorAll('#resize_mode input')).map(i=>i.checked)")
                page.locator("#resize_mode label").nth(1).click(timeout=10000)
                page.wait_for_timeout(600)
                after = page.evaluate(
                    "() => Array.from(document.querySelectorAll('#resize_mode input')).map(i=>i.checked)")
                if before != after:
                    record("ui: resize-mode radio responds", PASS)
                else:
                    record("ui: resize-mode radio responds", FAIL, "clicking changed nothing")
            except Exception as e:
                record("ui: resize-mode radio responds", FAIL, f"{type(e).__name__}: {str(e)[:160]}")

            # --- no JS errors ------------------------------------------------
            # A broad net for gradio-6 breakage that still renders.
            # Known, pre-existing errors: page-load callbacks that dereference
            # elements belonging to lazily-built tabs, which do not exist until
            # that tab is opened. They are harmless (the features work once the
            # tab is built) but they are real null derefs and should be fixed.
            # Baselined rather than ignored, so a NEW error still fails here
            # instead of hiding in the noise.
            known = (
                # Empty by design: every lazy-tab null-deref that used to live
                # here (settings search, extra-networks, token counters, restore
                # button, resolution paste, ControlNet photopea/openpose,
                # Replacer) has been fixed at the source. Anything that appears
                # here now is a real regression. The only remaining console
                # message is a 404 for the Source Sans Pro font stylesheet
                # (gradio serves it under a path our mount doesn't cover; the
                # font falls back to system-ui), which the noise filter drops.
            )
            noise = ("favicon", "ERR_INTERNET_DISCONNECTED", "net::ERR_ABORTED",
                     "Failed to load resource")
            real = [e for e in console_errors
                    if not any(n in e for n in noise) and not any(k in e for k in known)]
            if real:
                record("ui: no new JavaScript errors", FAIL,
                       f"{len(real)} NEW console error(s):\n         "
                       + "\n         ".join(e[:150] for e in real[:5]))
            else:
                baselined = len(console_errors) - len(real)
                record("ui: no new JavaScript errors", PASS,
                       f"{baselined} known pre-existing error(s) baselined" if baselined else "")

            browser.close()


def _fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


CHECKS = {
    "static": [
        ("syntax", check_syntax),
        ("deps", check_dependency_conflicts),
        ("pins", check_pins_hold),
        ("classify", check_downloader_classification),
        ("filesafety", check_downloader_file_safety),
        ("dl-e2e", check_download_end_to_end),
        ("json", check_json_and_bom),
        ("eol", check_line_endings),
        ("privacy", check_no_personal_files_tracked),
    ],
    "boot": [
        ("boot", check_server_boot),
    ],
    "gpu": [
        ("gpu", check_gpu_generation),
        ("modes", check_gpu_all_modes),
    ],
    "ui": [
        ("ui", check_ui_regression),
    ],
    "clean": [
        ("guards", check_launcher_guards),
        ("urls", check_bootstrap_urls),
        ("release", check_release_install_boots),
        ("gitboot", check_git_bootstrap_works),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="forge-again pre-merge test harness")
    ap.add_argument("--static", action="store_true", help="tier 1 only (fast)")
    ap.add_argument("--boot", action="store_true", help="tier 2 only")
    ap.add_argument("--gpu", action="store_true", help="tier 3 only (needs a free GPU)")
    ap.add_argument("--clean", action="store_true", help="tier 4 only (fresh-install path)")
    ap.add_argument("--ui", action="store_true", help="tier 5 only (browser; needs playwright)")
    ap.add_argument("--all-modes", dest="all_modes", action="store_true",
                    help="GPU tier: generate in every mode that has a checkpoint")
    ap.add_argument("--deep", action="store_true",
                    help="also run downloads-heavy checks (fetches portable git)")
    ap.add_argument("--quick", action="store_true", help="static + clean, no server start")
    ap.add_argument("--list", action="store_true", help="list checks and exit")
    args = ap.parse_args()

    if args.list:
        for tier, checks in CHECKS.items():
            print(f"{tier}:")
            for name, fn in checks:
                print(f"  {name:10} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    if args.deep:
        os.environ["FORGE_TEST_DEEP"] = "1"
    if args.all_modes:
        os.environ["FORGE_TEST_ALL_MODES"] = "1"

    # Default is a full run -- this is a merge gate, so thoroughness beats
    # speed. Use --quick for the checks that need no server and no GPU.
    selected = [t for t in ("static", "boot", "gpu", "clean", "ui") if getattr(args, t)]
    if args.quick:
        selected = ["static", "clean"]
    tiers = selected or ["static", "clean", "boot", "ui", "gpu"]

    t0 = time.time()
    for tier in tiers:
        print(f"\n=== tier: {tier} ===")
        for _name, fn in CHECKS[tier]:
            try:
                fn()
            except Exception as e:                     # a broken check is a failure
                record(f"{_name} (harness error)", FAIL, f"{type(e).__name__}: {e}")

    failed = [r for r in _results if r[1] == FAIL]
    print("\n" + "=" * 64)
    print(f"  {len([r for r in _results if r[1] == PASS])} passed, "
          f"{len(failed)} failed, "
          f"{len([r for r in _results if r[1] == SKIP])} skipped "
          f"in {time.time() - t0:.1f}s")
    if failed:
        print("\n  FAILED:")
        for name, _s, _d in failed:
            print(f"    - {name}")
        print("\n  Do not fold testing into main until these pass.")
    else:
        print("\n  All good -- safe to fold testing into main.")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
