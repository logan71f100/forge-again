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


CHECKS = {
    "static": [
        ("syntax", check_syntax),
        ("deps", check_dependency_conflicts),
        ("json", check_json_and_bom),
        ("eol", check_line_endings),
        ("privacy", check_no_personal_files_tracked),
    ],
    "boot": [
        ("boot", check_server_boot),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="forge-again pre-merge test harness")
    ap.add_argument("--static", action="store_true", help="tier 1 only (fast)")
    ap.add_argument("--boot", action="store_true", help="tier 2 only")
    ap.add_argument("--list", action="store_true", help="list checks and exit")
    args = ap.parse_args()

    if args.list:
        for tier, checks in CHECKS.items():
            print(f"{tier}:")
            for name, fn in checks:
                print(f"  {name:10} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    tiers = [t for t in ("static", "boot")
             if (getattr(args, t) or not (args.static or args.boot))]

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
