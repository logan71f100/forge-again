"""Mirror the server console to a rotating file.

The console is where Forge says everything that matters when a run misbehaves:
model loads and unloads, `[Memory Management]` decisions, per-run peak VRAM,
step rates, tracebacks. None of it survived. Launching normally sends it to a
window that closes; redirecting with `start.bat > run.log` block-buffers,
because Python only line-buffers when stdout is a console -- so during a stall,
which is exactly when the process will not exit to flush, the log ran thousands
of characters behind and the lines explaining the stall were unreachable.

So the server writes its own log instead of depending on how it was launched.
Enabled by the same `forge_client_forensics` setting that arms the browser-side
forensics, read straight from config.json because this has to install before
shared.opts exists -- the interesting output (model loads) happens during
startup.

Both streams are mirrored, the original is still written through unchanged
(tqdm's carriage returns keep working in the terminal), and the copy is flushed
on every write, since an unflushed diagnostic is not a diagnostic.
"""

import io
import json
import os
import sys
import threading
import time

MAX_BYTES = 8 * 1024 * 1024      # keeps roughly a long session; one .1 backup
_lock = threading.Lock()
_installed = False


class _Tee(io.TextIOBase):
    """Write through to the real stream, and copy into the log."""

    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink

    def write(self, s):
        n = self._stream.write(s)
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._sink.write(s)
        except Exception:
            pass                  # logging must never break the server
        return n

    def flush(self):
        for s in (self._stream, self._sink):
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        # tqdm asks: keep its live-rewriting behaviour on the real console
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._stream.fileno()


class _RotatingSink:
    def __init__(self, path):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8", errors="replace")

    def write(self, s):
        with _lock:
            self._fh.write(s)
            self._fh.flush()      # a stalled server never gets to flush later
            if self._fh.tell() > MAX_BYTES:
                self._rotate()

    def _rotate(self):
        try:
            self._fh.close()
            backup = self.path + ".1"
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(self.path, backup)
        except OSError:
            pass                  # a locked file just keeps growing; not fatal
        self._fh = open(self.path, "a", encoding="utf-8", errors="replace")

    def flush(self):
        with _lock:
            try:
                self._fh.flush()
            except Exception:
                pass


def _enabled(root):
    if os.environ.get("FORGE_CONSOLE_LOG") == "0":
        return False
    if os.environ.get("FORGE_CONSOLE_LOG") == "1":
        return True
    try:
        with open(os.path.join(root, "config.json"), "r", encoding="utf-8") as f:
            return bool(json.load(f).get("forge_client_forensics", False))
    except Exception:
        return False


def install(root=None):
    """Start mirroring stdout/stderr. Safe to call more than once.

    Once per PROCESS TREE, not once per process. Startup spawns short-lived
    children -- the extension installers among them -- which inherit the launch
    and would each open the log and announce themselves; the parent then
    re-prints what it captured from them, so a single launch wrote six headers
    and eleven notices. The marker goes in the environment precisely because
    children inherit it and skip.
    """
    global _installed
    if _installed or os.environ.get("_FORGE_CONSOLE_LOG_ACTIVE") == "1":
        return None
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not _enabled(root):
        return None
    try:
        path = os.path.join(root, "server-console.log")
        sink = _RotatingSink(path)
        sink.write("\n=== forge-again console log opened %s ===\n"
                   % time.strftime("%Y-%m-%d %H:%M:%S"))
        sys.stdout = _Tee(sys.stdout, sink)
        sys.stderr = _Tee(sys.stderr, sink)
        _installed = True
        os.environ["_FORGE_CONSOLE_LOG_ACTIVE"] = "1"   # children inherit; they skip
        print(f"[console-log] mirroring this console to {path} "
              f"(forge_client_forensics is on)")
        return path
    except Exception as e:
        print(f"[console-log] could not start console logging: {e}")
        return None
