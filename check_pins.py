#!/usr/bin/env python3
"""Verify — and repair — the pinned dependency versions.

Two things push an existing install off its pins:

1. `requirements_versions.txt` changes between releases, but the launcher's
   install step is gated on a stamp file, so an upgraded install never re-runs
   pip. Security bumps therefore only reached fresh installs.
2. `launch.py` installs extension requirements at startup. Some of those (for
   example onnxruntime, which wants protobuf>=4.25.8) pull versions straight
   past a cap this project deliberately documents — `protobuf<4`, required by
   open-clip-torch, the SDXL text-encoder dependency.

Because (1) means nothing ever re-applies the pins, drift from (2) is permanent.
This script closes that loop: it compares installed versions against every `==`
pin and reinstalls the ones that drifted. It's cheap when nothing is wrong
(metadata reads only, no network), so the launchers run it on every start.

    FORGE_NO_PIN_CHECK=1   skip entirely
    --check                report drift, change nothing (exit 1 if drifted)
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQS = os.path.join(HERE, "requirements_versions.txt")

# torch/torchvision carry a +cu126 local version and live on a separate index;
# reinstalling them here would be slow and could silently fetch a CPU wheel.
SKIP = {"torch", "torchvision"}


def normalise(name):
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def pinned():
    """{package: version} for exact (==) pins, ignoring comments and options."""
    out = {}
    if not os.path.exists(REQS):
        return out
    for raw in open(REQS, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*([A-Za-z0-9._*+!-]+)$", line)
        if m:
            name, ver = normalise(m.group(1)), m.group(2)
            if name not in SKIP and "+" not in ver:
                out[name] = ver
    return out


def installed():
    import importlib.metadata as md
    found = {}
    for dist in md.distributions():
        name = dist.metadata["Name"]
        if name:
            found[normalise(name)] = dist.version
    return found


def main():
    if os.environ.get("FORGE_NO_PIN_CHECK"):
        return 0

    want, have = pinned(), installed()
    drift = [(p, v, have[p]) for p, v in want.items() if p in have and have[p] != v]
    missing = [p for p in want if p not in have]

    if not drift:
        if missing:
            print(f"[pins] {len(missing)} pinned package(s) not installed: {', '.join(sorted(missing)[:6])}")
        return 0

    report = ", ".join(f"{p} {got}!={vers}" for p, vers, got in drift)
    if "--check" in sys.argv:
        print(f"[pins] DRIFT: {report}")
        return 1

    print(f"[pins] {len(drift)} package(s) drifted from requirements_versions.txt:")
    for p, vers, got in drift:
        print(f"[pins]   {p}: installed {got}, pinned {vers}")
    print("[pins] restoring pinned versions (extension installers can pull these past their caps) ...")

    specs = [f"{p}=={vers}" for p, vers, _got in drift]
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "--no-input", *specs],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print("[pins] WARNING: could not restore pins:")
        print((proc.stderr or proc.stdout or "").strip()[-600:])
        return 0        # never block startup
    print("[pins] pinned versions restored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
