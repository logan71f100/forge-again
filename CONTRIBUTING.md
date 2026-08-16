# Contributing to forge-again

Issues and pull requests are welcome. This is a solo-maintained fork, so please keep
changes focused and well-described — small, self-contained PRs get reviewed fastest.

## Running from source

No system Python or manual setup is needed — the launcher bootstraps everything:

```
git clone https://github.com/logan71f100/forge-again
cd forge-again
start.bat            # Windows   (or ./start.sh on Linux, ./start-macos.sh on macOS)
```

On first run it downloads a portable Python 3.12, builds the venv, installs torch
(cu126) and all dependencies, and launches on `http://0.0.0.0:7860`. Re-runs skip
completed steps. Set `FORGE_NO_LLM=1` to skip the ~18 GB AI-assistant model download
while working on unrelated areas. See the [README](README.md) for the model layout
and launch arguments.

## Branches

- **`main`** — the integration branch; open pull requests against it.
- **`testing`** — work-in-progress that hasn't been promoted to `main` yet.

## Tests

There's a test harness in [`tests/`](tests/README.md) — static checks, a real
server boot, a Playwright pass over the UI, and real generations on the GPU:

```
python tests/run_tests.py --static   # ~1 s, no server
python tests/run_tests.py            # everything (~5 min, needs a free GPU)
```

Day to day, targeted checks are enough. The full suite is the gate before folding
into `main`; it prints `All good -- safe to fold testing into main.` on a pass and
exits non-zero on any failure.

## Before you open a PR

- **Keep it focused.** One logical change per PR. Split unrelated cleanup out.
- **Test the modes you touched** end to end (`sd` / `xl` / `flux` as applicable).
- **Run at least `--static`**, and the full suite if you touched the UI, the
  generation path, or the launchers.
- **Include a screenshot** for anything visual — nearly every gradio-6 bug in this
  fork was caught from a screenshot plus a console traceback.
- **Mind the gradio-6 gotchas.** The migration notes in [PORTING.md](PORTING.md) list
  the recurring traps: `visible=False` components aren't mounted (mount CSS-hidden
  instead), checkboxes/radios bind to `change` not `input`, closed accordions and
  inactive tabs unmount their children, and same-trigger state refreshes must stay
  queued. New UI code should follow those patterns.

## Reporting bugs

Open an issue using the **Bug report** template and fill in the console output and a
screenshot where they apply. For security issues, follow [SECURITY.md](SECURITY.md)
instead of filing a public issue.

## License

By contributing, you agree that your contributions are licensed under **AGPL-3.0**,
the same license as the rest of the project.
