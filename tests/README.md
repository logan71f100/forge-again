# Pre-merge test harness

Run this on `testing` before folding it into `main`. No third-party packages —
plain stdlib, so it works anywhere Forge itself runs.

```
run-tests.bat                        # Windows, everything
python tests/run_tests.py            # everything (~20 s)
python tests/run_tests.py --static   # tier 1 only (~1 s)
python tests/run_tests.py --boot     # tier 2 only
python tests/run_tests.py --list     # show checks, run nothing
```

Exit code is non-zero if anything failed, so it can gate a merge.

## Tier 1 — static (~1 s, no GPU)

| Check | Catches |
|---|---|
| `syntax` | Any maintained `.py` that no longer compiles. Vendored trees are excluded. |
| `deps` | New dependency conflicts. `ACCEPTED_CONFLICTS` in the runner lists the ones we've knowingly taken; anything else fails. This is the guard against a transitive install quietly bumping a pinned package. |
| `json` | Malformed JSON, and **UTF-8 BOMs** — a BOM in `config.json` makes Forge fail to read it and silently reset every setting to defaults. |
| `eol` | `.bat` that isn't CRLF (cmd misparses labels) and `.sh` that isn't LF (bash rejects it). |
| `privacy` | Personal or generated files becoming tracked by git — `config.json`, `outputs/`, `extra-args.txt` and friends. |

## Tier 2 — boot (~15 s, no GPU needed)

Starts the real server on a spare port and interrogates it. **It uses its own
config fixture, copied to a temp file — your real `config.json` is never read
or written.**

| Check | Catches |
|---|---|
| `boot` | The server failing to start at all. |
| `no tracebacks` | Any exception during startup. Missing extension dependencies show up exactly this way. |
| `all scripts load` | `Error loading script` / callback failures, which Forge otherwise prints and carries on from. |
| `core options round-trip` | A core setting being renamed, dropped, or silently reset — the fixture uses deliberately non-default values, so a match proves the server really read them. |
| `settings survive a lifecycle` | Options being pruned from the config file across a start/stop, including extension options. |
| `sd-models responds` | The API not coming up with the UI. |

## The config fixture

`tests/fixtures/test-config.json` is the harness's own config. Values are
non-default on purpose — if the server ignored the file, the round-trip check
would see defaults and fail.

Add a `"key": value` entry there to bring another option under test. Core
options are additionally verified through `/sdapi/v1/options`; extension
options are verified for persistence only, because **Forge does not expose
extension-registered settings through that endpoint at all**. That was confirmed
against a real 425-key config: 33 `replacer_*` and 14 `forge_ai_*` options were
present in the file and absent from the API response. It's long-standing Forge
behaviour rather than a regression, so the harness works with it instead of
failing on it.

## Not covered yet

Deliberate gaps, in rough priority order:

- **GPU generation** — txt2img/img2img/inpaint per mode, asserting real images.
  Needs a GPU, so it can't run on a hosted CI runner.
- **UI regression** — driving the actual UI to catch gradio-6 issues like a
  control that stops responding after a profile is applied. This is where the
  recurring bugs in this fork have lived, and nothing below it can see them.
- **Docker image** — building and booting the container as part of the suite.

## Adding a check

Write a function that calls `record(name, PASS|FAIL|SKIP, detail)`, then add it
to `CHECKS` under the right tier. A check that raises is reported as a failure
rather than taking the run down.
