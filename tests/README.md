# Pre-merge test harness

Run this on `testing` before folding it into `main`. Every tier but the UI one
is plain stdlib, so it works anywhere Forge itself runs; the UI tier needs
Playwright and skips itself when it isn't installed.

Day to day on `testing`, targeted checks are enough — a syntax pass, a `--static`
run, a browser spot-check. The full suite is the **fold gate**: run it before
merging into `main`, not per commit.

```
run-tests.bat                          # Windows, everything
python tests/run_tests.py              # everything (~5 min)
python tests/run_tests.py --static     # tier 1 only (~1 s)
python tests/run_tests.py --boot       # tier 2 only
python tests/run_tests.py --ui         # tier 3 only (browser; needs playwright)
python tests/run_tests.py --gpu        # tier 4 only (needs a free GPU)
python tests/run_tests.py --clean      # tier 5 only
python tests/run_tests.py --quick      # static + clean, no server start
python tests/run_tests.py --all-modes  # also generate in every mode (a minute each)
python tests/run_tests.py --deep       # also exercise the portable-git bootstrap
python tests/run_tests.py --list       # show checks, run nothing
```

Exit code is non-zero if anything failed, so it can gate a merge. On a pass it
prints `All good -- safe to fold testing into main.`

Redirect the output if you want to read it as it goes: Python buffers stdout
through a pipe, so `python -u` is what gives you live progress rather than
everything at the end.

## Tier 1 — static (~1 s, no GPU)

| Check | Catches |
|---|---|
| `syntax` | Any maintained `.py` that no longer compiles. Vendored trees are excluded. |
| `deps` | New dependency conflicts. `ACCEPTED_CONFLICTS` in the runner lists the ones we've knowingly taken; anything else fails. This is the guard against a transitive install quietly bumping a pinned package. |
| `json` | Malformed JSON, and **UTF-8 BOMs** — a BOM in `config.json` makes Forge fail to read it and silently reset every setting to defaults. |
| `eol` | `.bat` that isn't CRLF (cmd misparses labels) and `.sh` that isn't LF (bash rejects it). |
| `privacy` | Personal or generated files becoming tracked by git — `config.json`, `outputs/`, `extra-args.txt` and friends. |
| `pins` | Installed versions drifting from the `==` pins. Extension installers run on every startup and pull packages past their caps, so this drifts silently on a working machine. |
| `classify` | The model downloader sorting a file into the wrong folder. Cases include the real regressions: `Kataragi_inpaintXL` (a ControlNet the "xl" rule used to claim as a checkpoint), `ae.safetensors` (the Flux VAE, which contains no "vae"), and the `JuggernautXL`-style names that broke when the XL match was made too strict. |
| `error-tips` | A recognizable runtime error losing its plain-language tip, and the tip losing the field it highlights. |
| `hints` | The `generation_hints` rule registry and `error_tips.register_tip` staying usable — these exist so failure messages can be extended without touching the generation path. |
| `filesafety` | The downloader's move/delete escaping their folder or clobbering files. Asserts path traversal is rejected, a move never overwrites an existing destination, and delete removes exactly the target. |
| `dl-e2e` | The download path end to end: actually fetches a file and confirms it lands in the right folder. |

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

## Tier 3 — GPU (~60 s, needs a free GPU)

Asserts images are **correct**, not merely that the endpoint returned 200. One
server session is shared, so the checkpoint load is paid once.

| Check | Catches |
|---|---|
| `txt2img size` | The requested resolution being ignored. |
| `not blank` | Black frames / VAE overflow — a 200 with a useless image. |
| `same seed` | Seed handling regressions. Compared with a tolerance, not byte-equality: Forge offloads dynamically, so an identical request can land a few LSBs apart under VRAM pressure. A real regression moves the mean diff into the tens. |
| `hires fix` | Hires returning the base resolution or a wrong aspect. Found a live bug: every API hires request 500'd. |
| `inpaint` | A mask being **ignored or inverted**. Compares the masked region against the untouched one, so a plausible-but-wrong image still fails. This is the operation Replacer is built on. |
| `img2img` | The input being passed through unchanged. |
| `generation hints` | Hints stopping at the console instead of reaching the user — asserts a rule fires and its text arrives in the result's `comments`, which is what the UI renders under the image. |
| `deepbooru` | The interrogator returning nothing, which is how its model-loading breaks present. |
| `every mode` | `--all-modes` only: generates in every mode that has a checkpoint, so an sd- or flux-specific break shows up while you're working in xl. |

## Tier 4 — clean install (~30 s)

| Check | Catches |
|---|---|
| `guards` | The launchers losing their git bootstrap, exit-code check, or `pause` — the regression that made `start.bat` exit silently. |
| `urls` | Link rot in the hardcoded Python/MinGit downloads. Invisible on a working machine, fatal on a new one. |
| `release` | Boots the server from a `git archive` export — no `.git`, no dev files — which is what a downloader actually runs. Verified to fail when the "fatal: not a git repository" fix is reverted. Note it exports **committed HEAD**, so an uncommitted fix looks broken here. |
| `gitboot` | `--deep` only: fetches portable git with git hidden and clones with it. |

## Tier 5 — UI (~3 min, browser)

Drives the real UI through Playwright (`pip install playwright && playwright
install chromium`; the tier skips itself if it isn't installed). This is where
this fork's recurring bugs live: gradio 6 mounts tabs on demand, so a control
can exist, look normal, and quietly ignore every click.

| Check | Catches |
|---|---|
| `page loads and hydrates` | A page that renders but never wires up. |
| `prompt accepts input` | The most basic control losing its value. |
| `send-to builds an unopened destination` | Send-to targeting a tab that hasn't been built yet — and the same thing from a **second** page session, which is what broke when process-wide paste registries kept the first session's dead component ids. |
| `Settings / Extensions / Extras / Img2img render` | A tab failing to build at all. |
| `hires-fix accordion toggles` | An InputAccordion that stops responding. |
| `lazy tab controls are interactive` | The whole of img2img coming up non-interactive — gradio infers `interactive` from event wiring, which a `gr.render` body doesn't have yet. |
| `session capture survives a tab switch` | The assistant's session capture missing values changed on a tab you have since left, or a nested tab selection ("Resize to / Resize by") it cannot see at all. Asserts the values reach `last_session.json` **on disk**. |
| `restore re-selects a nested tab` | Restore putting settings back but not the tab selection that decides which of them the backend uses. |
| `resize-mode radio responds` | A radio inside a lazily-built tab not toggling. |
| `img2img canvas syncs an upload` | ForgeCanvas not writing the upload into the component value the backend reads. |
| `detect-size reads dimensions` | The detect-size button not reaching the source image. |
| `mode switch refreshes checkpoint list` | The sd/xl/flux switch not re-scanning, in both directions. |
| `attributed error highlights its control` | An error losing the control it points at. |
| `no new JavaScript errors` | Console errors beyond a known baseline. The baseline counts one instance per page load, so a check that reloads legitimately raises the count. |
| `get_tab_index returns the real sub-tab` | Sub-tab index drift, which silently sends a generation to the wrong mode. |
| `extra-networks search filters` | The Lora/Checkpoints search box vanishing or not filtering. |

**Writing a UI check:** the tier shares ONE page across every check, so anything
you leave behind lands on a later one — and it fails *there*, several checks
away from the cause. Leaving a nested tab selected unmounts the pane a later
check clicks in; opening the assistant panel floats it over the right-hand
controls. End a disruptive check with `page.reload()` and re-open its tab rather
than unwinding state by hand. Two Playwright traps specific to this UI: gradio
renders more than one element per tab, so a bare `page.click('#x
button:text-is("Y")')` trips strict mode, and adding `>> visible=true` then
matches *nothing*, because headless reports tab-bar buttons as zero-box. Reach
nested tabs through `page.evaluate` over the tab bar's direct children.

## Not covered yet

Deliberate gaps, in rough priority order:

- **Docker image** — building and booting the container as part of the suite.
- **Generation from the UI** — the UI tier exercises controls, and the GPU tier
  generates through the API, but nothing clicks Generate and waits for a real
  image. The background-tab and progress-tracking bugs live in exactly that gap.
- **Extension surfaces** — Replacer and Segment Anything have their own tabs and
  neither is driven by the UI tier.

## Adding a check

Write a function that calls `record(name, PASS|FAIL|SKIP, detail)`, then add it
to `CHECKS` under the right tier. A check that raises is reported as a failure
rather than taking the run down.
