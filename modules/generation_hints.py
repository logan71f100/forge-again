"""Plain-language hints shown under the generation result.

Counterpart to error_tips (which explains FAILURES): these surface useful
non-fatal facts about a run that would otherwise hide in the console --
"your LoRA was applied online in fp16", "the reserve was raised for this
run", "these settings defeat your flash LoRA".

Two ways to add a hint:

1. One-off, from any code that has the processing object:

       from modules import generation_hints
       generation_hints.add(p, "text")        # deduped per run

2. Registered rule, evaluated once per generation just before the result
   is assembled -- return a string to show it, or None to stay silent:

       @generation_hints.rule
       def my_hint(p):
           if something(p):
               return "text"

Rules run inside try/except: a broken rule logs once and never breaks a
generation. Extensions may import and use both freely.
"""

import re

_RULES = []


def rule(fn):
    """Decorator: register fn(p) -> str | None as a per-generation hint rule."""
    _RULES.append(fn)
    return fn


def register_rule(fn):
    """Non-decorator registration, for extensions."""
    _RULES.append(fn)
    return fn


def add(p, text):
    """Attach a one-off hint to this run; shows under the generation result."""
    try:
        p.comment(str(text))
    except Exception:
        pass


def run_rules(p):
    """Evaluate every registered rule for this run. Called by processing."""
    for fn in _RULES:
        try:
            text = fn(p)
            if text:
                add(p, text)
        except Exception as e:
            print(f"[hints] rule {getattr(fn, '__name__', fn)!r} failed: {e}")


# ---- built-in rules ---------------------------------------------------------

@rule
def _flash_lora_settings(p):
    """Flash/distill LoRAs are trained for CFG 1.0 and few steps; running them
    at real CFG or high step counts multiplies the work with no quality gain.
    The client-side guard catches UI clicks -- this covers API/assistant runs
    and documents the condition in the result info either way."""
    prompt = (getattr(p, 'prompt', '') or '')
    if not re.search(r'<lora:[^>]*flash[^>]*:', prompt, re.IGNORECASE):
        return None
    cfg = getattr(p, 'cfg_scale', 1) or 1
    steps = getattr(p, 'steps', 0) or 0
    problems = []
    if cfg > 1.0:
        problems.append(f"CFG {cfg} (flash wants 1.0; CFG>1 doubles every step)")
    if steps > 16:
        problems.append(f"{steps} steps (flash converges in 8-16)")
    if not problems:
        return None
    return "Hint: flash LoRA with " + " and ".join(problems) + " -- this runs several times slower than the intended flash setup."
