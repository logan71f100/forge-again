"""First Block Cache (FBCache): skip most of a diffusion model's forward pass
when the step is predicted to change little.

The idea (TeaCache / WaveSpeed "First Block Cache"): run only the first
transformer block (or the first UNet input stage), compare its residual with
the residual seen on the previous computed step, and if the relative change is
below a threshold assume the rest of the network would also change little --
reuse the cached residual of the remaining blocks instead of computing them.
On a low-VRAM card this also skips the weight swap-ins for every block that is
not run, which is where most of the wall-clock goes when the model does not
fit.

State lives in one process-wide cache (Forge runs one generation at a time).
The script in extensions-builtin/sd_forge_first_block_cache configures it per
run and resets it before every sampling pass, so hires-fix passes never see
residuals from the first pass.

Cond and uncond may arrive in one batched call or as separate calls at the
same timestep. Calls are numbered within a timestep and each slot keeps its
own residuals, so slot k of step n is only ever compared with slot k of step
n-1. A mismatch (different batch shape, or a genuinely different input) just
fails the similarity test and computes the step in full -- the cache can only
skip work, never produce output from the wrong stream.
"""
import torch


class FirstBlockCache:
    def __init__(self):
        self.enabled = False
        self.threshold = 0.12
        self.start = 0.0
        self.end = 1.0
        self.max_consecutive = -1
        self.reset()

    def configure(self, enabled, threshold=0.12, start=0.0, end=1.0, max_consecutive=-1):
        self.enabled = bool(enabled)
        self.threshold = float(threshold)
        self.start = float(start)
        self.end = float(end)
        self.max_consecutive = int(max_consecutive)
        self.reset()

    def disable(self):
        self.enabled = False
        self.reset()

    def reset(self):
        self.entries = {}
        self.last_t = None
        self.call_idx = -1
        self.hits = 0
        self.calls = 0

    # -- per-forward protocol ---------------------------------------------

    def begin_call(self, progress, t_key):
        """Called once per model forward. `progress` is 0.0 at pure noise and
        1.0 at the end of sampling; `t_key` identifies the timestep (its value
        as a float) so calls within one step can be numbered.

        Returns a slot key, or None when the cache must not be used for this
        call (disabled, or outside the configured progress window)."""
        if not self.enabled:
            return None
        if self.last_t is None or t_key != self.last_t:
            if self.last_t is not None and t_key > self.last_t:
                # timesteps only decrease within a run: a jump back up means a
                # new sampling pass started without reset() -- drop everything
                self.entries = {}
            self.last_t = t_key
            self.call_idx = 0
        else:
            self.call_idx += 1
        self.calls += 1
        if progress < self.start or progress > self.end:
            return None
        return self.call_idx

    def should_use_cache(self, key, first_residual):
        """Decide from the first block's residual whether the rest of the
        network can be skipped. On a miss the new residual becomes the
        reference; on a hit the reference is kept, so drift accumulates until
        it crosses the threshold (this is what bounds the error)."""
        if key is None:
            return False
        entry = self.entries.get(key)
        if entry is None or entry['hidden_residual'] is None or entry['first_residual'].shape != first_residual.shape:
            self.entries[key] = {'first_residual': first_residual, 'hidden_residual': None, 'hits': 0}
            return False
        if self.max_consecutive >= 0 and entry['hits'] >= self.max_consecutive:
            entry['first_residual'] = first_residual
            entry['hits'] = 0
            return False
        prev = entry['first_residual']
        diff = relative_l1_distance(prev, first_residual)
        if diff < self.threshold:
            entry['hits'] += 1
            self.hits += 1
            return True
        entry['first_residual'] = first_residual
        entry['hits'] = 0
        return False

    def store(self, key, hidden_residual):
        if key is None:
            return
        entry = self.entries.get(key)
        if entry is not None:
            entry['hidden_residual'] = hidden_residual

    def apply(self, key, hidden):
        return hidden + self.entries[key]['hidden_residual']

    def summary(self):
        if self.calls == 0:
            return "First Block Cache: no forward passes recorded"
        return f"First Block Cache: skipped {self.hits} of {self.calls} model forward passes ({100.0 * self.hits / self.calls:.0f}%)"


def relative_l1_distance(prev, cur):
    """mean|cur - prev| / mean|prev|, computed in float32 so fp16 residuals do
    not overflow the reduction."""
    prev32 = prev.float()
    denom = prev32.abs().mean()
    if denom == 0:
        return float('inf')
    return ((cur.float() - prev32).abs().mean() / denom).item()


cache = FirstBlockCache()
