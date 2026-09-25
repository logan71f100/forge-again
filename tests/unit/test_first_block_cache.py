"""Unit tests for backend/misc/first_block_cache.py (CPU torch only).

    python -m unittest tests.unit.test_first_block_cache
"""
import unittest

import torch

from backend.misc.first_block_cache import FirstBlockCache, relative_l1_distance


class RelativeL1(unittest.TestCase):
    def test_identical_is_zero(self):
        a = torch.randn(2, 8, 4)
        self.assertEqual(relative_l1_distance(a, a.clone()), 0.0)

    def test_scaled(self):
        a = torch.ones(3, 3)
        self.assertAlmostEqual(relative_l1_distance(a, a * 1.1), 0.1, places=5)

    def test_zero_reference_never_matches(self):
        self.assertEqual(relative_l1_distance(torch.zeros(4), torch.ones(4)), float('inf'))

    def test_fp16_does_not_overflow(self):
        a = (torch.ones(1000) * 60000).half()
        self.assertAlmostEqual(relative_l1_distance(a, a), 0.0)


class CacheProtocol(unittest.TestCase):
    def setUp(self):
        self.cache = FirstBlockCache()
        self.cache.configure(True, threshold=0.1)

    def step(self, t, first_residual, hidden_residual=None):
        """Drive one forward pass through the cache protocol; returns whether it hit."""
        key = self.cache.begin_call(progress=1.0 - t, t_key=t)
        hit = self.cache.should_use_cache(key, first_residual)
        if not hit and hidden_residual is not None:
            self.cache.store(key, hidden_residual)
        return hit

    def test_first_call_never_hits(self):
        self.assertFalse(self.step(1.0, torch.ones(4)))

    def test_hits_when_residual_unchanged(self):
        r, h = torch.ones(4), torch.full((4,), 5.0)
        self.assertFalse(self.step(1.0, r, h))
        self.assertTrue(self.step(0.9, r.clone()))
        # a hit adds the stored residual to the given hidden state
        key = self.cache.call_idx
        self.assertTrue(torch.equal(self.cache.apply(key, torch.zeros(4)), h))
        self.assertEqual(self.cache.hits, 1)

    def test_misses_when_residual_moved(self):
        r = torch.ones(4)
        self.assertFalse(self.step(1.0, r, torch.zeros(4)))
        self.assertFalse(self.step(0.9, r * 1.5, torch.zeros(4)))

    def test_reference_kept_across_hits_so_drift_accumulates(self):
        # each step moves 6%: below the 10% threshold step to step, but the
        # reference is the last COMPUTED step, so the third comparison (18%) misses
        r = torch.ones(4)
        self.assertFalse(self.step(1.0, r, torch.zeros(4)))
        self.assertTrue(self.step(0.9, r * 1.06))
        self.assertFalse(self.step(0.8, r * 1.12, torch.zeros(4)))  # 12% vs the reference -> miss
        self.assertEqual(self.cache.hits, 1)

    def test_cond_and_uncond_slots_are_separate(self):
        cond, uncond = torch.ones(4), torch.ones(4) * -1
        # step 1: two calls at the same t
        self.assertFalse(self.step(1.0, cond, torch.zeros(4)))
        self.assertFalse(self.step(1.0, uncond, torch.ones(4)))
        # step 2: each slot compares against its own stream and hits
        self.assertTrue(self.step(0.9, cond.clone()))
        self.assertTrue(self.step(0.9, uncond.clone()))
        self.assertEqual(self.cache.hits, 2)

    def test_batched_call_is_one_slot(self):
        r = torch.ones(2, 4)
        self.assertFalse(self.step(1.0, r, torch.zeros(2, 4)))
        self.assertTrue(self.step(0.9, r.clone()))

    def test_shape_change_misses(self):
        self.assertFalse(self.step(1.0, torch.ones(4), torch.zeros(4)))
        self.assertFalse(self.step(0.9, torch.ones(8)))

    def test_new_run_resets_when_t_jumps_up(self):
        r = torch.ones(4)
        self.assertFalse(self.step(1.0, r, torch.zeros(4)))
        self.assertTrue(self.step(0.5, r.clone()))
        # a hires pass starts at a higher t again: must not reuse the residuals
        self.assertFalse(self.step(0.8, r.clone(), torch.zeros(4)))
        self.assertTrue(self.step(0.7, r.clone()))

    def test_window_excludes_early_and_late_steps(self):
        self.cache.configure(True, threshold=0.1, start=0.2, end=0.8)
        r = torch.ones(4)
        self.assertIsNone(self.cache.begin_call(progress=0.1, t_key=0.9))
        self.assertIsNotNone(self.cache.begin_call(progress=0.5, t_key=0.5))
        self.assertIsNone(self.cache.begin_call(progress=0.9, t_key=0.1))
        self.assertFalse(self.cache.should_use_cache(None, r))

    def test_max_consecutive_forces_a_full_step(self):
        self.cache.configure(True, threshold=0.1, max_consecutive=1)
        r = torch.ones(4)
        self.assertFalse(self.step(1.0, r, torch.zeros(4)))
        self.assertTrue(self.step(0.9, r.clone()))
        self.assertFalse(self.step(0.8, r.clone(), torch.zeros(4)))
        self.assertTrue(self.step(0.7, r.clone()))

    def test_disabled_cache_is_inert(self):
        self.cache.disable()
        self.assertIsNone(self.cache.begin_call(progress=0.5, t_key=0.5))
        self.assertIn("no forward passes", self.cache.summary())


class FluxModelHook(unittest.TestCase):
    """Runs a tiny IntegratedFluxTransformer2DModel with the cache on and off."""

    @classmethod
    def setUpClass(cls):
        try:
            import sys
            sys.argv = [sys.argv[0], '--always-cpu']   # backend.memory_management probes CUDA at import
            from backend.nn.flux import IntegratedFluxTransformer2DModel
            from backend.misc.first_block_cache import cache
        except Exception as e:  # missing heavy deps (diffusers, psutil, ...)
            raise unittest.SkipTest(f"backend model imports unavailable: {e}")
        torch.manual_seed(0)
        cls.cache = cache
        cls.model = IntegratedFluxTransformer2DModel(
            in_channels=4, vec_in_dim=8, context_in_dim=8, hidden_size=32, mlp_ratio=2.0, num_heads=2,
            depth=2, depth_single_blocks=2, axes_dim=[4, 6, 6], theta=10000, qkv_bias=True, guidance_embed=False,
        ).eval()

    def run_model(self, t):
        x = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
        ctx = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(2))
        y = torch.randn(1, 8, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            return self.model(x, torch.tensor([t]), ctx, y)

    def test_disabled_and_never_hitting_paths_agree(self):
        self.cache.disable()
        ref = self.run_model(0.9)
        self.cache.configure(True, threshold=0.0)   # a full step, cache bookkeeping on
        out = self.run_model(0.9)
        self.cache.disable()
        self.assertEqual(out.shape, (1, 4, 8, 8))
        self.assertTrue(torch.allclose(ref, out, atol=1e-6))

    def test_hit_skips_blocks_and_stays_finite(self):
        calls = {'double1': 0, 'single0': 0}
        h1 = self.model.double_blocks[1].register_forward_hook(lambda *a: calls.__setitem__('double1', calls['double1'] + 1))
        h2 = self.model.single_blocks[0].register_forward_hook(lambda *a: calls.__setitem__('single0', calls['single0'] + 1))
        try:
            self.cache.configure(True, threshold=float('inf'))
            self.run_model(1.0)
            self.assertEqual(calls, {'double1': 1, 'single0': 1})   # full step: everything ran
            second = self.run_model(0.9)
            self.assertEqual(self.cache.hits, 1)
            self.assertEqual(calls, {'double1': 1, 'single0': 1})   # hit: later blocks were skipped
            self.assertEqual(second.shape, (1, 4, 8, 8))
            self.assertTrue(torch.isfinite(second).all())
            # a full step is forced when the residual moves past the threshold
            self.cache.threshold = 0.0
            self.run_model(0.8)
            self.assertEqual(calls, {'double1': 2, 'single0': 2})
        finally:
            h1.remove()
            h2.remove()
            self.cache.disable()


if __name__ == '__main__':
    unittest.main()
