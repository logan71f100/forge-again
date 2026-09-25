"""Unit tests for the solvers in backend/modules/k_diffusion_extra.py (CPU torch).

Every sampler is run against a toy denoiser whose x0 prediction is a fixed
target image: with a perfect denoiser any consistent solver must land exactly
on the target once sigma reaches 0, and must never produce NaN/inf along the
way. The rectified-flow helpers are checked against closed forms.

    python -m unittest tests.unit.test_samplers
"""
import unittest

import torch

try:
    import sys
    sys.argv = [sys.argv[0], '--always-cpu']   # backend.memory_management probes CUDA at import
    from backend.modules import k_diffusion_extra as kde
except Exception as e:  # heavy import chain (diffusers)
    kde = None
    IMPORT_ERROR = e


class _Predictor:
    """stands in for backend.modules.k_prediction predictors"""
    def percent_to_sigma(self, percent):
        return 1.0 - percent


class ToyDenoiser(torch.nn.Module):
    """x0-prediction model that always returns `target` (a perfect denoiser)."""
    def __init__(self, target, rf=False):
        super().__init__()
        self.target = target
        self.calls = 0
        inner = torch.nn.Module()
        inner.predictor = _Predictor()
        self.inner_model = inner
        self._rf = rf

    def forward(self, x, sigma, **kwargs):
        self.calls += 1
        return self.target.expand_as(x).clone()


def karras_sigmas(n, sigma_min=0.03, sigma_max=14.6, rho=7.0):
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def flow_sigmas(n):
    return torch.cat([torch.linspace(1.0, 0.05, n), torch.zeros(1)])


@unittest.skipIf(kde is None, "k_diffusion_extra import failed")
class SolversConvergeOnPerfectDenoiser(unittest.TestCase):
    SAMPLERS = ['sample_res_multistep', 'sample_res_multistep_ancestral', 'sample_gradient_estimation',
                'sample_er_sde', 'sample_seeds_2', 'sample_seeds_3']

    def _run(self, name, sigmas):
        torch.manual_seed(0)
        target = torch.randn(1, 4, 8, 8)
        model = ToyDenoiser(target)
        x = torch.randn(1, 4, 8, 8) * sigmas[0]
        fn = getattr(kde, name)
        out = fn(model, x, sigmas, extra_args={}, disable=True)
        self.assertTrue(torch.isfinite(out).all(), f"{name} produced non-finite values")
        self.assertTrue(torch.allclose(out, target, atol=1e-4), f"{name} did not land on the target")
        self.assertGreaterEqual(model.calls, len(sigmas) - 1)

    def test_eps_style_schedule(self):
        for name in self.SAMPLERS:
            with self.subTest(sampler=name):
                self._run(name, karras_sigmas(12))

    def test_flow_schedule(self):
        for name in self.SAMPLERS:
            with self.subTest(sampler=name):
                self._run(name, flow_sigmas(12))

    def test_second_order_samplers_call_model_more_than_once_per_step(self):
        sigmas = karras_sigmas(6)
        model = ToyDenoiser(torch.zeros(1, 4, 8, 8))
        kde.sample_seeds_3(model, torch.randn(1, 4, 8, 8), sigmas, disable=True)
        self.assertEqual(model.calls, 3 * (len(sigmas) - 2) + 1)


@unittest.skipIf(kde is None, "k_diffusion_extra import failed")
class RectifiedFlowHelpers(unittest.TestCase):
    def test_half_log_snr_round_trip(self):
        s = torch.tensor([0.01, 0.3, 0.7, 0.99])
        for rf in (False, True):
            l = kde.sigma_to_half_log_snr(s, rf)
            self.assertTrue(torch.allclose(kde.half_log_snr_to_sigma(l, rf), s, atol=1e-6))

    def test_flow_first_sigma_is_nudged_below_one(self):
        model = ToyDenoiser(torch.zeros(1))
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        out = kde.offset_first_sigma_for_snr(sigmas, model, rf=True)
        self.assertLess(out[0].item(), 1.0)
        self.assertTrue(torch.isfinite(kde.sigma_to_half_log_snr(out[:-1], True)).all())   # the final 0 is never evaluated
        # untouched for non-flow models and for tensors that are already < 1
        self.assertTrue(torch.equal(kde.offset_first_sigma_for_snr(sigmas, model, rf=False), sigmas))

    def test_ancestral_step(self):
        down, up = kde.get_ancestral_step(torch.tensor(2.0), torch.tensor(1.0), eta=0.0)
        self.assertEqual((float(down), up), (1.0, 0.0))
        down, up = kde.get_ancestral_step(torch.tensor(2.0), torch.tensor(1.0), eta=1.0)
        self.assertAlmostEqual(float(down ** 2 + up ** 2), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
