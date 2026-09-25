from modules import sd_samplers_kdiffusion, sd_samplers_common
from backend.modules import k_diffusion_extra


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name):
        self.sampler_name = sampler_name
        self.unet = sd_model.forge_objects.unet
        sampler_function = getattr(k_diffusion_extra, "sample_{}".format(sampler_name))
        super().__init__(sampler_function, sd_model, None)


def build_constructor(sampler_name):
    def constructor(m):
        return AlterSampler(m, sampler_name)

    return constructor


samplers_data_alter = [
    sd_samplers_common.SamplerData('DDPM', build_constructor(sampler_name='ddpm'), ['ddpm'], {}),
    # 2025 ComfyUI solvers (backend/modules/k_diffusion_extra.py). Same step
    # count, better quality per step; the SDE ones are stochastic.
    sd_samplers_common.SamplerData('Res Multistep', build_constructor(sampler_name='res_multistep'), ['res_multistep'], {}),
    sd_samplers_common.SamplerData('Res Multistep Ancestral', build_constructor(sampler_name='res_multistep_ancestral'), ['res_multistep_ancestral'], {"uses_ensd": True}),
    sd_samplers_common.SamplerData('Gradient Estimation', build_constructor(sampler_name='gradient_estimation'), ['gradient_estimation'], {}),
    sd_samplers_common.SamplerData('ER SDE', build_constructor(sampler_name='er_sde'), ['er_sde'], {"uses_ensd": True}),
    sd_samplers_common.SamplerData('SEEDS 2', build_constructor(sampler_name='seeds_2'), ['seeds_2'], {"uses_ensd": True, "second_order": True}),
    sd_samplers_common.SamplerData('SEEDS 3', build_constructor(sampler_name='seeds_3'), ['seeds_3'], {"uses_ensd": True, "second_order": True}),
]
