import gradio as gr
import torch

from modules import scripts, shared
from modules.ui_components import InputAccordion


# https://github.com/WeichenFan/CFG-Zero-star (as adopted by ComfyUI's CFGZeroStar node)
def optimized_scale(positive, negative):
    positive_flat = positive.reshape(positive.shape[0], -1)
    negative_flat = negative.reshape(negative.shape[0], -1)

    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star.reshape([positive.shape[0]] + [1] * (positive.ndim - 1))


class CFGZeroStarForForge(scripts.Script):
    """CFG-Zero*: rescales the unconditional branch by the projection of the
    conditional prediction onto it before applying the guidance scale, and
    optionally zeroes the very first step(s). Built for flow models (Flux with
    real CFG, Chroma, SD3) where plain CFG over-steers early; also usable on
    SDXL. Does nothing at CFG 1.
    """

    sorting_priority = 14

    def title(self):
        return "CFG-Zero* (guidance rescale for flow models)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title(), elem_id=self.elem_id("enabled")) as enabled:
            with gr.Row():
                zero_init_steps = gr.Slider(label='Zero-init steps', minimum=0, maximum=4, step=1, value=0,
                                            info='Return the input unchanged for this many first steps (the paper\'s "zero-init"; 1 helps flow models at low step counts).')
                scale_strength = gr.Slider(label='Strength', minimum=0.0, maximum=1.0, step=0.05, value=1.0,
                                           info='Blend between plain CFG (0) and the optimized scale (1).')

        self.infotext_fields = [
            (enabled, lambda d: d.get("cfg_zero_star", None) is not None),
            (zero_init_steps, "cfg_zero_star_zero_init"),
            (scale_strength, "cfg_zero_star"),
        ]

        return enabled, zero_init_steps, scale_strength

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, zero_init_steps, scale_strength = script_args

        if not enabled:
            return

        zero_init_steps = int(zero_init_steps)
        scale_strength = float(scale_strength)

        unet = p.sd_model.forge_objects.unet.clone()

        def cfg_zero_star(args):
            x = args['input']
            out = args['denoised']

            if zero_init_steps > 0 and shared.state.sampling_step < zero_init_steps:
                # zero prediction: the sampler sees d = (x - x) / sigma = 0
                return x

            uncond_p = args['uncond_denoised']
            cond_p = args['cond_denoised']
            if uncond_p is None or cond_p is None:
                return out

            guidance_scale = args['cond_scale']
            alpha = optimized_scale(x - cond_p, x - uncond_p)
            if scale_strength != 1.0:
                alpha = 1.0 + (alpha - 1.0) * scale_strength

            # out = uncond + s * (cond - uncond); rewrite with uncond -> uncond * alpha
            return out + uncond_p * (alpha - 1.0) + guidance_scale * uncond_p * (1.0 - alpha)

        unet.set_model_sampler_post_cfg_function(cfg_zero_star)
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            cfg_zero_star=scale_strength,
            cfg_zero_star_zero_init=zero_init_steps,
        ))
