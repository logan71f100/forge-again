"""Kontext reference editing (img2img).

FLUX.1 Kontext checkpoints edit an image from a text instruction: the source
image rides along as extra reference tokens (see backend/nn/flux.py, the
kontext_latents path) while the target is generated from noise. A Kontext
state dict is indistinguishable from flux-dev's, so activation is explicit --
this checkbox -- rather than autodetected (ComfyUI makes the same choice).

Usage: img2img, load a Kontext checkpoint, drop the source image on the
canvas, write the edit instruction as the prompt, enable this, and set
denoising to 1.0 (the reference guides the result; the init pixels are fully
re-generated -- lower denoise mixes both mechanisms and muddies the edit).
"""

import gradio as gr

from modules import scripts


class KontextEditScript(scripts.Script):
    sorting_priority = 15

    def title(self):
        return "Kontext reference edit"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        with gr.Accordion(label=self.title(), open=False):
            enabled = gr.Checkbox(
                label="Use the init image as a Kontext reference (needs a Kontext checkpoint; set denoising to 1.0)",
                value=False)
        return [enabled]

    def process_before_every_sampling(self, p, enabled=False, **kwargs):
        unet = getattr(getattr(p.sd_model, 'forge_objects', None), 'unet', None)
        if unet is None:
            return
        to = unet.model_options.setdefault('transformer_options', {})
        # Always clear first: model_options persist on the loaded model across
        # generations, and a stale reference from a previous run must never
        # leak into a plain generation.
        to.pop('kontext_latents', None)

        if not enabled:
            return
        init = getattr(p, 'init_latent', None)
        if init is None:
            print('[Kontext] enabled but there is no init image; ignoring.')
            return
        to['kontext_latents'] = [init]
        p.extra_generation_params['Kontext reference'] = True
        print(f'[Kontext] reference latents attached: {tuple(init.shape)}')
