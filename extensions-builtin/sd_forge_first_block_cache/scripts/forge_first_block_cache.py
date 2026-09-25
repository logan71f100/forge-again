import gradio as gr

from modules import scripts
from modules.ui_components import InputAccordion
from backend.misc.first_block_cache import cache as first_block_cache


class FirstBlockCacheForForge(scripts.Script):
    """First Block Cache (TeaCache / WaveSpeed-style step caching).

    Runs only the first transformer block (Flux, Chroma) or the first UNet
    stage (SD1.x, SD2.x, SDXL) and, when its output has barely changed since
    the last fully computed step, reuses that step's cached residual for the
    rest of the network. Typical: 1.3x-2x faster at the same step count, at a
    small cost in fine detail that grows with the threshold. On a card where
    the model does not fit in VRAM the skipped blocks are also skipped
    swap-ins, so the saving is larger than the compute share alone.
    """

    sorting_priority = 13

    def title(self):
        return "First Block Cache (faster sampling: SD, SDXL, Flux, Chroma)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title(), elem_id=self.elem_id("enabled")) as enabled:
            with gr.Row():
                threshold = gr.Slider(label='Residual difference threshold', minimum=0.0, maximum=1.0, step=0.01, value=0.12,
                                      info='Skip a step when the first block moved less than this (relative L1). 0.05 is conservative, 0.12 is the common Flux default, 0.2+ trades visible detail for speed.')
                max_consecutive = gr.Slider(label='Max consecutive skips', minimum=-1, maximum=10, step=1, value=-1,
                                            info='Force a full step after this many skipped ones. -1 = unlimited.')
            with gr.Row():
                start = gr.Slider(label='Start (fraction of the run)', minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                                  info='No caching before this point. The first steps set the composition; 0.1-0.2 keeps them exact.')
                end = gr.Slider(label='End (fraction of the run)', minimum=0.0, maximum=1.0, step=0.01, value=1.0)

        self.infotext_fields = [
            (enabled, lambda d: d.get("fbcache_threshold", None) is not None),
            (threshold, "fbcache_threshold"),
            (start, "fbcache_start"),
            (end, "fbcache_end"),
            (max_consecutive, "fbcache_max_consecutive"),
        ]

        return enabled, threshold, start, end, max_consecutive

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # called once per sampling pass (twice with hires fix): reconfiguring
        # also resets the cache, so a pass never reuses the other pass's residuals
        enabled, threshold, start, end, max_consecutive = script_args

        if not enabled:
            first_block_cache.disable()
            return

        first_block_cache.configure(True, threshold=threshold, start=start, end=end, max_consecutive=max_consecutive)

        p.extra_generation_params.update(dict(
            fbcache_threshold=threshold,
            fbcache_start=start,
            fbcache_end=end,
            fbcache_max_consecutive=max_consecutive,
        ))

    def postprocess(self, p, processed, *args):
        if first_block_cache.enabled:
            print(first_block_cache.summary())
        first_block_cache.disable()
