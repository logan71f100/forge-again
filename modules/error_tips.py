"""Plain-language tips for common runtime errors.

The UI shows raw exceptions ("mat1 and mat2 shapes cannot be multiplied ...")
that mean nothing to most users. This module maps recognizable error messages
to a short, actionable hint that is shown underneath the error in the UI.

To add a tip: append a (compiled regex, tip builder) pair to TIPS. The builder
receives the regex match and returns the tip text. Keep tips to one or two
sentences, in plain language, and only add entries whose cause is reasonably
unambiguous -- a wrong tip is worse than no tip.
"""

import re

# text-encoder / cross-attention context dims per architecture, used to guess
# which two model families got mixed when a matmul shape error appears.
_ARCH_BY_DIM = {
    768: "SD 1.5",
    1024: "SD 2.x",
    1280: "SDXL",
    2048: "SDXL",
    4096: "Flux",
}


def _arch(dim: int) -> str:
    return _ARCH_BY_DIM.get(dim, f"a model with {dim}-dim conditioning")


def _matmul_tip(m: re.Match) -> str:
    # "mat1 and mat2 shapes cannot be multiplied (154x2048 and 768x320)"
    a_in, b_in = int(m.group(2)), int(m.group(3))
    if (a_in, b_in) == (64, 384):
        # Flux Fill's img_in expects 384 channels (64 latent + masked-image
        # conditioning); plain txt2img only supplies the 64.
        return ("This checkpoint is a Flux Fill (inpainting) model -- it needs an input image and "
                "a mask, so it works from img2img inpaint or the Replacer tab, not plain txt2img. "
                "For txt2img, pick a regular (dev/schnell) Flux checkpoint.")
    left, right = _ARCH_BY_DIM.get(a_in), _ARCH_BY_DIM.get(b_in)
    if left and right and left != right:
        return (f"This usually means two different model families got mixed: something in this "
                f"generation is built for {right} but the checkpoint pipeline is {left}. "
                f"The most common cause is a ControlNet model that doesn't match the checkpoint "
                f"(e.g. an SD 1.5 ControlNet with an SDXL checkpoint, or vice versa). "
                f"Check your ControlNet / LoRA / embedding against the checkpoint's base model.")
    return ("This usually means a model in this generation (ControlNet, LoRA, embedding or VAE) "
            "was built for a different base architecture than the checkpoint (SD 1.5 vs SDXL vs Flux). "
            "Check that every extra model matches the checkpoint's family.")


def _oom_tip(m: re.Match) -> str:
    return ("The GPU ran out of memory. Lower the 'GPU Weights (MB)' slider at the top, reduce the "
            "image resolution or batch size, or switch 'Diffusion in Low Bits' to a smaller dtype.")


def _device_tip(m: re.Match) -> str:
    return ("Part of the model ended up on the CPU while the rest is on the GPU -- this usually "
            "follows an earlier out-of-memory error or an interrupted model load. Try clicking the "
            "checkpoint refresh button and generating again; if it persists, restart the server.")


def _none_image_tip(m: re.Match) -> str:
    return ("The input image never reached the backend. Make sure an image is actually loaded on "
            "the canvas (re-upload it if the preview looks empty), then try again.")


def _no_model_tip(m: re.Match) -> str:
    return ("No model is loaded -- the previous checkpoint load failed (the earlier error above or "
            "in the console says why). Fix that or select a different checkpoint, then try again.")


# Each entry: (pattern, tip builder, field selector or None).
# The field selector points at the UI control most likely responsible; the
# "{tab}" placeholder is resolved client-side to the active tab (txt2img/
# img2img) by javascript/errorHighlight.js, which outlines the control and
# scrolls it into view (issue #6). Use None when no single control is at fault.
TIPS = [
    (re.compile(r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)"), _matmul_tip,
     "#{tab}_controlnet"),
    (re.compile(r"(?:CUDA out of memory|OutOfMemoryError|not enough memory)", re.IGNORECASE), _oom_tip,
     None),
    (re.compile(r"Expected all tensors to be on the same device"), _device_tip,
     None),
    (re.compile(r"'NoneType' object has no attribute 'mode'"), _none_image_tip,
     "#{tab}_image"),
    (re.compile(r"'NoneType' object has no attribute 'sd_checkpoint_info'"), _no_model_tip,
     None),
]


def register_tip(pattern, tip, field=None):
    """Public registration for extensions and other modules.

    pattern: regex string or compiled pattern matched against the raw error.
    tip:     plain string, or a callable receiving the regex match.
    field:   optional CSS selector for the responsible control ('{tab}' is
             resolved client-side to the active tab by errorHighlight.js).
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    builder = tip if callable(tip) else (lambda m, _t=tip: _t)
    TIPS.append((pattern, builder, field))


def tip_and_field_for(error_message: str) -> tuple[str | None, str | None]:
    """Return (tip, field selector) for a raw error message; (None, None) if unknown."""
    if not error_message:
        return None, None
    for pattern, build, field in TIPS:
        m = pattern.search(error_message)
        if m:
            return build(m), field
    return None, None


def tip_for(error_message: str) -> str | None:
    """Return a plain-language tip for a raw error message, or None if unknown."""
    return tip_and_field_for(error_message)[0]
