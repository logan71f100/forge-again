import json, os, sys

# Deployable/isolated: config files live in THIS project dir (not a hardcoded
# F:\forge); model paths come from MODELS_DIR (env, default the shared models).
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("FORGE_MODELS_DIR", os.path.join(HERE, "models"))

FLUX_MODS = [
    os.path.join(MODELS_DIR, "VAE", "ae.safetensors"),
    os.path.join(MODELS_DIR, "text_encoder", "clip_l.safetensors"),
    os.path.join(MODELS_DIR, "text_encoder", "t5xxl_fp8_e4m3fn.safetensors"),
]
# model + main-tab (txt2img/img2img) optimized defaults
MODELS = {
    "sd":   {"ckpt": "epicrealism_pureEvolutionV5.safetensors",    "mods": [],        "W": 512,  "H": 512,  "cfg": 7.0, "dcfg": 3.5, "denoise": 0.5,  "hires_denoise": 0.4,  "steps": 28, "sampler": "DPM++ 2M",     "scheduler": "Karras", "infmem": 1025, "udtype": ""},
    "xl":   {"ckpt": "epicrealismXL_vxviiCrystalclear.safetensors", "mods": [],        "W": 1024, "H": 1024, "cfg": 6.0, "dcfg": 3.5, "denoise": 0.7,  "hires_denoise": 0.35, "steps": 30, "sampler": "DPM++ 2M SDE", "scheduler": "Karras", "infmem": 1025, "udtype": ""},
    "flux": {"ckpt": "fluxunchained-fill-full-Q6_K.gguf",           "mods": FLUX_MODS, "W": 1024, "H": 1024, "cfg": 1.0, "dcfg": 3.5, "denoise": 1.0,  "hires_denoise": 0.3,  "steps": 25, "sampler": "Euler",        "scheduler": "Simple", "infmem": 1025, "udtype": "Automatic (fp16 LoRA)"},
}
# Replacer optimized profile (read at UI build by the patched make_advanced_options.py / inpaint.py)
REPLACER = {
    "sd":   {"REPLACER_DEF_SAMPLER": "DPM++ 2M",     "REPLACER_DEF_SCHEDULER": "Karras", "REPLACER_DEF_WIDTH": 512,  "REPLACER_DEF_HEIGHT": 512,  "REPLACER_DEF_STEPS": 28, "REPLACER_DEF_CFG": 7.0, "REPLACER_DEF_DENOISE": 0.5, "REPLACER_DEF_MASK_BLUR": 4, "REPLACER_DEF_PADDING": 32, "REPLACER_FLUX_GUIDANCE": 3.5},
    "xl":   {"REPLACER_DEF_SAMPLER": "DPM++ 2M SDE", "REPLACER_DEF_SCHEDULER": "Karras", "REPLACER_DEF_WIDTH": 1024, "REPLACER_DEF_HEIGHT": 1024, "REPLACER_DEF_STEPS": 30, "REPLACER_DEF_CFG": 6.0, "REPLACER_DEF_DENOISE": 0.7, "REPLACER_DEF_MASK_BLUR": 0, "REPLACER_DEF_PADDING": 48, "REPLACER_FLUX_GUIDANCE": 3.5},
    "flux": {"REPLACER_DEF_SAMPLER": "Euler",        "REPLACER_DEF_SCHEDULER": "Simple", "REPLACER_DEF_WIDTH": 1024, "REPLACER_DEF_HEIGHT": 1024, "REPLACER_DEF_STEPS": 25, "REPLACER_DEF_CFG": 1.0, "REPLACER_DEF_DENOISE": 1.0, "REPLACER_DEF_MASK_BLUR": 6, "REPLACER_DEF_PADDING": 48, "REPLACER_FLUX_GUIDANCE": 30},
}
COMMON = {"REPLACER_DEF_MASK_EXPAND": 15, "REPLACER_DEF_BOX_THRESHOLD": 0.35, "REPLACER_DEF_FILL": "original"}

# Replacer quick-chip examples (newline-separated; clicking a chip REPLACES the field) -- per mode
EXAMPLES = {
    "sd": {
        "det": "shirt\nhair\nsunglasses",
        "pos": ("RAW photo of a woman in a red summer dress, detailed fabric texture, realistic, highly detailed, 8k"
                "\nphoto of a man in a black leather jacket, detailed skin, realistic, masterpiece, best quality, ultra detailed"),
        "neg": "cartoon, painting, illustration, (worst quality, low quality, normal quality:2), bad anatomy, deformed, extra limbs, fused fingers, watermark, text, blurry, lowres",
    },
    "xl": {
        "det": "shirt\nhair\nsunglasses",
        "pos": ("score_9, score_8_up, score_7_up, score_6_up, photo of a woman in an elegant evening gown, detailed fabric, realistic, 8k uhd, professional photography"
                "\nphoto of a man in a tailored navy suit, detailed fabric texture, realistic, professional photography, 8k, sharp focus"
                "\n<lora:RealisticSkinTexture-XL:0.6>, "),
        "neg": "score_6, score_5, score_4, cartoon, 3d, render, anime, painting, worst quality, low quality, bad anatomy, deformed, extra limbs, watermark, text, blurry",
    },
    "flux": {
        "det": "shirt\nhair\nsunglasses",
        "pos": ("photo of a woman wearing a blue denim jacket, detailed fabric texture, natural lighting, "
                "<lora:RealisticSkinTexture-Flux:0.6>"
                "\n<lora:RealisticSkinTexture-Flux:0.6>, "),
        "neg": "blurry, low quality, deformed",  # flux runs at CFG 1 -> negatives are ignored anyway
    },
}

def _hires_upscaler():
    """Best available hires-fix upscaler: the user's 4x-UltraSharp if installed,
    else R-ESRGAN 4x+ (bundled/auto-downloaded), never Latent."""
    try:
        if any(f.lower().startswith("4x-ultrasharp") for f in os.listdir(os.path.join(MODELS_DIR, "ESRGAN"))):
            return "4x-UltraSharp"
    except OSError:
        pass
    return "R-ESRGAN 4x+"


def write_mode_files(mode, here=None):
    """Write this project's config/ui/profile files for `mode`.

    Returns (m, ex): the model-defaults dict and the Replacer examples dict, so
    an in-process caller (the live mode switch in main_entry) can also push the
    model-affecting values straight into shared.opts without a subprocess.
    """
    mode = (mode or "xl").lower()
    if mode not in ("sd", "xl", "flux"):
        mode = "xl"
    here = here or HERE

    m = MODELS[mode]
    ex = dict(EXAMPLES[mode])

    # 1) config.json -> checkpoint + modules + memory + per-mode Replacer example chips
    cp = os.path.join(here, "config.json")
    # fresh clone: no config.json yet -- start from empty and create it below
    try:
        c = json.load(open(cp, encoding="utf-8"))
    except FileNotFoundError:
        c = {}

    # per-mode chip overrides from user settings (Settings > Replacer, stored in the
    # untracked config.json). Empty/missing = the SFW built-in defaults above.
    for short, name in (("det", "detection"), ("pos", "positive"), ("neg", "negative")):
        ov = c.get(f"replacer_mode_chips_{mode}_{name}", "")
        if isinstance(ov, str) and ov.strip():
            ex[short] = ov
    c["sd_model_checkpoint"] = m["ckpt"]
    c["forge_additional_modules"] = m["mods"]
    c["forge_preset"] = mode
    c["forge_inference_memory"] = m["infmem"]   # GPU Weights = total - this (1025 -> ~10239)
    c["forge_async_loading"] = "Queue"
    c["forge_pin_shared_memory"] = "CPU"
    c["forge_unet_storage_dtype"] = m["udtype"]
    c["replacer_detection_prompt_examples"] = ex["det"]
    c["replacer_positive_prompt_examples"] = ex["pos"]
    c["replacer_negative_prompt_examples"] = ex["neg"]
    json.dump(c, open(cp, "w", encoding="utf-8"), indent=4)

    # saved-mode marker (launcher/radio restart into same mode + start.bat reads it for --ckpt-dir)
    open(os.path.join(here, "current_mode.txt"), "w").write(mode)

    # 2) ui-config.json -> main txt2img/img2img defaults; strip Replacer keys (profile file wins)
    up = os.path.join(here, "ui-config.json")
    if os.path.exists(up):
        u = json.load(open(up, encoding="utf-8"))
        for tab in ("txt2img", "img2img"):
            u[f"{tab}/Width/value"] = m["W"]
            u[f"{tab}/Height/value"] = m["H"]
            u[f"{tab}/CFG Scale/value"] = m["cfg"]
            u[f"{tab}/Distilled CFG Scale/value"] = m["dcfg"]
            u[f"{tab}/Denoising strength/value"] = m["denoise"]
            u[f"{tab}/Sampling steps/value"] = m["steps"]
            u[f"{tab}/Sampling method/value"] = m["sampler"]
            u[f"{tab}/Schedule type/value"] = m["scheduler"]
        # hires-fix defaults: in txt2img the "Denoising strength" slider IS the
        # hires-fix denoise. Latent upscaling needs high denoise, and high
        # denoise at 2x redraws anatomy (elongated bodies on XL) -- a GAN
        # upscaler at low denoise keeps the composition and just adds detail.
        u["txt2img/Denoising strength/value"] = m["hires_denoise"]
        u["txt2img/Upscaler/value"] = _hires_upscaler()
        u = {k: v for k, v in u.items() if not k.startswith("Replacer/")}
        json.dump(u, open(up, "w", encoding="utf-8"), indent=4)

    # 3) mode_profile.json -> Replacer defaults (+ current mode marker)
    prof = dict(REPLACER[mode]); prof.update(COMMON); prof["_mode"] = mode
    json.dump(prof, open(os.path.join(here, "mode_profile.json"), "w", encoding="utf-8"), indent=4)
    return m, ex


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else "xl"
    _m, _ = write_mode_files(_mode)
    _mode = (_mode or "xl").lower()
    if _mode not in ("sd", "xl", "flux"):
        _mode = "xl"
    print("mode=" + _mode + " ckpt=" + _m["ckpt"] + " sampler=" + _m["sampler"] + "/" + _m["scheduler"])
