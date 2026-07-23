import os as _os
import json as _json
import gradio as gr
from modules import shared, sd_samplers, ui, ui_settings, errors
from modules.paths_internal import script_path as _webui_root

_MODE_PROFILE_PATH = _os.path.join(_webui_root, "mode_profile.json")
_CURRENT_MODE_PATH = _os.path.join(_webui_root, "current_mode.txt")


def _mp():
    try:
        return _json.load(open(_MODE_PROFILE_PATH, encoding="utf-8"))
    except Exception:
        return {}


def _envdef(key, default):
    v = _mp().get(key, _os.environ.get(key, default))
    try:
        return type(default)(v)
    except Exception:
        return default


def _envchoice(key, default, choices):
    v = _mp().get(key, _os.environ.get(key))
    return v if v in choices else default

from modules.ui_components import ToolButton
from modules.ui_common import refresh_symbol
from replacer.options import (EXT_NAME_LOWER, doNotShowUnloadButton, getAvoidancePromptExamples,
    getAvoidancePromptExamplesNumber, getMaskColorStr
)
from replacer.extensions import replacer_extensions
from replacer.ui.tools_ui import IS_WEBUI_1_9, AttrDict, IS_WEBUI_1_5, OverrideCustomScriptSource



def makeAdvancedOptions(comp: AttrDict, isDedicatedPage: bool):
    with gr.Accordion("Advanced options", open=False, elem_id='replacer_advanced_options'):
        with gr.Tabs(elem_id="replacer_advanced_options_tabs"):
            with gr.Tab('Generation'):
                with gr.Row():
                    sampler_names = sd_samplers.visible_sampler_names()
                    defaultSampler = "DPM++ 2M SDE" if IS_WEBUI_1_9 else "DPM++ 2M SDE Karras"
                    comp.sampler = gr.Dropdown(
                        label='Sampling method',
                        elem_id="replacer_sampler",
                        choices=sampler_names,
                        value=_envchoice('REPLACER_DEF_SAMPLER', defaultSampler, sampler_names)
                    )

                    if IS_WEBUI_1_9:
                        from modules import sd_schedulers
                        scheduler_names = [x.label for x in sd_schedulers.schedulers]
                        comp.scheduler = gr.Dropdown(
                            label='Schedule type',
                            elem_id=f"replacer_scheduler",
                            choices=scheduler_names,
                            value=_envchoice('REPLACER_DEF_SCHEDULER', scheduler_names[0], scheduler_names))
                    else:
                        comp.scheduler = gr.Textbox("", visible=False)

                    comp.steps = gr.Slider(
                        label='Steps',
                        value=_envdef('REPLACER_DEF_STEPS', 20),
                        step=1,
                        minimum=1,
                        maximum=150,
                        elem_id="replacer_steps"
                    )

                with gr.Row():
                    comp.cfg_scale = gr.Slider(label='CFG Scale',
                        value=_envdef('REPLACER_DEF_CFG', 5.5), elem_id="replacer_cfg_scale",
                        minimum=1.0, maximum=30.0, step=0.5)

                    comp.correct_aspect_ratio = gr.Checkbox(label='Correct aspect ratio',
                        value=True, elem_id="replacer_fix_steps")

                with gr.Row():
                    comp.distilled_cfg_scale = gr.Slider(label='Distilled CFG Scale (Flux only)', visible=(_mp().get("_mode", "flux") == "flux"),
                        info='Flux guidance strength. Higher (35-50) = follows your prompt and mask edges harder with more contrast, but can oversaturate or look baked-in. Lower (10-20) = softer, more natural blend that may drift from the prompt. Default 30. No effect in SD/SDXL modes.',
                        value=_envdef('REPLACER_FLUX_GUIDANCE', 30.0), elem_id="replacer_distilled_cfg_scale",
                        minimum=1.0, maximum=60.0, step=0.5)
                    def _save_dcfg(v):
                        try:
                            import json as __j
                            _p = _MODE_PROFILE_PATH
                            _d = __j.load(open(_p, encoding="utf-8"))
                            _d["REPLACER_FLUX_GUIDANCE"] = float(v)
                            __j.dump(_d, open(_p, "w", encoding="utf-8"), indent=4)
                        except Exception:
                            pass
                        import os as __o
                        __o.environ["REPLACER_FLUX_GUIDANCE"] = str(v)
                    comp.distilled_cfg_scale.release(_save_dcfg, inputs=[comp.distilled_cfg_scale])
                import os as _qlo
                _qmode = "xl"
                try:
                    _qmode = open(_CURRENT_MODE_PATH).read().strip()
                except Exception:
                    pass
                _qld = shared.cmd_opts.lora_dir if getattr(shared.cmd_opts, 'lora_dir', None) else _qlo.path.join(_webui_root, "models", "Lora")
                _qloras = []
                for _qd in (_qld, _qlo.path.join(_qld, _qmode)):
                    if _qlo.path.isdir(_qd):
                        _qloras += [_x[:-12] for _x in _qlo.listdir(_qd) if _x.endswith(".safetensors")]
                _qloras = sorted(set(_qloras))
                with gr.Row():
                    gr.HTML("<b>Quick-add LoRA</b> (" + _qmode + " mode) - click to append to the positive prompt")
                with gr.Row(elem_id="replacer_quick_loras"):
                    for _qln in _qloras:
                        _qjs = "() => {var t=document.querySelector('#replacer_positivePrompt textarea'); if(t){t.value += ' <lora:" + _qln + ":1>'; t.dispatchEvent(new Event('input', {bubbles:true}));}}"
                        gr.Button("+ " + _qln, size="sm").click(None, js=_qjs)

                with gr.Row():
                    with gr.Column(elem_id="replacer_width_height_column", elem_classes="replacer-generation-size"):
                        comp.width = gr.Slider(label='width',
                            value=_envdef('REPLACER_DEF_WIDTH', 512), elem_id="replacer_width",
                            minimum=64, maximum=2048, step=8)
                        comp.height = gr.Slider(label='height',
                            value=_envdef('REPLACER_DEF_HEIGHT', 512), elem_id="replacer_height",
                            minimum=64, maximum=2048, step=8)
                    with gr.Column(elem_id="replacer_batch_count_size_column", elem_classes="replacer-batch-count-size"):
                        comp.batch_count = gr.Slider(label='batch count',
                            value=1, elem_id="replacer_batch_count",
                            minimum=1, maximum=12, step=1)
                        comp.batch_size = gr.Slider(label='batch size',
                            value=1, elem_id="replacer_batch_size",
                            minimum=1, maximum=12, step=1)

                with gr.Row():
                    comp.upscaler_for_img2img = gr.Dropdown(
                        value="None",
                        choices=[x.name for x in shared.sd_upscalers],
                        label="Upscaler for img2Img",
                        elem_id="replacer_upscaler_for_img2img",
                    )

                    comp.seed = gr.Number(label='Seed', value=-1, elem_id="replacer_seed", min_width=100, precision=0)

                    comp.random_seed = ToolButton(
                        ui.random_symbol,
                        elem_id="replacer_random_seed",
                        label='Random seed'
                    )
                    comp.reuse_seed = ToolButton(
                        ui.reuse_symbol,
                        elem_id="replacer_reuse_seed",
                        label='Reuse seed'
                    )

                with gr.Row():
                    with OverrideCustomScriptSource("arplusplus"):
                        if replacer_extensions.arplusplus.SCRIPT:
                            try:
                                replacer_extensions.arplusplus.SCRIPT.i2i_w = comp.width
                                replacer_extensions.arplusplus.SCRIPT.i2i_h = comp.height
                                replacer_extensions.arplusplus.SCRIPT.ui(True)
                            except Exception as e:
                                errors.report(f"Cannot add arplusplus ui {e}", exc_info=True)
                                replacer_extensions.arplusplus.SCRIPT = None


            with gr.Tab('Detection'):
                with gr.Row():
                    comp.box_threshold = gr.Slider(label='Box Threshold',
                        value=_envdef('REPLACER_DEF_BOX_THRESHOLD', 0.3), elem_id="replacer_box_threshold",
                        minimum=0.0, maximum=1.0, step=0.01)
                    comp.mask_expand = gr.Slider(label='Mask Expand',
                        value=_envdef('REPLACER_DEF_MASK_EXPAND', 35), elem_id="replacer_mask_expand",
                        minimum=-50, maximum=100, step=1)

                with gr.Row():
                    if not doNotShowUnloadButton():
                        comp.unload = gr.Button(
                            value="Unload detection models",
                            elem_id="replacer_unload_detection_models")

                    comp.max_resolution_on_detection = gr.Slider(
                        label='Max resolution on detection',
                        value=1280,
                        step=1,
                        minimum=64,
                        maximum=2560,
                        elem_id="replacer_max_resolution_on_detection"
                    )

                with gr.Row():
                    from scripts.sam import sam_model_list, refresh_sam_models
                    from scripts.dino import dino_model_list

                    comp.sam_model_name = gr.Dropdown(label="SAM Model", choices=sam_model_list,
                        value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                    comp.sam_refresh_models = ToolButton(value=refresh_symbol)
                    comp.sam_refresh_models.click(refresh_sam_models, comp.sam_model_name, comp.sam_model_name)

                    comp.dino_model_name = gr.Dropdown(label="GroundingDINO Model", choices=dino_model_list, value=dino_model_list[0])

                with gr.Row():
                    comp.mask_num = gr.Radio(label='Mask num',
                        choices=['Random', '1', '2', '3'],
                        value='Random', type="value", elem_id="replacer_mask_num")
                    comp.reuse_mask_num = ToolButton(
                        ui.reuse_symbol,
                        elem_id="replacer_reuse_mask_num",
                        label='Reuse mask num'
                    )

                with gr.Row():
                    comp.extra_includes = ui_settings.create_setting_component(EXT_NAME_LOWER + "_default_extra_includes")
                    comp.extra_includes.label = 'Extra include in gallery'

            with gr.Tab('Inpainting'):
                with gr.Row():
                    comp.mask_blur = gr.Slider(label='Mask Blur',
                        value=_envdef('REPLACER_DEF_MASK_BLUR', 4), elem_id="replacer_mask_blur",
                        minimum=0, maximum=100, step=1)
                    comp.inpaint_padding = gr.Slider(label='Padding',
                        value=_envdef('REPLACER_DEF_PADDING', 90), elem_id="replacer_inpaint_padding",
                        minimum=0, maximum=1000, step=1)

                with gr.Row():
                    comp.denoise = gr.Slider(label='Denoising',
                        value=_envdef('REPLACER_DEF_DENOISE', 1.0), elem_id="replacer_denoise",
                        minimum=0.0, maximum=1.0, step=0.01)
                    comp.forbid_too_small_crop_region = gr.Checkbox(label='Forbid too small crop region',
                        value=True, visible=hasattr(shared.opts, 'forbid_too_small_crop_region'),
                        elem_id="replacer_forbid_too_small_crop_region")

                with gr.Row():
                    comp.inpainting_fill = gr.Radio(label='Masked content',
                        choices=['fill', 'original', 'latent noise', 'latent nothing'],
                        value=_envchoice('REPLACER_DEF_FILL', 'fill', ['fill', 'original', 'latent noise', 'latent nothing']),
                        type="index", elem_id="replacer_inpainting_fill")

                with gr.Row():
                    comp.inpainting_mask_invert = gr.Radio(
                        label='Mask mode',
                        choices=['Inpaint masked', 'Inpaint not masked'],
                        value='Inpaint masked',
                        type="index",
                        elem_id="replacer_mask_mode")

                comp.soft_inpaint_inputs = []

                with OverrideCustomScriptSource("soft_inpainting"):
                    if replacer_extensions.soft_inpainting.SCRIPT:
                        try:
                            with gr.Row():
                                replacer_extensions.soft_inpainting.needWatchSoftInpaintUI = True
                                comp.soft_inpaint_inputs = list(replacer_extensions.soft_inpainting.SCRIPT.ui(True))
                                replacer_extensions.soft_inpainting.needWatchSoftInpaintUI = False
                                from modules.ui_components import InputAccordion
                                new_soft_inpaint_accordion = InputAccordion(False, label="Soft inpainting", elem_id="replacer_soft_inpainting_enabled")
                                new_soft_inpaint_accordion.accordion.children = comp.soft_inpaint_inputs[0].accordion.children
                                for child in new_soft_inpaint_accordion.accordion.children:
                                    child.parent = new_soft_inpaint_accordion.accordion
                                comp.soft_inpaint_inputs[0].accordion.visible = False
                                comp.soft_inpaint_inputs[0] = new_soft_inpaint_accordion
                        except Exception as e:
                            errors.report(f"Cannot add soft inpaint accordion {e}", exc_info=True)
                            replacer_extensions.soft_inpainting.SCRIPT = None


            with gr.Tab('Others'):
                with gr.Row():
                    comp.variation_seed = gr.Number(label='Variation seed', value=-1, elem_id="replacer_variation_seed", min_width=100, precision=0)

                    comp.random_variation_seed = ToolButton(
                        ui.random_symbol,
                        elem_id="replacer_random_variation_seed",
                        label='Random variation seed'
                    )
                    comp.reuse_variation_seed = ToolButton(
                        ui.reuse_symbol,
                        elem_id="replacer_reuse_variation_seed",
                        label='Reuse variation seed'
                    )

                    comp.variation_strength = gr.Slider(label='Variation strength',
                        value=0.0, elem_id="replacer_variation_strength",
                        minimum=0.0, maximum=1.0, step=0.01)


                with gr.Row():
                    comp.rotation_fix = gr.Radio(label='Rotation fix',
                        choices=['-', '⟲', '⟳', '🗘'],
                        value='-', type="value", elem_id="replacer_rotation_fix")
                    comp.integer_only_masked = gr.Checkbox(label='Correct padding to have integer upscaling to better image fit',
                        value=False, visible=hasattr(shared.opts, 'integer_only_masked'), elem_id="replacer_integer_only_masked")

                with gr.Row():
                    comp.fix_steps = gr.Checkbox(label='Do exactly the amount of steps the slider specifies',
                        value=False, elem_id="replacer_fix_steps")

                with gr.Row():
                    if not isDedicatedPage:
                        comp.sd_model_checkpoint = ui_settings.create_setting_component('sd_model_checkpoint')
                        comp.override_sd_model = gr.Checkbox(label='Override stable diffusion model',
                            value=False, elem_id="replacer_override_sd_model")

                    comp.clip_skip = ui_settings.create_setting_component('CLIP_stop_at_last_layers')


            with gr.Tab('Avoidance'):
                with gr.Row():
                    comp.avoidancePrompt = gr.Textbox(label="Avoidance prompt",
                                        show_label=True,
                                        lines=1,
                                        elem_classes=["avoidancePrompt"],
                                        placeholder=None,
                                        elem_id="replacer_avoidancePrompt")

                    if getAvoidancePromptExamplesNumber() > 0:
                        gr.Examples(
                            examples=getAvoidancePromptExamples(),
                            inputs=comp.avoidancePrompt,
                            label="",
                            elem_id="replacer_avoidancePrompt_examples",
                            examples_per_page=getAvoidancePromptExamplesNumber(),
                        )

                with gr.Row():
                    comp.avoid_mask_create_canvas = gr.Button('Create canvas', elem_id='replacer_avoid_mask_create_canvas')
                    comp.avoid_mask_need_limit = gr.Checkbox(value=True, label='Limit avoidance mask canvas resolution on creating')
                    comp.avoid_mask_mode = gr.CheckboxGroup(['Draw mask', 'Upload mask'], value=['Draw mask'], label="Canvas mask source")
                with gr.Row():
                    comp.avoidance_mask = gr.Image(
                        label="Avoidance mask",
                        show_label=False,
                        elem_id="replacer_avoidance_mask",
                        source="upload",
                        interactive=True,
                        type="pil",
                        tool="sketch",
                        image_mode="RGB",
                        brush_color=getMaskColorStr(),
                        height=getattr(shared.opts, "img2img_editor_height", 720)
                    )
                with gr.Row():
                    gr.Markdown("*If image isn't fit, use fullscreen hotkey or resize column*")
                with gr.Row():
                    comp.avoid_mask_brush_color = gr.ColorPicker(
                        getMaskColorStr(), label='Brush color',
                        info='visual only, use when brush color is hard to see'
                    )
                    if IS_WEBUI_1_5:
                        comp.avoid_mask_brush_color.visible = False

            with gr.Tab('Custom mask'):
                with gr.Row():
                    comp.only_custom_mask = gr.Checkbox(label='Do not use detection prompt if use custom mask',
                        value=True, elem_id="replacer_only_custom_mask")

                with gr.Row():
                    comp.create_canvas_custom_mask = gr.Button('Create canvas', elem_id='replacer_create_canvas_custom_mask')
                    comp.custom_mask_need_limit = gr.Checkbox(value=True, label='Limit custom mask canvas resolution on creating')
                    comp.custom_mask_mode = gr.CheckboxGroup(['Draw mask', 'Upload mask'], value=['Draw mask'], label="Canvas mask source")
                with gr.Row():
                    comp.custom_mask = gr.Image(
                        label="Custom mask",
                        show_label=False,
                        elem_id="replacer_custom_mask",
                        source="upload",
                        interactive=True,
                        type="pil",
                        tool="sketch",
                        image_mode="RGB",
                        brush_color=getMaskColorStr(),
                        height=getattr(shared.opts, "img2img_editor_height", 720)
                    )
                with gr.Row():
                    gr.Markdown("*If image isn't fit, use fullscreen hotkey or resize column*")
                with gr.Row():
                    comp.custom_mask_brush_color = gr.ColorPicker(
                        getMaskColorStr(), label='Brush color',
                        info='visual only, use when brush color is hard to see')
                    if IS_WEBUI_1_5:
                        comp.custom_mask_brush_color.visible = False
                    comp.do_not_use_mask = gr.Checkbox(value=False, label="Do not use mask", info="Ignore any masks, equivalent of img2img")

            with OverrideCustomScriptSource("inpaint_diff"):
                with (gr.Tab('Inpaint Diff') if replacer_extensions.inpaint_difference.Globals
                        else gr.Group()) as comp.inpaint_diff_tab:
                    with gr.Row():
                        comp.inpaint_diff_create = gr.Button('Create', elem_id='replacer_inpaint_diff_create')
                        comp.use_inpaint_diff = gr.Checkbox(label='Use inpaint difference',
                            value=True, elem_id="replacer_use_inpaint_diff")
                    with gr.Row():
                        comp.non_altered_image_for_inpaint_diff = gr.Image(
                            label="Non altered image",
                            show_label=True,
                            elem_id="replacer_non_altered_image_for_inpaint_diff",
                            source="upload",
                            type="pil",
                            image_mode="RGBA",
                        )
                        comp.inpaint_diff_mask_view = gr.Image(label="Difference mask",
                            interactive=True, type="pil",
                            elem_id="replacer_inpaint_diff_mask_view")
                    with gr.Row():
                        comp.inpaint_diff_threshold = gr.Slider(label='Difference threshold',
                            maximum=1, step=0.01, value=1, elem_id='inpaint_difference_difference_threshold')
                        comp.inpaint_diff_mask_expand = gr.Slider(label='Mask dilation',
                            value=5, elem_id="replacer_inpaint_diff_mask_expand",
                            minimum=0, maximum=100, step=1)
                        comp.inpaint_diff_mask_erosion = gr.Slider(label='Mask erosion',
                            maximum=100, step=1, value=0, elem_id='inpaint_difference_mask_erosion')
                    with gr.Row():
                        comp.inpaint_diff_contours_only = gr.Checkbox(label='Contours only',
                            value=False, elem_id='inpaint_difference_contours_only')
                if not replacer_extensions.inpaint_difference.Globals:
                    comp.inpaint_diff_tab.visible = False
                    comp.inpaint_diff_tab.render = False

