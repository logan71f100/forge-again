import os
import torch
import gradio as gr

from gradio.context import Context
from modules import shared_items, shared, ui_common, sd_models, processing, infotext_utils, paths
from backend import memory_management, stream
from backend.args import dynamic_args


total_vram = int(memory_management.total_vram)

ui_forge_preset: gr.Radio = None

ui_checkpoint: gr.Dropdown = None
ui_vae: gr.Dropdown = None
ui_clip_skip: gr.Slider = None

ui_forge_unet_storage_dtype_options: gr.Radio = None
ui_forge_async_loading: gr.Radio = None
ui_forge_pin_shared_memory: gr.Radio = None
ui_forge_inference_memory: gr.Slider = None



forge_unet_storage_dtype_options = {
    'Automatic': (None, False),
    'Automatic (fp16 LoRA)': (None, True),
    'bnb-nf4': ('nf4', False),
    'bnb-nf4 (fp16 LoRA)': ('nf4', True),
    'float8-e4m3fn': (torch.float8_e4m3fn, False),
    'float8-e4m3fn (fp16 LoRA)': (torch.float8_e4m3fn, True),
    'bnb-fp4': ('fp4', False),
    'bnb-fp4 (fp16 LoRA)': ('fp4', True),
    'float8-e5m2': (torch.float8_e5m2, False),
    'float8-e5m2 (fp16 LoRA)': (torch.float8_e5m2, True),
}

module_list = {}


def bind_to_opts(comp, k, save=False, callback=None):
    def on_change(v):
        shared.opts.set(k, v)
        if save:
            shared.opts.save(shared.config_filename)
        if callback is not None:
            callback()
        return

    comp.change(on_change, inputs=[comp], queue=False, show_progress=False)
    return


def make_checkpoint_manager_ui():
    global ui_checkpoint, ui_vae, ui_clip_skip, ui_forge_unet_storage_dtype_options, ui_forge_async_loading, ui_forge_pin_shared_memory, ui_forge_inference_memory, ui_forge_preset

    if shared.opts.sd_model_checkpoint in [None, 'None', 'none', '']:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set('sd_model_checkpoint', next(iter(sd_models.checkpoints_list.values())).name)

    ui_forge_preset = gr.Radio(label="UI", value=lambda: shared.opts.forge_preset, choices=['sd', 'xl', 'flux'], elem_id="forge_ui_preset")

    ckpt_list, vae_list = refresh_models()

    ui_checkpoint = gr.Dropdown(
        value=lambda: shared.opts.sd_model_checkpoint,
        label="Checkpoint",
        elem_classes=['model_selection'],
        choices=ckpt_list,
        # tolerate a value that is transiently not in `choices`: during an in-place
        # mode switch the choices list is repopulated for the new mode while the
        # dropdown's .change fires; checkpoint_change resolves aliases itself, so
        # strict membership validation would otherwise raise a preprocess Error.
        allow_custom_value=True,
    )

    ui_vae = gr.Dropdown(
        value=lambda: [os.path.basename(x) for x in shared.opts.forge_additional_modules],
        multiselect=True,
        label="VAE / Text Encoder",
        render=False,
        choices=vae_list,
        # flux's modules live under the shared FORGE_MODELS_DIR, not this project's
        # models_path, so their basenames are not in the scanned `choices`; allow them
        # so an in-place flux switch can display the active modules as tokens.
        allow_custom_value=True,
    )

    def gr_refresh_models():
        a, b = refresh_models()
        return gr.update(choices=a), gr.update(choices=b)

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id=f"forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(
        fn=gr_refresh_models,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )
    Context.root_block.load(
        fn=gr_refresh_models,
        inputs=[],
        outputs=[ui_checkpoint, ui_vae],
        show_progress=False,
        queue=False
    )

    ui_vae.render()

    ui_forge_unet_storage_dtype_options = gr.Dropdown(label="Diffusion in Low Bits", value=lambda: shared.opts.forge_unet_storage_dtype, choices=list(forge_unet_storage_dtype_options.keys()))
    bind_to_opts(ui_forge_unet_storage_dtype_options, 'forge_unet_storage_dtype', save=True, callback=refresh_model_loading_parameters)

    ui_forge_async_loading = gr.Radio(label="Swap Method", value=lambda: shared.opts.forge_async_loading, choices=['Queue', 'Async'])
    ui_forge_pin_shared_memory = gr.Radio(label="Swap Location", value=lambda: shared.opts.forge_pin_shared_memory, choices=['CPU', 'Shared'])
    ui_forge_inference_memory = gr.Slider(label="GPU Weights (MB)", value=lambda: total_vram - shared.opts.forge_inference_memory, minimum=0, maximum=int(memory_management.total_vram), step=1)

    mem_comps = [ui_forge_inference_memory, ui_forge_async_loading, ui_forge_pin_shared_memory]

    ui_forge_inference_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_async_loading.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)
    ui_forge_pin_shared_memory.change(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    Context.root_block.load(ui_refresh_memory_management_settings, inputs=mem_comps, queue=False, show_progress=False)

    ui_clip_skip = gr.Slider(label="Clip skip", value=lambda: shared.opts.CLIP_stop_at_last_layers, **{"minimum": 1, "maximum": 12, "step": 1})
    bind_to_opts(ui_clip_skip, 'CLIP_stop_at_last_layers', save=True)

    ui_checkpoint.change(checkpoint_change, inputs=[ui_checkpoint], show_progress=False)
    ui_vae.change(modules_change, inputs=[ui_vae], queue=False, show_progress=False)

    return


def find_files_with_extensions(base_path, extensions):
    found_files = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                found_files[file] = full_path
    return found_files


def refresh_models():
    global module_list

    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

    file_extensions = ['ckpt', 'pt', 'bin', 'safetensors', 'gguf']

    module_list.clear()
    
    module_paths = [
        os.path.abspath(os.path.join(paths.models_path, "VAE")),
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
    ]

    if isinstance(shared.cmd_opts.vae_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.vae_dir))
    if isinstance(shared.cmd_opts.text_encoder_dir, str):
        module_paths.append(os.path.abspath(shared.cmd_opts.text_encoder_dir))

    for vae_path in module_paths:
        vae_files = find_files_with_extensions(vae_path, file_extensions)
        module_list.update(vae_files)

    return ckpt_list, module_list.keys()


def ui_refresh_memory_management_settings(model_memory, async_loading, pin_shared_memory):
    """ Passes precalculated 'model_memory' from "GPU Weights" UI slider (skip redundant calculation) """
    refresh_memory_management_settings(
        async_loading=async_loading,
        pin_shared_memory=pin_shared_memory,
        model_memory=model_memory  # Use model_memory directly from UI slider value
    )

def refresh_memory_management_settings(async_loading=None, inference_memory=None, pin_shared_memory=None, model_memory=None):
    # Fallback to defaults if values are not passed
    async_loading = async_loading if async_loading is not None else shared.opts.forge_async_loading
    inference_memory = inference_memory if inference_memory is not None else shared.opts.forge_inference_memory
    pin_shared_memory = pin_shared_memory if pin_shared_memory is not None else shared.opts.forge_pin_shared_memory

    # If model_memory is provided, calculate inference memory accordingly, otherwise use inference_memory directly
    if model_memory is None:
        model_memory = total_vram - inference_memory
    else:
        inference_memory = total_vram - model_memory

    shared.opts.set('forge_async_loading', async_loading)
    shared.opts.set('forge_inference_memory', inference_memory)
    shared.opts.set('forge_pin_shared_memory', pin_shared_memory)

    stream.stream_activated = async_loading == 'Async'
    memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Convert MB to bytes
    memory_management.PIN_SHARED_MEMORY = pin_shared_memory == 'Shared'

    log_dict = dict(
        stream=stream.should_use_stream(),
        inference_memory=memory_management.minimum_inference_memory() / (1024 * 1024),
        pin_shared_memory=memory_management.PIN_SHARED_MEMORY
    )

    print(f'Environment vars changed: {log_dict}')

    if inference_memory < min(512, total_vram * 0.05):
        print('------------------')
        print(f'[Low VRAM Warning] You just set Forge to use 100% GPU memory ({model_memory:.2f} MB) to load model weights.')
        print('[Low VRAM Warning] This means you will have 0% GPU memory (0.00 MB) to do matrix computation. Computations may fallback to CPU or go Out of Memory.')
        print('[Low VRAM Warning] In many cases, image generation will be 10x slower.')
        print("[Low VRAM Warning] To solve the problem, you can set the 'GPU Weights' (on the top of page) to a lower value.")
        print("[Low VRAM Warning] If you cannot find 'GPU Weights', switch to the 'xl' or 'flux' mode in the 'UI' area on the left-top corner of the webpage.")
        print('[Low VRAM Warning] Make sure that you know what you are testing.')
        print('------------------')
    else:
        compute_percentage = (inference_memory / total_vram) * 100.0
        print(f'[GPU Setting] You will use {(100 - compute_percentage):.2f}% GPU memory ({model_memory:.2f} MB) to load weights, and use {compute_percentage:.2f}% GPU memory ({inference_memory:.2f} MB) to do matrix computation.')

    processing.need_global_unload = True
    return


def refresh_model_loading_parameters():
    from modules.sd_models import select_checkpoint, model_data

    checkpoint_info = select_checkpoint()

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    dynamic_args['online_lora'] = lora_fp16

    model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        unet_storage_dtype=unet_storage_dtype
    )

    print(f'Model selected: {model_data.forge_loading_parameters}')
    print(f'Using online LoRAs in FP16: {lora_fp16}')
    processing.need_global_unload = True

    return


def checkpoint_change(ckpt_name:str, save=True, refresh=True):
    """ checkpoint name can be a number of valid aliases. Returns True if checkpoint changed. """
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(shared.opts.data.get('sd_model_checkpoint', ''))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set('sd_model_checkpoint', ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def modules_change(module_values:list, save=True, refresh=True) -> bool:
    """ module values may be provided as file paths, or just the module names. Returns True if modules changed. """
    modules = []
    for v in module_values:
        module_name = os.path.basename(v) # If the input is a filepath, extract the file name
        if module_name in module_list:
            modules.append(module_list[module_name])
    
    # skip further processing if value unchanged
    if sorted(modules) == sorted(shared.opts.data.get('forge_additional_modules', [])):
        return False

    shared.opts.set('forge_additional_modules', modules)

    if save:
        shared.opts.save(shared.config_filename)
    if refresh:
        refresh_model_loading_parameters()
    return True


def get_a1111_ui_component(tab, label):
    # LAZY IMG2IMG: when a tab body is deferred (built via gr.render on first select), its
    # add_paste_fields(tab, ...) has not run yet at forge_main_entry() time, so the tab key is
    # absent from paste_fields. Return None so the preset output_targets filtering below drops it
    # (the deferred body rebuilds fresh on preset change via full page reload). Eager tabs are
    # unaffected -- their key is present.
    tab_fields = infotext_utils.paste_fields.get(tab)
    if tab_fields is None:
        return None
    fields = tab_fields['fields']
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_width = get_a1111_ui_component('txt2img', 'Size-1')
    ui_txt2img_height = get_a1111_ui_component('txt2img', 'Size-2')
    ui_txt2img_cfg = get_a1111_ui_component('txt2img', 'CFG scale')
    ui_txt2img_distilled_cfg = get_a1111_ui_component('txt2img', 'Distilled CFG Scale')
    ui_txt2img_sampler = get_a1111_ui_component('txt2img', 'sampler_name')
    ui_txt2img_scheduler = get_a1111_ui_component('txt2img', 'scheduler')

    ui_img2img_width = get_a1111_ui_component('img2img', 'Size-1')
    ui_img2img_height = get_a1111_ui_component('img2img', 'Size-2')
    ui_img2img_cfg = get_a1111_ui_component('img2img', 'CFG scale')
    ui_img2img_distilled_cfg = get_a1111_ui_component('img2img', 'Distilled CFG Scale')
    ui_img2img_sampler = get_a1111_ui_component('img2img', 'sampler_name')
    ui_img2img_scheduler = get_a1111_ui_component('img2img', 'scheduler')

    ui_txt2img_hr_cfg = get_a1111_ui_component('txt2img', 'Hires CFG Scale')
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component('txt2img', 'Hires Distilled CFG Scale')

    output_targets = [
        ui_vae,
        ui_clip_skip,
        ui_forge_unet_storage_dtype_options,
        ui_forge_async_loading,
        ui_forge_pin_shared_memory,
        ui_forge_inference_memory,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_hr_cfg,
        ui_txt2img_hr_distilled_cfg,
    ]

    # LAZY IMG2IMG: when the img2img tab body is deferred (built via gr.render only on first tab
    # select), its 6 components above (ui_img2img_width/height/cfg/distilled_cfg/sampler/scheduler)
    # do not exist yet at UI-build time, so get_a1111_ui_component('img2img', ...) returns None.
    # The preset page-LOAD initializer (.load below) must only target components that exist right
    # now, so drop the None entries and filter on_preset_change's aligned return the same way.
    # When img2img is eager (all 20 present) this is a no-op. A preset CHANGE reloads the whole
    # page (see the js below), so a deferred img2img rebuilds fresh in the new mode and never needs
    # to be a live target here.
    _valid_target_idx = [i for i, c in enumerate(output_targets) if c is not None]
    output_targets = [output_targets[i] for i in _valid_target_idx]

    def _on_preset_change_filtered(preset=None):
        full = on_preset_change(preset)
        if full is None:
            return None
        return [full[i] for i in _valid_target_idx]

    # IN-PLACE mode switch (no restart, no page reload). Two chained handlers on the
    # same trigger, run in registration order:
    #   1. preset_apply_and_refresh_checkpoint: writes files + pushes opts + repoints
    #      the checkpoint scan dir + hot-swaps the model, and returns the checkpoint
    #      dropdown update (per-mode choices + this mode's default).
    #   2. _on_preset_change_filtered: returns the main txt2img/img2img control updates,
    #      keyed off the forge_preset that handler #1 just set.
    ui_forge_preset.change(preset_apply_and_refresh_checkpoint, inputs=[ui_forge_preset], outputs=[ui_checkpoint, ui_vae], queue=False, show_progress=False)
    ui_forge_preset.change(_on_preset_change_filtered, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)
    ui_forge_preset.change(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    Context.root_block.load(_on_preset_change_filtered, inputs=None, outputs=output_targets, queue=False, show_progress=False)

    refresh_model_loading_parameters()
    return


_set_mode_module = None


def _load_set_mode():
    """Import THIS project's set_mode.py as a module (cached) so mode switching can
    reuse its file-writing + defaults tables in-process (no subprocess/restart)."""
    global _set_mode_module
    if _set_mode_module is None:
        import importlib.util
        from modules.paths_internal import script_path
        _path = os.path.join(script_path, "set_mode.py")
        _spec = importlib.util.spec_from_file_location("forge_set_mode", _path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _set_mode_module = _mod
    return _set_mode_module


def _apply_mode_inplace(mode):
    """Switch mode LIVE, without restarting the process.

    - writes this project's config/ui/profile files (persists for the next launch)
    - pushes the model-affecting values into the live shared.opts
    - repoints the checkpoint scan dir to the mode's subfolder (keeps the dropdown
      per-mode) and rescans
    - hot-swaps flux modules + checkpoint (both flag need_global_unload, so the new
      model loads on the next generation -- the only unavoidable cost)

    Returns (ckpt_list, default_ckpt) for updating the checkpoint dropdown, or
    (None, None) if the switch could not be applied.
    """
    try:
        sm = _load_set_mode()
        m, ex = sm.write_mode_files(mode)   # also persists config.json/current_mode.txt/etc.

        # live opts (the config.json write above does not touch the running session)
        shared.opts.set('forge_preset', mode)
        shared.opts.set('forge_inference_memory', m['infmem'])
        shared.opts.set('forge_async_loading', 'Queue')
        shared.opts.set('forge_pin_shared_memory', 'CPU')
        shared.opts.set('forge_unet_storage_dtype', m['udtype'])
        shared.opts.set('replacer_detection_prompt_examples', ex['det'])
        shared.opts.set('replacer_positive_prompt_examples', ex['pos'])
        shared.opts.set('replacer_negative_prompt_examples', ex['neg'])

        # repoint the checkpoint scan dir to this mode's subfolder so the dropdown
        # shows only this mode's checkpoints (list_models reads cmd_opts.ckpt_dir
        # fresh on every scan), then rescan.
        ckpt_list = None
        cur = shared.cmd_opts.ckpt_dir
        if cur:
            base = os.path.dirname(cur.rstrip('/\\'))
            shared.cmd_opts.ckpt_dir = os.path.join(base, mode)
        ckpt_list, _ = refresh_models()   # rescans new dir; populates sd_models.checkpoints_list

        # hot-swap. Set the additional modules DIRECTLY to the mode's full paths
        # (flux's ae/clip_l/t5xxl live under the shared FORGE_MODELS_DIR, NOT under
        # this project's models_path -- so modules_change(), which filters through
        # module_list scanned from models_path, would silently drop them to []). This
        # mirrors what set_mode.py writes to config.json for a restart-based switch.
        shared.opts.set('forge_additional_modules', list(m['mods']))
        checkpoint_change(m['ckpt'], save=False, refresh=False)
        refresh_model_loading_parameters()   # sets model_data.forge_loading_parameters + need_global_unload

        shared.opts.save(shared.config_filename)

        # the dropdown's value must be one of its choices exactly. The tiles may be
        # bare filenames (short-name mode) or "filename [hash]" -- match m['ckpt']
        # against the freshly-scanned tiles so the format lines up either way.
        value = m['ckpt']
        if ckpt_list:
            value = next((c for c in ckpt_list if c == m['ckpt'] or c.startswith(m['ckpt'])), m['ckpt'])
        vae_value = [os.path.basename(x) for x in m['mods']]
        return ckpt_list, value, vae_value
    except Exception as _e:
        import traceback
        print("[mode-switch] in-place switch failed:", _e)
        traceback.print_exc()
        return None, None, None


def preset_apply_and_refresh_checkpoint(preset=None):
    """ui_forge_preset.change handler #1: apply the mode in-place and return the
    checkpoint + VAE dropdown updates. Runs before the main-component handler below."""
    if preset is None:
        return gr.update(), gr.update()
    ckpt_list, default_ckpt, vae_value = _apply_mode_inplace(preset)
    if ckpt_list is None:
        return gr.update(), gr.update()
    return gr.update(choices=ckpt_list, value=default_ckpt), gr.update(value=vae_value)


def on_preset_change(preset=None):
    # NOTE: the actual mode switch (opts + files + model hot-swap) is done in-place by
    # preset_apply_and_refresh_checkpoint (wired first). This function only returns the
    # UI gr.update()s for the main txt2img/img2img controls, keyed off the now-current
    # shared.opts.forge_preset. No restart, no page reload.
    #
    # The legacy 'all' preset was removed (it mapped to no mode profile and just
    # un-hid every advanced control). Normalize it -- or any unknown value left in
    # an old config.json -- to xl so the page-load initializer still works.
    current = shared.opts.forge_preset
    if current not in ('sd', 'xl', 'flux'):
        current = 'xl'
        shared.opts.set('forge_preset', current)

    if current == 'sd':
        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=True, value=1),                                           # ui_clip_skip
            gr.update(visible=False, value='Automatic'),                                # ui_forge_unet_storage_dtype_options
            gr.update(visible=False, value='Queue'),                                    # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=False, value=total_vram - 1024),                          # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "sd_t2i_width", 512)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "sd_i2i_width", 512)),                 # ui_img2img_width
            gr.update(value=getattr(shared.opts, "sd_t2i_height", 640)),                # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "sd_i2i_height", 512)),                # ui_img2img_height
            gr.update(value=getattr(shared.opts, "sd_t2i_cfg", 7)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "sd_i2i_cfg", 7)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "sd_t2i_sampler", 'Euler a')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "sd_i2i_sampler", 'Euler a')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "sd_t2i_scheduler", 'Automatic')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "sd_i2i_scheduler", 'Automatic')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "sd_t2i_hr_cfg", 7.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
        ]

    if current == 'xl':
        model_mem = getattr(shared.opts, "xl_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=False, value='Queue'),                                    # ui_forge_async_loading
            gr.update(visible=False, value='CPU'),                                      # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "xl_t2i_width", 896)),                 # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "xl_i2i_width", 1024)),                # ui_img2img_width
            gr.update(value=getattr(shared.opts, "xl_t2i_height", 1152)),               # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "xl_i2i_height", 1024)),               # ui_img2img_height
            gr.update(value=getattr(shared.opts, "xl_t2i_cfg", 5)),                     # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "xl_i2i_cfg", 5)),                     # ui_img2img_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_distilled_cfg
            gr.update(visible=False, value=3.5),                                        # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "xl_t2i_sampler", 'Euler a')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "xl_i2i_sampler", 'Euler a')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "xl_t2i_scheduler", 'Automatic')),     # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "xl_i2i_scheduler", 'Automatic')),     # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "xl_t2i_hr_cfg", 5.0)),  # ui_txt2img_hr_cfg
            gr.update(visible=False, value=3.5),                                        # ui_txt2img_hr_distilled_cfg
        ]

    if current == 'flux':
        model_mem = getattr(shared.opts, "flux_GPU_MB", total_vram - 1024)
        if model_mem < 0 or model_mem > total_vram:
            model_mem = total_vram - 1024
        return [
            gr.update(visible=True),                                                    # ui_vae
            gr.update(visible=False, value=1),                                          # ui_clip_skip
            gr.update(visible=True, value='Automatic'),                                 # ui_forge_unet_storage_dtype_options
            gr.update(visible=True, value='Queue'),                                     # ui_forge_async_loading
            gr.update(visible=True, value='CPU'),                                       # ui_forge_pin_shared_memory
            gr.update(visible=True, value=model_mem),                                   # ui_forge_inference_memory
            gr.update(value=getattr(shared.opts, "flux_t2i_width", 896)),               # ui_txt2img_width
            gr.update(value=getattr(shared.opts, "flux_i2i_width", 1024)),              # ui_img2img_width
            gr.update(value=getattr(shared.opts, "flux_t2i_height", 1152)),             # ui_txt2img_height
            gr.update(value=getattr(shared.opts, "flux_i2i_height", 1024)),             # ui_img2img_height
            gr.update(value=getattr(shared.opts, "flux_t2i_cfg", 1)),                   # ui_txt2img_cfg
            gr.update(value=getattr(shared.opts, "flux_i2i_cfg", 1)),                   # ui_img2img_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_d_cfg", 3.5)), # ui_txt2img_distilled_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_i2i_d_cfg", 3.5)), # ui_img2img_distilled_cfg
            gr.update(value=getattr(shared.opts, "flux_t2i_sampler", 'Euler')),         # ui_txt2img_sampler
            gr.update(value=getattr(shared.opts, "flux_i2i_sampler", 'Euler')),         # ui_img2img_sampler
            gr.update(value=getattr(shared.opts, "flux_t2i_scheduler", 'Simple')),      # ui_txt2img_scheduler
            gr.update(value=getattr(shared.opts, "flux_i2i_scheduler", 'Simple')),      # ui_img2img_scheduler
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_cfg", 1.0)),    # ui_txt2img_hr_cfg
            gr.update(visible=True, value=getattr(shared.opts, "flux_t2i_hr_d_cfg", 3.5)),  # ui_txt2img_hr_distilled_cfg
        ]

    # unreachable: `current` is normalized to sd/xl/flux above (legacy 'all' removed)
    return None

shared.options_templates.update(shared.options_section(('ui_sd', "UI defaults 'sd'", "ui"), {
    "sd_t2i_width":  shared.OptionInfo(512,  "txt2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "sd_t2i_height": shared.OptionInfo(640,  "txt2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "sd_t2i_cfg":    shared.OptionInfo(7,    "txt2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "sd_t2i_hr_cfg": shared.OptionInfo(7,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "sd_i2i_width":  shared.OptionInfo(512,  "img2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "sd_i2i_height": shared.OptionInfo(512,  "img2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "sd_i2i_cfg":    shared.OptionInfo(7,    "img2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
}))
shared.options_templates.update(shared.options_section(('ui_xl', "UI defaults 'xl'", "ui"), {
    "xl_t2i_width":  shared.OptionInfo(896,  "txt2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_t2i_height": shared.OptionInfo(1152, "txt2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_t2i_cfg":    shared.OptionInfo(5,    "txt2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_t2i_hr_cfg": shared.OptionInfo(5,    "txt2img HiRes CFG",  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_i2i_width":  shared.OptionInfo(1024, "img2img width",      gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_i2i_height": shared.OptionInfo(1024, "img2img height",     gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "xl_i2i_cfg":    shared.OptionInfo(5,    "img2img CFG",        gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "xl_GPU_MB":     shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)", gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
}))
shared.options_templates.update(shared.options_section(('ui_flux', "UI defaults 'flux'", "ui"), {
    "flux_t2i_width":    shared.OptionInfo(896,  "txt2img width",                gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "flux_t2i_height":   shared.OptionInfo(1152, "txt2img height",               gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "flux_t2i_cfg":      shared.OptionInfo(1,    "txt2img CFG",                  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "flux_t2i_hr_cfg":   shared.OptionInfo(1,    "txt2img HiRes CFG",            gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "flux_t2i_d_cfg":    shared.OptionInfo(3.5,  "txt2img Distilled CFG",        gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "flux_t2i_hr_d_cfg": shared.OptionInfo(3.5,  "txt2img Distilled HiRes CFG",  gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "flux_i2i_width":    shared.OptionInfo(1024, "img2img width",                gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "flux_i2i_height":   shared.OptionInfo(1024, "img2img height",               gr.Slider, {"minimum": 64, "maximum": 2048, "step": 8}),
    "flux_i2i_cfg":      shared.OptionInfo(1,    "img2img CFG",                  gr.Slider, {"minimum": 1,  "maximum": 30,   "step": 0.1}),
    "flux_i2i_d_cfg":    shared.OptionInfo(3.5,  "img2img Distilled CFG",        gr.Slider, {"minimum": 0,  "maximum": 30,   "step": 0.1}),
    "flux_GPU_MB":       shared.OptionInfo(total_vram - 1024, "GPU Weights (MB)",gr.Slider, {"minimum": 0,  "maximum": total_vram,   "step": 1}),
}))
