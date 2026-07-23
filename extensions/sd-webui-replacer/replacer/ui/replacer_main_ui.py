import gradio as gr
from modules import infotext_utils
from replacer.extensions import replacer_extensions
from replacer.ui.tools_ui import AttrDict
from replacer.ui.replacer_tab_ui import getTabUI
from replacer.ui.video.replacer_video_tab_ui import getVideoTabUI


try:
    from modules.ui_components import ResizeHandleRow
except:
    ResizeHandleRow = gr.Row



class ReplacerMainUI:
    def __init__(self, isDedicatedPage: bool):
        self.replacerTabUI = None
        self.replacerVideoTabUI = None
        self.components = AttrDict()
        self.init_tab(isDedicatedPage)

    def init_tab(self, isDedicatedPage: bool):
        comp = AttrDict()
        self.replacerTabUI = getTabUI(comp, isDedicatedPage)
        self.replacerVideoTabUI = getVideoTabUI(comp, isDedicatedPage)

        self.components = comp


    def getReplacerTabUI(self):
        return self.replacerTabUI

    def getReplacerVideoTabUI(self):
        return self.replacerVideoTabUI


replacerMainUI: ReplacerMainUI = None
replacerMainUI_dedicated: ReplacerMainUI = None

registered_param_bindings_main_ui = []

def initMainUI(*args):
    global replacerMainUI, replacerMainUI_dedicated, registered_param_bindings_main_ui
    lenBefore = len(infotext_utils.registered_param_bindings)
    try:
        replacer_extensions.initAllScripts()
        replacerMainUI = ReplacerMainUI(isDedicatedPage=False)
        replacerMainUI_dedicated = ReplacerMainUI(isDedicatedPage=True)
    finally:
        replacer_extensions.restoreTemporaryChangedThings()

    registered_param_bindings_main_ui = infotext_utils.registered_param_bindings[lenBefore:]


def reinitMainUIAfterUICreated():
    replacer_extensions.reinitAllScriptsAfterUICreated()

    # LAZY IMG2IMG: reinit copies args_from/args_to from the real img2img script
    # instances, but the img2img tab body (where scripts.py assigns those ranges) is
    # built lazily on first open -- so at this point they can still be None, which
    # crashed generation ('NoneType' - 'NoneType'). Replacer swaps in its OWN script
    # copies at generate time (applyScripts builds a fresh p.script_args), so the
    # ranges only need to be self-consistent: lay them out back-to-back sized by
    # Replacer's own CN / soft-inpaint UI component counts.
    comp = replacerMainUI.components if replacerMainUI else None
    if comp is not None:
        pos = 0
        cn = replacer_extensions.controlnet.SCRIPT
        if cn is not None:
            if cn.args_from is None or cn.args_to is None:
                n = len(getattr(comp, 'cn_inputs', None) or [])
                cn.args_from, cn.args_to = pos, pos + n
            pos = cn.args_to
        si = replacer_extensions.soft_inpainting.SCRIPT
        if si is not None and (si.args_from is None or si.args_to is None):
            n = len(getattr(comp, 'soft_inpaint_inputs', None) or [])
            si.args_from, si.args_to = pos, pos + n

    infotext_utils.registered_param_bindings += registered_param_bindings_main_ui

