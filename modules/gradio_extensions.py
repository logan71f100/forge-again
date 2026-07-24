import contextlib
import inspect
import re
import types
import warnings
from functools import wraps

import gradio as gr
import gradio.component_meta


from modules import scripts, ui_tempdir, patches


class GradioDeprecationWarning(DeprecationWarning):
    pass


def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """

    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(getattr(comp, 'elem_classes', None) or [])]

    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')


def IOComponent_init(self, *args, **kwargs):
    self.webui_tooltip = kwargs.pop('tooltip', None)

    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    scripts.script_callbacks.before_component_callback(self, **kwargs)

    res = original_IOComponent_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    scripts.script_callbacks.after_component_callback(self, **kwargs)

    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res


def Block_get_config(self, cls=None):
    config = original_Block_get_config(self, cls)

    webui_tooltip = getattr(self, 'webui_tooltip', None)
    if webui_tooltip:
        config["webui_tooltip"] = webui_tooltip

    config.pop('example_inputs', None)

    return config


def BlockContext_init(self, *args, **kwargs):
    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    scripts.script_callbacks.before_component_callback(self, **kwargs)

    res = original_BlockContext_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    scripts.script_callbacks.after_component_callback(self, **kwargs)

    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res


def Blocks_get_config_file(self, *args, **kwargs):
    # gradio 6 raises AttributeError building the config if any event's inputs or
    # outputs list contains a None (block_function.py: `block._id for block in
    # self.inputs`). gradio 4 tolerated it. Some Forge extensions wire events with
    # conditionally-None components (e.g. Replacer's dedicated page). Strip the
    # Nones to restore gradio-4 behavior. No-op for events that have no None.
    for fn in getattr(self, 'fns', {}).values():
        if getattr(fn, 'inputs', None):
            fn.inputs = [b for b in fn.inputs if b is not None]
        if getattr(fn, 'outputs', None):
            fn.outputs = [b for b in fn.outputs if b is not None]

    config = original_Blocks_get_config_file(self, *args, **kwargs)

    for comp_config in config["components"]:
        if "example_inputs" in comp_config:
            comp_config["example_inputs"] = {"serialized": []}

    return config


def Slider_preprocess(self, payload):
    # gradio 6's Slider.preprocess raises if the incoming value is outside
    # [minimum, maximum]; gradio 4 tolerated it. Forge/extensions ship sliders whose
    # default is an out-of-range sentinel -- notably ControlNet's `processor_res`,
    # `threshold_a`, `threshold_b` default to -1 ("auto") on min=64 sliders. Under
    # gradio 6 that made EVERY generate fail at input preprocessing. Restore gradio-4
    # behavior: pass the value through unchanged (rounded), no bounds check -- the
    # sentinel must survive so the backend can interpret it.
    from gradio.components.number import Number as _Number
    return _Number.round_to_precision(payload, self.precision)


def Block_get_component_class_id(cls):
    # gradio 6 hashes inspect.getfile(cls) into a stable component-class id, but
    # classes defined inside the webui's dynamically-loaded extension scripts (module
    # names like "sam.py" that are not registered in sys.modules) make inspect.getfile
    # raise TypeError ("... is a built-in class"), which killed the whole extension's
    # UI (e.g. segment-anything). gradio only catches OSError -- catch TypeError too
    # and fall back to the module name, mirroring gradio's own OSError fallback.
    import hashlib
    try:
        module_path = inspect.getfile(cls)
    except (OSError, TypeError):
        module_path = cls.__module__
    return hashlib.sha256(f"{cls.__name__}_{module_path}".encode()).hexdigest()


def Dropdown_preprocess(self, payload):
    # gradio 6 raises if a dropdown's incoming value is not in its choices; gradio 4
    # tolerated it. The webui stores values in ui-config.json/config.json that can
    # refer to models not currently installed/scanned (e.g. hires upscaler
    # "4x-UltraSharp"), which made every generate in that tab fail at preprocessing.
    # Restore gradio-4 tolerance; backends already handle unknown names gracefully.
    prev = self.allow_custom_value
    self.allow_custom_value = True
    try:
        return original_Dropdown_preprocess(self, payload)
    finally:
        self.allow_custom_value = prev


original_IOComponent_init = patches.patch(__name__, obj=gr.components.Component, field="__init__", replacement=IOComponent_init)
original_Block_get_config = patches.patch(__name__, obj=gr.blocks.Block, field="get_config", replacement=Block_get_config)
original_BlockContext_init = patches.patch(__name__, obj=gr.blocks.BlockContext, field="__init__", replacement=BlockContext_init)
original_Blocks_get_config_file = patches.patch(__name__, obj=gr.blocks.Blocks, field="get_config_file", replacement=Blocks_get_config_file)
original_Slider_preprocess = patches.patch(__name__, obj=gr.components.slider.Slider, field="preprocess", replacement=Slider_preprocess)
original_Dropdown_preprocess = patches.patch(__name__, obj=gr.components.dropdown.Dropdown, field="preprocess", replacement=Dropdown_preprocess)
original_Block_get_component_class_id = patches.patch(__name__, obj=gr.blocks.Block, field="get_component_class_id", replacement=classmethod(Block_get_component_class_id))


ui_tempdir.install_ui_tempdir_override()


def gradio_component_meta_create_or_modify_pyi(component_class, class_name, events):
    if hasattr(component_class, 'webui_do_not_create_gradio_pyi_thank_you'):
        return

    gradio_component_meta_create_or_modify_pyi_original(component_class, class_name, events)


# this prevents creation of .pyi files in webui dir
gradio_component_meta_create_or_modify_pyi_original = patches.patch(__file__, gradio.component_meta, 'create_or_modify_pyi', gradio_component_meta_create_or_modify_pyi)

# this function is broken and does not seem to do anything useful
gradio.component_meta.updateable = lambda x: x


_BARE_JS_NAME = re.compile(r'[A-Za-z_$][\w$]*\Z')


def fix_bare_js_name(js):
    """gradio 4 resolved js='functionName' to the global function and CALLED it with
    the event's input values (the webui wires nearly everything that way: 'submit',
    'submit_img2img', 'restart_reload', ...). gradio 6 evaluates the js string as an
    expression -- a bare identifier evaluates to the function object but never invokes
    it, so e.g. submit() (which starts the progress bar) silently did not run. Rewrite
    bare names into a calling arrow; late-binds via window so load order doesn't matter."""
    if isinstance(js, str):
        name = js.strip()
        if _BARE_JS_NAME.fullmatch(name):
            return f'(...args) => window.{name}(...args)'
    return js


class EventWrapper:
    def __init__(self, replaced_event):
        self.replaced_event = replaced_event
        self.has_trigger = getattr(replaced_event, 'has_trigger', None)
        self.event_name = getattr(replaced_event, 'event_name', None)
        self.callback = getattr(replaced_event, 'callback', None)
        self.real_self = getattr(replaced_event, '__self__', None)

    def __call__(self, *args, **kwargs):
        if '_js' in kwargs:
            kwargs['js'] = kwargs['_js']
            del kwargs['_js']
        if 'js' in kwargs:
            kwargs['js'] = fix_bare_js_name(kwargs['js'])
        # gradio 6 builds the event's Dependency config immediately on wiring and
        # crashes if inputs/outputs contain a None (block._id on None); gradio 4
        # tolerated it. Extension events (e.g. Replacer's video tab) wire
        # conditionally-None components. Strip Nones here so wiring succeeds.
        for _key in ('inputs', 'outputs'):
            _v = kwargs.get(_key)
            if isinstance(_v, (list, tuple, set)):
                kwargs[_key] = [_b for _b in _v if _b is not None]
            elif _v is None and _key in kwargs:
                kwargs[_key] = []
        return self.replaced_event(*args, **kwargs)

    @property
    def __self__(self):
        return self.real_self


def repair(grclass):
    if not getattr(grclass, 'EVENTS', None):
        return

    @wraps(grclass.__init__)
    def __repaired_init__(self, *args, tooltip=None, source=None, original=grclass.__init__, **kwargs):
        if source:
            kwargs["sources"] = [source]

        allowed_kwargs = inspect.signature(original).parameters
        fixed_kwargs = {}
        for k, v in kwargs.items():
            if k in allowed_kwargs:
                fixed_kwargs[k] = v
            else:
                warnings.warn(f"unexpected argument for {grclass.__name__}: {k}", GradioDeprecationWarning, stacklevel=2)

        original(self, *args, **fixed_kwargs)

        self.webui_tooltip = tooltip

        for event in self.EVENTS:
            replaced_event = getattr(self, str(event))
            fun = EventWrapper(replaced_event)
            setattr(self, str(event), fun)

    grclass.__init__ = __repaired_init__
    grclass.update = gr.update


for component in set(gr.components.__all__ + gr.layouts.__all__):
    repair(getattr(gr, component, None))


class Dependency(gr.events.Dependency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def then(*xargs, _js=None, **xkwargs):
            if _js:
                xkwargs['js'] = _js
            if 'js' in xkwargs:
                xkwargs['js'] = fix_bare_js_name(xkwargs['js'])

            return original_then(*xargs, **xkwargs)

        original_then = self.then
        self.then = then


gr.events.Dependency = Dependency

gr.Box = gr.Group


@contextlib.contextmanager
def force_interactive_components():
    """Make components built inside a `gr.render` body interactive.

    Gradio treats `interactive=None` as "infer it", and infers by checking
    whether the component is an input to some event. For a lazily-built tab the
    body is constructed inside `gr.render` and its events are wired afterwards,
    so that inference sees no inputs and renders EVERY control disabled --
    sliders, radios, dropdowns, the lot. The result looks like a normal UI that
    simply refuses to respond to clicks.

    Not hypothetical: this made the entire img2img tab non-interactive (resize
    mode, sampling, steps, width/height, batch count...), while txt2img -- built
    eagerly from the very same helper functions -- was fine.

    An explicit interactive=False is still honoured; only "infer" is overridden.
    """
    from gradio.components.base import Component

    original_init = Component.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            if getattr(self, "interactive", False) is None:
                self.interactive = True
        except Exception:
            pass

    Component.__init__ = patched_init
    try:
        yield
    finally:
        Component.__init__ = original_init

