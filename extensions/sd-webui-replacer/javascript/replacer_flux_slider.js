// Show the Replacer's "Distilled CFG Scale (Flux only)" slider only in flux
// mode, following the mode radio LIVE. The slider is built always-mounted
// (see make_advanced_options.py): it is created in on_before_ui, before the
// preset radio exists, so gradio-side event wiring is impossible — and a
// visible=False build would be unmounted under gradio 6, unreachable from JS.
// CSS-hiding a mounted component is the established webui pattern instead.
(function () {
    'use strict';

    function currentPreset() {
        var checked = gradioApp().querySelector('#forge_ui_preset input[type=radio]:checked');
        return checked ? checked.value : null;
    }

    function sync() {
        var slider = gradioApp().querySelector('#replacer_distilled_cfg_scale');
        if (!slider) return;              // dedicated page or tab not built yet
        var preset = currentPreset();
        if (preset === null) return;      // no radio on this page: leave as built
        slider.style.display = (preset === 'flux') ? '' : 'none';
    }

    // initial state + live follow: the radio fires change on user clicks, and
    // onAfterUiUpdate covers programmatic updates and late mounts (lazy tabs).
    document.addEventListener('change', function (ev) {
        if (ev.target && ev.target.matches && ev.target.matches('#forge_ui_preset input[type=radio]')) sync();
    }, true);
    if (typeof onAfterUiUpdate === 'function') onAfterUiUpdate(sync);
    else if (typeof onUiLoaded === 'function') onUiLoaded(sync);
})();
