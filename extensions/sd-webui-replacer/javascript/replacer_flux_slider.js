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

    // Quick-add LoRA chips: single delegated listener for ALL chips (they are
    // plain HTML buttons with data-lora, deliberately NOT gradio events — see
    // make_advanced_options.py). Delegation cannot double-fire across gradio 6
    // remounts, and exactly one chip matches any click.
    document.addEventListener('click', function (ev) {
        var chip = ev.target && ev.target.closest ? ev.target.closest('.replacer-lora-chip') : null;
        if (!chip) return;
        var name = chip.getAttribute('data-lora');
        var ta = gradioApp().querySelector('#replacer_positivePrompt textarea');
        if (!name || !ta) return;
        ta.value += ' <lora:' + name + ':1>';
        ta.dispatchEvent(new Event('input', { bubbles: true }));
    }, true);

    // Show only the chips for the mode that is CURRENTLY selected.
    //
    // The chip HTML is built once, when the server constructs its Blocks, so it
    // cannot be regenerated on a mode switch -- a page reload just re-serves
    // the same markup. A server that started in xl therefore offered xl LoRAs
    // for the rest of its life however many times the user switched to flux.
    // Every mode's chips are emitted with a data-mode instead, and the live
    // mode radio decides which are visible; chips at the Lora root carry an
    // empty data-mode and always show.
    function currentModeName() {
        var radio = gradioApp().querySelector('#forge_ui_preset');
        if (!radio) return null;
        var inputs = radio.getElementsByTagName('input');
        for (var i = 0; i < inputs.length; i++) {
            if (inputs[i].checked) return ['sd', 'xl', 'flux'][i] || null;
        }
        return null;
    }

    function applyLoraChipFilter() {
        var mode = currentModeName();
        var chips = gradioApp().querySelectorAll('.replacer-lora-chip');
        if (!chips.length) return;
        var shown = 0;
        chips.forEach(function (chip) {
            var m = chip.getAttribute('data-mode') || '';
            var visible = !mode || !m || m === mode;
            chip.style.display = visible ? '' : 'none';
            if (visible) shown++;
        });
        gradioApp().querySelectorAll('.replacer-lora-mode').forEach(function (el) {
            el.textContent = mode ? '(' + mode + ' mode, ' + shown + ')' : '';
        });
    }

    // re-apply on any mode change, and after gradio remounts the row
    document.addEventListener('change', function (ev) {
        if (ev.target && ev.target.closest && ev.target.closest('#forge_ui_preset')) {
            setTimeout(applyLoraChipFilter, 50);
        }
    }, true);
    if (typeof onAfterUiUpdate === 'function') onAfterUiUpdate(applyLoraChipFilter);
    else setInterval(applyLoraChipFilter, 2000);
})();
