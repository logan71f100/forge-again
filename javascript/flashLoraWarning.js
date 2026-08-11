// One-time-per-server-launch guard for flash/distill LoRA settings drift.
// Flash LoRAs (e.g. chroma-flash-heun) are trained for CFG 1.0 and 8-16 steps;
// CFG > 1 doubles every model evaluation, Heun doubles them again, and high
// step counts multiply the rest — the classic drift runs 4-6x slower than the
// intended flash configuration with no quality gain.
// On the FIRST offending Generate click after a server launch, the submit is
// held and a choice is offered: apply the recommended settings and run, or run
// as-is. Either choice dismisses the guard until the next server boot.
(function () {
    'use strict';

    var bootId = null;
    fetch('./internal/ping', { cache: 'no-store' })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (d) { bootId = (d && d.boot_id) || null; })
        .catch(function () { /* offline; sessionStorage fallback below */ });

    function warnedKey() { return 'flashLoraWarned:' + (bootId || 'session'); }
    function alreadyWarned() {
        try {
            return bootId ? !!localStorage.getItem(warnedKey()) : !!sessionStorage.getItem(warnedKey());
        } catch (e) { return false; }
    }
    function markWarned() {
        try {
            (bootId ? localStorage : sessionStorage).setItem(warnedKey(), '1');
        } catch (e) { /* storage unavailable */ }
    }

    function showDialog(msg, onApply, onRunAsIs) {
        forgeNotify.warn(msg, {
            id: 'flash-lora',
            timeout: 0,
            buttons: [
                { label: '⚡ Apply flash settings & run', primary: true, onClick: onApply },
                { label: 'Run as is', onClick: onRunAsIs },
            ],
        });
    }

    function num(sel) {
        var el = gradioApp().querySelector(sel);
        var v = el ? parseFloat(el.value) : NaN;
        return isNaN(v) ? null : v;
    }
    function setNum(sel, v) {
        gradioApp().querySelectorAll(sel).forEach(function (el) {
            el.value = v;
            el.dispatchEvent(new Event('input', { bubbles: true }));
        });
    }

    document.addEventListener('click', function (ev) {
        if (!ev.isTrusted) return;                                // our re-click passes through
        var btn = ev.target && ev.target.closest ? ev.target.closest('button[id$="_generate"]') : null;
        if (!btn) return;
        var tab = btn.id.slice(0, -'_generate'.length);
        if (tab !== 'txt2img' && tab !== 'img2img') return;
        if (alreadyWarned()) return;
        try {
            var promptEl = gradioApp().querySelector('#' + tab + '_prompt textarea');
            var prompt = (promptEl && promptEl.value) || '';
            if (!/<lora:[^>]*flash[^>]*:/i.test(prompt)) return;

            var cfg = num('#' + tab + '_cfg_scale input[type=number]');
            var steps = num('#' + tab + '_steps input[type=number]');
            var samplerEl = gradioApp().querySelector('#' + tab + '_sampling input');
            var sampler = (samplerEl && samplerEl.value) || '';
            if (cfg === null || steps === null) return;
            var cfgTooHigh = cfg > 1.0;
            var stepsTooHigh = steps > 16;
            if (!cfgTooHigh && !stepsTooHigh) return;

            // hold the submit: this is the one warning for this server launch
            ev.stopPropagation();
            ev.preventDefault();

            var evals = steps * (/heun/i.test(sampler) ? 2 : 1) * (cfgTooHigh ? 2 : 1);
            var ratio = Math.max(1, Math.round(evals / 16));
            var parts = [];
            if (cfgTooHigh) parts.push('CFG ' + cfg + ' (flash wants 1.0)');
            if (stepsTooHigh) parts.push(steps + ' steps (flash converges in 8–16)');
            var msg = '⚠ Flash LoRA with ' + parts.join(' and ') +
                (ratio > 1 ? ' — ~' + ratio + '× slower than the intended flash setup.' : '.');

            var regenerate = function () { btn.click(); };        // untrusted → passes the guard
            showDialog(msg,
                function () { setNum('#' + tab + '_cfg_scale input', 1.0); setNum('#' + tab + '_steps input', 16); markWarned(); setTimeout(regenerate, 250); },
                function () { markWarned(); regenerate(); });
        } catch (e) { /* never block generate on an internal error */ }
    }, true);
})();
