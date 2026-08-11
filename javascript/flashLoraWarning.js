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

    var dialog = null;
    function showDialog(msg, onApply, onRunAsIs) {
        if (dialog) dialog.remove();
        dialog = document.createElement('div');
        dialog.id = 'flash-lora-warning';
        dialog.style.cssText =
            'position:fixed;top:44px;left:50%;transform:translateX(-50%);z-index:99998;' +
            'padding:12px 20px;border-radius:8px;font:600 13px/1.6 sans-serif;' +
            'background:#b7791f;color:#fff;box-shadow:0 2px 12px rgba(0,0,0,.45);' +
            'max-width:80%;text-align:center;';
        var text = document.createElement('div');
        text.textContent = msg;
        var row = document.createElement('div');
        row.style.cssText = 'margin-top:10px;display:flex;gap:10px;justify-content:center;';
        var mk = function (label, fn, primary) {
            var b = document.createElement('button');
            b.textContent = label;
            b.style.cssText = 'padding:6px 14px;border-radius:6px;border:none;cursor:pointer;font:600 13px sans-serif;' +
                (primary ? 'background:#fff;color:#7a4f12;' : 'background:rgba(255,255,255,.25);color:#fff;');
            b.addEventListener('click', function () { dialog.remove(); dialog = null; fn(); });
            return b;
        };
        row.appendChild(mk('⚡ Apply flash settings & run', onApply, true));
        row.appendChild(mk('Run as is', onRunAsIs, false));
        dialog.appendChild(text);
        dialog.appendChild(row);
        (document.body || document.documentElement).appendChild(dialog);
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
