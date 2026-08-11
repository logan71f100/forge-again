// Unified notifications for the webui frontend: one stacked, consistently
// styled system for every banner/toast/choice-dialog, replacing the ad-hoc
// per-feature banners (connection watchdog, flash-LoRA guard, ...).
//
//   forgeNotify.info('msg', opts)      neutral, auto-hides
//   forgeNotify.success('msg', opts)   green, auto-hides fast
//   forgeNotify.warn('msg', opts)      amber, auto-hides slow
//   forgeNotify.error('msg', opts)     red, sticky by default
//
// opts (all optional):
//   id: 'name'        same-id notifications REPLACE each other (state banners)
//   timeout: ms       0 = sticky (manual close / buttons only)
//   buttons: [{label, primary, onClick}]  choice dialogs; clicking closes first
//   once: 'boot' | 'session'  with id: show at most once per server launch
//                     (boot_id-keyed, survives page reloads) or browser tab
//   closable: true    force a ✕ even on auto-hiding toasts
//
// Returns {close(), el} or null when suppressed by `once`.
// forgeNotify.dismiss(id) closes a notification by id from anywhere.
(function () {
    'use strict';

    var LEVELS = {
        info:    { bg: '#2d3748', timeout: 6000 },
        success: { bg: '#1f9d55', timeout: 3000 },
        warn:    { bg: '#b7791f', timeout: 10000 },
        error:   { bg: '#c0392b', timeout: 0 },
    };

    var bootId = null;
    try {
        fetch('./internal/ping', { cache: 'no-store' })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (d) { bootId = (d && d.boot_id) || null; })
            .catch(function () { /* offline; session-keyed fallback */ });
    } catch (e) { /* fetch unavailable */ }

    var stack = null;
    var byId = {};

    function ensureStack() {
        if (stack && stack.isConnected) return stack;
        stack = document.getElementById('forge-notify-stack');
        if (!stack) {
            stack = document.createElement('div');
            stack.id = 'forge-notify-stack';
            stack.style.cssText =
                'position:fixed;top:0;left:50%;transform:translateX(-50%);z-index:99999;' +
                'display:flex;flex-direction:column;align-items:center;gap:6px;' +
                'pointer-events:none;max-width:min(720px,90vw);';
            (document.body || document.documentElement).appendChild(stack);
        }
        return stack;
    }

    function suppressed(opts) {
        if (!opts.once || !opts.id) return false;
        try {
            var store = (opts.once === 'boot' && bootId) ? localStorage : sessionStorage;
            var key = 'forgeNotify:' + opts.id + ':' + (opts.once === 'boot' ? (bootId || 'session') : 'session');
            if (store.getItem(key)) return true;
            store.setItem(key, '1');
        } catch (e) { /* storage unavailable — show it */ }
        return false;
    }

    function notify(level, msg, opts) {
        opts = opts || {};
        if (suppressed(opts)) return null;
        if (opts.id && byId[opts.id]) byId[opts.id].close();

        var conf = LEVELS[level] || LEVELS.info;
        var el = document.createElement('div');
        el.className = 'forge-notify forge-notify-' + level;
        if (opts.id) el.id = 'forge-notify-' + opts.id;
        el.style.cssText =
            'pointer-events:auto;padding:8px 18px;border-radius:0 0 8px 8px;' +
            'font:600 13px/1.6 sans-serif;color:#fff;background:' + conf.bg + ';' +
            'box-shadow:0 2px 10px rgba(0,0,0,.4);text-align:center;';

        var text = document.createElement('div');
        text.textContent = msg;
        el.appendChild(text);

        var handle = {
            el: el,
            close: function () {
                if (el.isConnected) el.remove();
                if (opts.id && byId[opts.id] === handle) delete byId[opts.id];
            },
        };

        var timeout = (opts.timeout !== undefined) ? opts.timeout : conf.timeout;

        if (opts.buttons && opts.buttons.length) {
            var row = document.createElement('div');
            row.style.cssText = 'margin-top:8px;display:flex;gap:10px;justify-content:center;';
            opts.buttons.forEach(function (b) {
                var btn = document.createElement('button');
                btn.textContent = b.label;
                btn.style.cssText =
                    'padding:6px 14px;border-radius:6px;border:none;cursor:pointer;font:600 13px sans-serif;' +
                    (b.primary ? 'background:#fff;color:#333;' : 'background:rgba(255,255,255,.25);color:#fff;');
                btn.addEventListener('click', function () {
                    handle.close();
                    if (typeof b.onClick === 'function') b.onClick();
                });
                row.appendChild(btn);
            });
            el.appendChild(row);
            timeout = (opts.timeout !== undefined) ? opts.timeout : 0;   // choices default sticky
        }

        if (timeout === 0 || opts.closable) {
            var x = document.createElement('span');
            x.textContent = '✕';
            x.style.cssText = 'position:absolute;top:4px;right:8px;cursor:pointer;opacity:.7;font-size:12px;';
            el.style.position = 'relative';
            el.style.paddingRight = '28px';
            x.addEventListener('click', handle.close);
            el.appendChild(x);
        }

        ensureStack().appendChild(el);
        if (opts.id) byId[opts.id] = handle;
        if (timeout > 0) setTimeout(handle.close, timeout);
        return handle;
    }

    window.forgeNotify = {
        info:    function (m, o) { return notify('info', m, o); },
        success: function (m, o) { return notify('success', m, o); },
        warn:    function (m, o) { return notify('warn', m, o); },
        error:   function (m, o) { return notify('error', m, o); },
        dismiss: function (id) { if (byId[id]) byId[id].close(); },
    };
})();
