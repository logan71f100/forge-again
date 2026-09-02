// Page-performance forensics: answers "why did that click lag?" from the
// user's own page, in any browser, without a devtools session.
//
// Three probes, all cheap enough to leave on:
//   * every trusted click/keydown is timed from the moment it enters the page
//     until two animation frames have painted afterwards -- the time the main
//     thread stayed busy because of it. Anything over LAG_MS is logged with
//     WHAT ran in that window;
//   * a requestAnimationFrame heartbeat measures frame gaps while the tab is
//     visible, so lag that is not tied to a click (a 900ms freeze mid-typing)
//     is caught too;
//   * attribution: every event listener registered through addEventListener
//     and every MutationObserver callback is wrapped in a timer, aggregated
//     per frame -- the log line names the handlers that consumed the frame.
//     Where the browser supports long-animation-frame entries (Chromium),
//     script-level attribution (file, function, invoker) is attached as well.
//
// Lines go to the same ring buffer + client-debug.log channel as
// connectionWatchdog.js ([perf] prefix), gated on the same
// forge_client_forensics setting. Wrapping listeners costs two
// performance.now() calls per dispatch; the heartbeat is one rAF per frame.
(function () {
    'use strict';
    if (window.__forgePerfInstalled) return;
    window.__forgePerfInstalled = true;

    var LAG_MS = 100;        // a click that keeps the main thread busy longer than this is logged
    var JANK_MS = 150;       // a frame gap longer than this (visible tab) is logged
    var HEALTH_EVERY_MS = 120000;

    var ring = window.faPerfLog = window.faPerfLog || [];
    var shippedIdx = 0;
    var lastShip = 0;
    var shipTimer = null;
    var origFetch = window.fetch;

    function forensicsOn() {
        try { return !!(window.opts && window.opts.forge_client_forensics); } catch (e) { return false; }
    }

    function log(msg) {
        var line = new Date().toISOString() + ' [perf] ' + msg;
        ring.push(line);
        if (ring.length > 300) { ring.shift(); shippedIdx = Math.max(0, shippedIdx - 1); }
        if (forensicsOn()) {
            console.log('[fa-perf] ' + msg);
            scheduleShip();
        }
    }

    // batch: at most one POST per 5s, so a laggy page is never made laggier
    // by its own diagnostics
    function scheduleShip() {
        if (shipTimer) return;
        var wait = Math.max(0, 5000 - (Date.now() - lastShip));
        shipTimer = setTimeout(function () { shipTimer = null; ship(); }, wait);
    }

    function ship() {
        if (!forensicsOn()) { shippedIdx = ring.length; return; }
        if (shippedIdx >= ring.length) return;
        var lines = ring.slice(shippedIdx);
        shippedIdx = ring.length;
        lastShip = Date.now();
        try {
            origFetch('./internal/client-log', {
                method: 'POST', cache: 'no-store',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lines: lines })
            }).catch(function () { /* server down; the ring keeps the lines */ });
        } catch (e) { /* ignore */ }
    }

    // ---- attribution: timed listeners and observers -------------------------
    var frameCost = {};      // key -> ms consumed since the last heartbeat frame
    var wrapCount = 0;

    function keyFor(kind, fn) {
        var src = '';
        try { src = Function.prototype.toString.call(fn).replace(/\s+/g, ' ').slice(0, 90); } catch (e) { src = '?'; }
        return kind + ' ' + (fn.name || 'anon') + ' {' + src + '}';
    }

    function charge(key, ms) {
        frameCost[key] = (frameCost[key] || 0) + ms;
    }

    function timed(kind, fn) {
        var key = null;
        var w = function () {
            var t0 = performance.now();
            try { return fn.apply(this, arguments); }
            finally {
                var d = performance.now() - t0;
                if (d > 0.5) { if (!key) key = keyFor(kind, fn); charge(key, d); }
            }
        };
        wrapCount++;
        return w;
    }

    var wrappers = new WeakMap();   // original fn -> {type -> wrapper}
    var origAdd = EventTarget.prototype.addEventListener;
    var origRemove = EventTarget.prototype.removeEventListener;
    EventTarget.prototype.addEventListener = function (type, fn, opts) {
        if (typeof fn === 'function') {
            var per = wrappers.get(fn);
            if (!per) { per = {}; wrappers.set(fn, per); }
            if (!per[type]) per[type] = timed('on:' + type, fn);
            return origAdd.call(this, type, per[type], opts);
        }
        return origAdd.call(this, type, fn, opts);
    };
    EventTarget.prototype.removeEventListener = function (type, fn, opts) {
        if (typeof fn === 'function') {
            var per = wrappers.get(fn);
            if (per && per[type]) return origRemove.call(this, type, per[type], opts);
        }
        return origRemove.call(this, type, fn, opts);
    };

    if (window.MutationObserver) {
        var OrigMO = window.MutationObserver;
        var WrappedMO = function (cb) {
            return new OrigMO(typeof cb === 'function' ? timed('observer', cb) : cb);
        };
        WrappedMO.prototype = OrigMO.prototype;
        window.MutationObserver = WrappedMO;
    }

    function topCosts(n) {
        var arr = [];
        for (var k in frameCost) arr.push([k, frameCost[k]]);
        arr.sort(function (a, b) { return b[1] - a[1]; });
        return arr.slice(0, n).map(function (kv) { return Math.round(kv[1]) + 'ms ' + kv[0]; });
    }

    // long-animation-frame script attribution (Chromium); kept per frame gap
    var loafScripts = [];
    try {
        new PerformanceObserver(function (list) {
            list.getEntries().forEach(function (e) {
                (e.scripts || []).forEach(function (s) {
                    if (s.duration < 20) return;
                    loafScripts.push({ ms: s.duration, text: Math.round(s.duration) + 'ms ' + ((s.sourceURL || '').split('/').pop() || '?').slice(0, 40) +
                        (s.sourceFunctionName ? ':' + s.sourceFunctionName : '') + ' via ' + (s.invokerType || '?') +
                        (s.invoker ? '(' + String(s.invoker).slice(0, 50) + ')' : '') +
                        (s.forcedStyleAndLayoutDuration > 10 ? ' forcedLayout=' + Math.round(s.forcedStyleAndLayoutDuration) : '') });
                });
                if (loafScripts.length > 24) loafScripts = loafScripts.slice(-24);
            });
        }).observe({ type: 'long-animation-frame', buffered: false });
    } catch (e) { /* not supported (Firefox) -- listener attribution still works */ }

    // ---- what was the user doing: recent trusted input events -----------------
    var recentEvents = [];
    function describe(t) {
        if (!t || !t.tagName) return '?';
        var s = t.tagName.toLowerCase();
        if (t.id) s += '#' + t.id;
        else {
            var blk = t.closest && t.closest('[id]');
            if (blk && blk.id) s += ' in #' + blk.id;
            else if (t.className && typeof t.className === 'string') s += '.' + t.className.split(' ')[0];
        }
        var txt = (t.textContent || '').trim();
        if (txt && t.tagName === 'BUTTON') s += ' "' + txt.slice(0, 20) + '"';
        return s;
    }
    function noteEvent(ev) {
        if (!ev.isTrusted) return;
        recentEvents.push({ t: performance.now(), type: ev.type, target: describe(ev.target) });
        if (recentEvents.length > 8) recentEvents.shift();
    }
    ['pointerdown', 'click', 'input', 'change', 'keydown'].forEach(function (type) {
        origAdd.call(document, type, noteEvent, true);
    });

    // ---- probe 1: click -> paint --------------------------------------------
    // A report is written ~300ms AFTER the slow frame: long-animation-frame
    // entries for that frame are delivered asynchronously and would otherwise
    // be missed by the very rAF that notices the gap. Handler costs are
    // snapshotted synchronously (the heartbeat resets them per frame); script
    // attribution is picked up when the report fires.
    function report(prefix) {
        var handlers = topCosts(5);
        var dom = document.getElementsByTagName('*').length;
        setTimeout(function () {
            var parts = handlers.slice();
            if (loafScripts.length) {
                loafScripts.sort(function (a, b) { return b.ms - a.ms; });
                parts.push('scripts: ' + loafScripts.slice(0, 6).map(function (s) { return s.text; }).join(' | '));
            }
            loafScripts = [];
            log(prefix + ' -- ' + (parts.length ? parts.join(' ; ') : 'no wrapped handler over 0.5ms (framework-internal work)') + ' ; dom=' + dom);
        }, 300);
    }

    function recent(sinceMs) {
        var now = performance.now();
        return recentEvents.filter(function (e) { return now - e.t <= sinceMs; })
            .map(function (e) { return e.type + ' ' + e.target + ' ' + Math.round((now - e.t) / 100) / 10 + 's ago'; });
    }

    origAdd.call(document, 'click', function (ev) {
        if (!ev.isTrusted) return;
        var t0 = performance.now();
        var target = describe(ev.target);
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                var d = performance.now() - t0;
                if (d < LAG_MS) return;
                report('click lag ' + Math.round(d) + 'ms on ' + target);
            });
        });
    }, true);

    // ---- probe 2: frame-gap heartbeat --------------------------------------
    var lastFrame = null;
    function heartbeat(now) {
        if (lastFrame !== null && !document.hidden) {
            var gap = now - lastFrame;
            if (gap > JANK_MS) {
                var ev = recent(gap + 3000);
                report('frame gap ' + Math.round(gap) + 'ms' + (ev.length ? ' after ' + ev.join(', ') : ' (no user input in the last 3s)'));
            }
        }
        lastFrame = now;
        frameCost = {};
        requestAnimationFrame(heartbeat);
    }
    requestAnimationFrame(heartbeat);
    document.addEventListener('visibilitychange', function () { lastFrame = null; });

    // ---- probe 3: periodic health line -------------------------------------
    function health() {
        if (document.hidden) return;
        try {
            var imgs = document.images;
            var dataBytes = 0, dataImgs = 0;
            for (var i = 0; i < imgs.length; i++) {
                var src = imgs[i].getAttribute('src') || '';
                if (src.indexOf('data:') === 0) { dataBytes += src.length; dataImgs++; }
            }
            var taBytes = 0;
            var tas = document.getElementsByTagName('textarea');
            for (var j = 0; j < tas.length; j++) taBytes += (tas[j].value || '').length;
            var mounted = [];
            var roots = document.querySelectorAll('[id^="tab_"]');
            for (var k = 0; k < roots.length; k++) if (!/-button$/.test(roots[k].id)) mounted.push(roots[k].id.slice(4));
            var heap = (performance.memory && performance.memory.usedJSHeapSize) ? ' heap=' + Math.round(performance.memory.usedJSHeapSize / 1048576) + 'MB' : '';
            log('health dom=' + domSize() + ' imgs=' + imgs.length + ' dataUriImgs=' + dataImgs + ' (' + Math.round(dataBytes / 1024) + 'KB)' +
                ' textareaChars=' + Math.round(taBytes / 1024) + 'K listeners=' + wrapCount + ' styleTags=' + document.querySelectorAll('style').length +
                ' mounted=' + mounted.join(',') + heap + ' uptime=' + Math.round(performance.now() / 60000) + 'min');
        } catch (e) { /* diagnostics never throw */ }
    }
    setInterval(health, HEALTH_EVERY_MS);
    setTimeout(health, 20000);
})();
