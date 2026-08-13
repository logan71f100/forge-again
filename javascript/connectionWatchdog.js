// Front-end connection resilience for gradio 6.
//
// Two failure modes this addresses, both rooted in browsers throttling
// backgrounded tabs (main-thread setInterval/setTimeout drop to <=1/min and
// gradio's own keep-alive starves, so the session/queue connection goes stale):
//   1) after the tab has been idle or backgrounded, a Generate click shows
//      "Queued…" but never reaches the server;
//   2) switching tabs mid-flight leaves the request never sent.
//
// Strategy: run the heartbeat inside a Web Worker (NOT throttled when hidden) to
// keep the session warm and to detect up/down reliably; show a clear banner; and
// on reconnect / tab re-focus, clear any stuck generation UI and restore the
// Generate button so the user can re-run immediately. Never reloads the page, so
// every input value / setting is preserved.
(function () {
    'use strict';

    var online = true;
    var everConnected = false;

    function recoverStuckUI(fullCleanup) {
        // Backend is reachable again — clear any stuck "Queued…" placeholder and
        // bring the Generate button back so the user can re-submit. No reload.
        // fullCleanup additionally removes ACTIVE progress bars — only safe when
        // the server has confirmed nothing is running or queued.
        try {
            var app = (typeof gradioApp === 'function') ? gradioApp() : document;
            var sel = fullCleanup ? '.progressDiv' : '.progressDiv.pending-placeholder';
            app.querySelectorAll(sel).forEach(function (p) { p.remove(); });
            if (typeof showSubmitButtons === 'function') {
                app.querySelectorAll('button[id$="_generate"]').forEach(function (btn) {
                    showSubmitButtons(btn.id.slice(0, -'_generate'.length), true);
                });
            }
        } catch (e) { /* UI not ready */ }
    }

    // ---- missed-result recovery ---------------------------------------------
    // submit() stores the task id in localStorage and only a DELIVERED
    // completion removes it. If the completion was lost (throttled background
    // tab, dead stream), the id lingers and the server still holds the result
    // in progress.recorded_results -- the hidden {tab}_restore_progress button
    // fetches it back through a FRESH connection. Fire it on reconnect, on
    // orphan recovery, and on tab-refocus with an idle server; one attempt per
    // task id so a truly unknown id can't loop.
    var attemptedRestores = {};

    function recoverMissedResult(reason) {
        ['txt2img', 'img2img'].forEach(function (tab) {
            var id = null;
            try { id = localStorage.getItem(tab + '_task_id'); } catch (e) { return; }
            if (!id || attemptedRestores[id]) return;
            var btn = (typeof gradioApp === 'function' ? gradioApp() : document).getElementById(tab + '_restore_progress');
            if (!btn) return;
            attemptedRestores[id] = true;
            slog('recovering missed result for ' + tab + ' (task ' + id + ', trigger: ' + reason + ')');
            if (typeof forgeNotify !== 'undefined') {
                forgeNotify.info('↺ Recovering the finished image…', { id: 'fa-recover', timeout: 5000 });
            }
            var app = (typeof gradioApp === 'function') ? gradioApp() : document;
            var gallery = app.querySelector('#' + tab + '_gallery');
            var hadImg = !!(gallery && gallery.querySelector('img'));
            btn.click();
            // One shot -- but the removal MUST lag the click. The button's
            // handler is a gradio `js=` function (restoreProgressTxt2img) that
            // reads this very key, and gradio 6 evaluates it asynchronously,
            // AFTER this synchronous click handler returns. Removing it inline
            // won the race and handed restore_progress a null id, so recovery
            // silently no-op'd and then blamed the server for evicting it.
            // attemptedRestores[] already prevents a retry loop.
            setTimeout(function () {
                try { localStorage.removeItem(tab + '_task_id'); } catch (e2) { }
            }, 3000);
            // confirm it actually delivered -- restore_progress returns a
            // "results discarded" message (no image) when the server has
            // already evicted this task's result. Tell the user plainly instead
            // of leaving a "recovering..." toast that did nothing.
            setTimeout(function () {
                var nowImg = !!(gallery && gallery.querySelector('img'));
                if (nowImg && !hadImg) {
                    slog('recovery delivered an image for ' + tab);
                    if (typeof forgeNotify !== 'undefined') forgeNotify.success('✓ Recovered the finished image', { id: 'fa-recover', timeout: 3000 });
                } else if (!nowImg) {
                    slog('recovery found no stored result for ' + tab + ' (server evicted it)');
                    if (typeof forgeNotify !== 'undefined') forgeNotify.warn('The finished image is in your output folder — the live result was already cleared from the server.', { id: 'fa-recover', timeout: 7000 });
                }
            }, 4000);
        });
    }

    // ---- orphaned-generation detection --------------------------------------
    // gradio 6 delivers completion over an SSE event stream. If that stream
    // dies mid-job (network blip, sleep/wake, proxy timeout) the server keeps
    // working and finishes, but the page never hears about it: the UI shows
    // "generating" forever while the server sits idle. The ping payload's
    // `busy` flag (running or queued anywhere) lets us catch that state:
    // UI-generating + server-idle for several consecutive beats => reset the
    // controls in place (no reload; the finished image is in the output dir).
    var idleStrikes = 0;

    function uiLooksGenerating() {
        try {
            var app = (typeof gradioApp === 'function') ? gradioApp() : document;
            var buttons = app.querySelectorAll('button[id$="_interrupt"]');
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].offsetParent && getComputedStyle(buttons[i]).display !== 'none') return true;
            }
        } catch (e) { /* UI not ready */ }
        return false;
    }

    function checkOrphanedGeneration(serverBusy) {
        if (serverBusy !== false) { idleStrikes = 0; return; }   // busy or older server (no flag)
        if (!uiLooksGenerating()) { idleStrikes = 0; return; }
        idleStrikes++;
        if (idleStrikes < 4) return;                             // ~16s of disagreement
        idleStrikes = 0;
        recoverStuckUI(true);
        recoverMissedResult('orphan');
        forgeNotify.warn('⚠ Lost the generation connection — recovering the finished image…', { id: 'fa-orphan', timeout: 8000 });
    }

    function setOnline(nowOnline) {
        if (nowOnline) everConnected = true;
        if (nowOnline === online) return;
        online = nowOnline;
        if (!online) {
            forgeNotify.error('⚠ Lost connection to the server — reconnecting…', { id: 'fa-conn', timeout: 0 });
        } else if (everConnected) {
            forgeNotify.success('✓ Reconnected', { id: 'fa-conn', timeout: 2500 });
            recoverStuckUI();
            recoverMissedResult('reconnect');
        }
    }

    // ---- queue-stream forensics ---------------------------------------------
    // gradio 6 receives ALL queue events (progress, completion + the result
    // gallery payload) over a fetch'd text/event-stream on /queue/data. When it
    // dies mid-job the server finishes but the page shows stale results — the
    // orphan detector below cleans up, but to actually FIX the drops we need to
    // see them. Wrap fetch for queue/data: log open/close/error/cancel with
    // durations ([fa-stream] in the console, ring buffer on
    // window.faStreamLog), and banner immediately when a stream dies while the
    // UI is mid-generation.
    window.faStreamLog = window.faStreamLog || [];
    var shippedIdx = 0;
    function slog(msg) {
        var line = new Date().toISOString() + ' ' + msg;
        window.faStreamLog.push(line);
        if (window.faStreamLog.length > 200) { window.faStreamLog.shift(); shippedIdx = Math.max(0, shippedIdx - 1); }
        console.log('[fa-stream] ' + line);
    }

    // Ship new log lines to the server (client-debug.log) so connection
    // problems can be diagnosed server-side. Piggybacks on the ping cadence.
    function shipLog() {
        if (shippedIdx >= window.faStreamLog.length) return;
        var lines = window.faStreamLog.slice(shippedIdx);
        shippedIdx = window.faStreamLog.length;
        try {
            origFetch('./internal/client-log', {
                method: 'POST', cache: 'no-store',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lines: lines })
            }).catch(function () { /* server down; lines stay in console */ });
        } catch (e) { /* ignore */ }
    }

    // Decode the SSE bytes flowing through the monitored stream and log every
    // queue event type — this shows definitively whether process_completed
    // (which carries the result gallery payload) ever reached this page.
    function makeSSEDecoder() {
        var buf = '';
        var dec = new TextDecoder();
        var state = { sawClose: false };
        var feed = function (chunk) {
            try {
                buf += dec.decode(chunk, { stream: true });
                var parts = buf.split('\n\n');
                buf = parts.pop();
                parts.forEach(function (block) {
                    var m = block.match(/^data:\s*(.*)$/m);
                    if (!m) return;
                    try {
                        var ev = JSON.parse(m[1]);
                        var msg = ev.msg || '(no msg)';
                        if (msg === 'process_completed') {
                            var out = ev.output || {};
                            var summary = 'success=' + ev.success;
                            try {
                                var dataArr = out.data || [];
                                var imgs = JSON.stringify(dataArr).match(/"(?:url|path)":/g);
                                summary += ' outputs=' + dataArr.length + ' fileRefs=' + (imgs ? imgs.length : 0);
                            } catch (e2) { /* summary best-effort */ }
                            slog('event process_completed ' + summary);
                        } else if (msg !== 'progress' && msg !== 'estimation') {
                            if (msg === 'close_stream') state.sawClose = true;
                            slog('event ' + msg);
                        }
                    } catch (e3) { /* non-JSON data line */ }
                });
            } catch (e) { /* decoder hiccup — never break the stream */ }
        };
        feed.state = state;
        return feed;
    }

    // Gallery image-load failures: the completion event can arrive fine and the
    // <img> fetch still 404 (tempdir/cache URL problems) — catch those too.
    document.addEventListener('error', function (ev) {
        var el = ev.target;
        if (!el || el.tagName !== 'IMG') return;
        if (!el.closest('.gradio-gallery')) return;
        var src = el.src || '(no src)';
        slog('gallery IMG failed to load: ' + src.slice(0, 300));
        try {
            origFetch(src, { method: 'GET', cache: 'no-store' }).then(function (r) {
                slog('gallery IMG url probe: HTTP ' + r.status + ' for ' + src.slice(0, 120));
                shipLog();
            }).catch(function (e) { slog('gallery IMG url probe failed: ' + e); shipLog(); });
        } catch (e) { /* ignore */ }
    }, true);

    function streamDied(kind, seconds) {
        slog('queue stream ' + kind + ' after ' + seconds + 's');
        if (!uiLooksGenerating()) return;
        forgeNotify.warn('⚠ Generation stream ' + kind + ' after ' + seconds + 's — the run continues server-side; result lands in the output folder.', { id: 'fa-stream', timeout: 8000 });
    }

    var origFetch = window.fetch;
    window.fetch = function (input) {
        var url = (typeof input === 'string') ? input : ((input && input.url) || '');
        if (url.indexOf('queue/data') === -1) return origFetch.apply(this, arguments);
        var t0 = Date.now();
        var secs = function () { return ((Date.now() - t0) / 1000).toFixed(1); };
        slog('queue stream opened');
        return origFetch.apply(this, arguments).then(function (resp) {
            try {
                if (!resp || !resp.body || !window.ReadableStream) return resp;
                var reader = resp.body.getReader();
                var decode = makeSSEDecoder();
                var monitored = new ReadableStream({
                    pull: function (controller) {
                        return reader.read().then(function (r) {
                            if (r.done) { slog('queue stream closed cleanly after ' + secs() + 's'); shipLog(); controller.close(); return; }
                            decode(r.value);
                            controller.enqueue(r.value);
                        }).catch(function (err) {
                            if (decode.state.sawClose) {
                                // gradio aborts its own fetch after close_stream — normal teardown
                                slog('queue stream ended (client abort after close_stream) after ' + secs() + 's');
                            } else {
                                streamDied('DROPPED (' + (err && err.name || 'error') + ')', secs());
                            }
                            shipLog();
                            controller.error(err);
                        });
                    },
                    cancel: function (reason) {
                        slog('queue stream cancelled by page after ' + secs() + 's');
                        return reader.cancel(reason);
                    }
                });
                return new Response(monitored, { status: resp.status, statusText: resp.statusText, headers: resp.headers });
            } catch (e) { return resp; }
        }, function (err) {
            streamDied('FAILED to open (' + (err && err.name || 'error') + ')', secs());
            throw err;
        });
    };

    // ---- pre-generate source snapshot ---------------------------------------
    // "output is a blank/white image" reports: log what the img2img source
    // binds actually hold at the moment Generate is clicked (length, decoded
    // dimensions, mean RGBA of a 16x16 downsample). If the source is blank AT
    // SUBMIT the canvas->bind sync lost it; if it's intact the bug is
    // server-side. Ships with the forensics log.
    function snapshotBind(ta, name) {
        var v = (ta && ta.value) || '';
        if (!v.startsWith('data:image')) {
            slog('generate: ' + name + ' bind len=' + v.length + (v ? ' head=' + v.slice(0, 24) : ' (EMPTY)'));
            return;
        }
        var img = new Image();
        img.onload = function () {
            try {
                var cc = document.createElement('canvas'); cc.width = 16; cc.height = 16;
                var g = cc.getContext('2d'); g.drawImage(img, 0, 0, 16, 16);
                var d = g.getImageData(0, 0, 16, 16).data;
                var s = [0, 0, 0, 0];
                for (var i = 0; i < d.length; i += 4) { s[0] += d[i]; s[1] += d[i + 1]; s[2] += d[i + 2]; s[3] += d[i + 3]; }
                var n = d.length / 4;
                slog('generate: ' + name + ' bind ' + img.width + 'x' + img.height + ' len=' + v.length +
                     ' meanRGBA=' + s.map(function (x) { return Math.round(x / n); }).join(','));
                shipLog();
            } catch (e) { slog('generate: ' + name + ' bind decode stats failed: ' + e); shipLog(); }
        };
        img.onerror = function () { slog('generate: ' + name + ' bind data URL FAILED to decode (len=' + v.length + ')'); shipLog(); };
        img.src = v;
    }

    document.addEventListener('click', function (ev) {
        var btn = ev.target && ev.target.closest ? ev.target.closest('button[id$="_generate"]') : null;
        if (!btn || btn.id.indexOf('img2img') === -1) return;
        try {
            var app = (typeof gradioApp === 'function') ? gradioApp() : document;
            var panes = app.querySelectorAll('#mode_img2img div.tabitem');
            for (var i = 0; i < panes.length; i++) {
                var pane = panes[i];
                var dc = pane.querySelector('canvas[id^=drawingCanvas_]');
                if (!dc || !dc.offsetParent) continue;   // not the visible pane
                var uuid = dc.id.replace('drawingCanvas_', '');
                slog('generate clicked: visible pane ' + (pane.id || i) + ' canvas ' + dc.width + 'x' + dc.height);
                snapshotBind(app.querySelector('.logical_image_background[id="' + uuid + '"] textarea'), 'background');
                snapshotBind(app.querySelector('.logical_image_foreground[id="' + uuid + '"] textarea'), 'foreground');
            }
            shipLog();
        } catch (e) { /* forensics must never break generate */ }
    }, true);

    // ---- server-restart detection -------------------------------------------
    // A restart is NOT a blip: this page's gradio session died with the old
    // process, so progress events freeze mid-run and new submits misbehave in
    // subtle ways. The ping payload carries a per-process boot_id; when it
    // changes, save the prompt fields and reload into the new server. The
    // sessionStorage guard makes the reload one-shot per boot so a broken
    // server can't cause a reload loop.
    var bootId = null;

    function savePrompts() {
        try {
            var app = (typeof gradioApp === 'function') ? gradioApp() : document;
            var saved = { t: Date.now(), fields: {} };
            ['txt2img_prompt', 'txt2img_neg_prompt', 'img2img_prompt', 'img2img_neg_prompt'].forEach(function (id) {
                var ta = app.querySelector('#' + id + ' textarea');
                if (ta && ta.value) saved.fields[id] = ta.value;
            });
            localStorage.setItem('fa-restart-prompts', JSON.stringify(saved));
        } catch (e) { /* storage unavailable */ }
    }

    function restorePrompts(attempt) {
        attempt = attempt || 0;
        try {
            var raw = localStorage.getItem('fa-restart-prompts');
            if (!raw) return;
            var saved = JSON.parse(raw);
            if (!saved || Date.now() - saved.t > 10 * 60 * 1000) {          // stale
                localStorage.removeItem('fa-restart-prompts');
                return;
            }
            // gradio renders after DOMContentLoaded (and some tabs lazily), so
            // wait until at least one saved field's textarea exists before
            // consuming the save; give up quietly after ~20s.
            var app = (typeof gradioApp === 'function') ? gradioApp() : document;
            var ids = Object.keys(saved.fields);
            var present = ids.filter(function (id) { return app.querySelector('#' + id + ' textarea'); });
            if (present.length === 0) {
                if (attempt < 40) setTimeout(function () { restorePrompts(attempt + 1); }, 500);
                else localStorage.removeItem('fa-restart-prompts');
                return;
            }
            localStorage.removeItem('fa-restart-prompts');
            ids.forEach(function (id) {
                var ta = app.querySelector('#' + id + ' textarea');
                if (ta && !ta.value) {
                    ta.value = saved.fields[id];
                    if (typeof updateInput === 'function') updateInput(ta);
                    else ta.dispatchEvent(new Event('input', { bubbles: true }));
                }
            });
        } catch (e) { /* storage unavailable */ }
    }

    function onServerRestarted(newBootId) {
        if (sessionStorage.getItem('fa-reloaded-for') === newBootId) return;   // already reloaded once for this boot
        sessionStorage.setItem('fa-reloaded-for', newBootId);
        savePrompts();
        forgeNotify.warn('↻ Server restarted — reloading the UI…', { id: 'fa-conn', timeout: 0 });
        setTimeout(function () { location.reload(); }, 800);
    }

    var inflight = false;
    function ping() {
        if (inflight) return;
        inflight = true;
        var ctrl = new AbortController();
        var to = setTimeout(function () { ctrl.abort(); }, 6000);
        fetch('./internal/ping', { method: 'GET', cache: 'no-store', signal: ctrl.signal })
            .then(function (r) {
                clearTimeout(to); inflight = false;
                setOnline(!!r && r.ok);
                if (!r || !r.ok) return null;
                return r.json().catch(function () { return null; });
            })
            .then(function (data) {
                shipLog();   // piggyback: flush any new forensic lines to the server
                if (!data) return;
                checkOrphanedGeneration(data.busy);
                // idle server + lingering task id = a completion this page never
                // received (classic throttled-background-tab loss)
                if (data.busy === false && !uiLooksGenerating()) recoverMissedResult('idle-check');
                var id = data.boot_id;
                if (!id) return;                       // older server: no boot_id, keep blip-only behavior
                if (bootId === null) { bootId = id; return; }
                if (id !== bootId) onServerRestarted(id);
            })
            .catch(function () { clearTimeout(to); inflight = false; setOnline(false); });
    }

    // Heartbeat from a Web Worker so it keeps firing at a steady 4s even when the
    // tab is backgrounded (a plain setInterval would be throttled to >=60s, which
    // is what lets the connection go stale in the first place). Fall back to
    // setInterval if workers are unavailable (CSP, etc.).
    var started = false;
    function start() {
        if (started) return;
        started = true;
        var src = 'var t=setInterval(function(){postMessage(0)},4000);onmessage=function(){clearInterval(t)};';
        var worker = null;
        try {
            worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'application/javascript' })));
            worker.onmessage = ping;
        } catch (e) {
            setInterval(ping, 4000);
        }
        ping();   // immediate first check
        restorePrompts();   // bring back prompts saved just before a restart-reload

        // Re-check the instant the tab regains focus/visibility, so returning to a
        // backgrounded tab recovers a stuck generation right away.
        document.addEventListener('visibilitychange', function () { if (!document.hidden) ping(); });
        window.addEventListener('focus', ping);
        window.addEventListener('online', ping);
        window.addEventListener('offline', function () { setOnline(false); });
        // flush unshipped forensics when the page goes away
        window.addEventListener('pagehide', function () {
            if (shippedIdx >= window.faStreamLog.length || !navigator.sendBeacon) return;
            var lines = window.faStreamLog.slice(shippedIdx);
            shippedIdx = window.faStreamLog.length;
            navigator.sendBeacon('./internal/client-log',
                new Blob([JSON.stringify({ lines: lines })], { type: 'application/json' }));
        });
    }

    if (document.body) start();
    else document.addEventListener('DOMContentLoaded', start);
})();
