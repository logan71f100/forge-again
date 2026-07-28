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
    var banner = null;

    function ensureBanner() {
        if (banner) return banner;
        banner = document.createElement('div');
        banner.id = 'fa-conn-banner';
        banner.style.cssText =
            'position:fixed;top:0;left:50%;transform:translateX(-50%);z-index:99999;' +
            'padding:6px 18px;border-radius:0 0 8px 8px;font:600 13px/1.5 sans-serif;' +
            'box-shadow:0 2px 8px rgba(0,0,0,.35);display:none;pointer-events:none;';
        (document.body || document.documentElement).appendChild(banner);
        return banner;
    }

    function recoverStuckUI() {
        // Backend is reachable again — clear any stuck "Queued…" placeholder and
        // bring the Generate button back so the user can re-submit. No reload.
        try {
            var app = (typeof gradioApp === 'function') ? gradioApp() : document;
            app.querySelectorAll('.progressDiv.pending-placeholder').forEach(function (p) { p.remove(); });
            if (typeof showSubmitButtons === 'function') {
                app.querySelectorAll('button[id$="_generate"]').forEach(function (btn) {
                    showSubmitButtons(btn.id.slice(0, -'_generate'.length), true);
                });
            }
        } catch (e) { /* UI not ready */ }
    }

    function setOnline(nowOnline) {
        if (nowOnline) everConnected = true;
        if (nowOnline === online) return;
        online = nowOnline;
        var b = ensureBanner();
        if (!online) {
            b.textContent = '⚠ Lost connection to the server — reconnecting…';
            b.style.background = '#c0392b'; b.style.color = '#fff'; b.style.display = 'block';
        } else if (everConnected) {
            b.textContent = '✓ Reconnected';
            b.style.background = '#1f9d55'; b.style.color = '#fff'; b.style.display = 'block';
            setTimeout(function () { if (online) b.style.display = 'none'; }, 2500);
            recoverStuckUI();
        }
    }

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
        var b = ensureBanner();
        b.textContent = '↻ Server restarted — reloading the UI…';
        b.style.background = '#b7791f'; b.style.color = '#fff'; b.style.display = 'block';
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
                var id = data && data.boot_id;
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
    }

    if (document.body) start();
    else document.addEventListener('DOMContentLoaded', start);
})();
