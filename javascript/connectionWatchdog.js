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

    var inflight = false;
    function ping() {
        if (inflight) return;
        inflight = true;
        var ctrl = new AbortController();
        var to = setTimeout(function () { ctrl.abort(); }, 6000);
        fetch('./internal/ping', { method: 'GET', cache: 'no-store', signal: ctrl.signal })
            .then(function (r) { clearTimeout(to); inflight = false; setOnline(!!r && r.ok); })
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
