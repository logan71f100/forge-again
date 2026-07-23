(function () {
    function tryInit(el) {
        if (!el || el.dataset.forgeLazyDone) return;
        // ForgeCanvas is a top-level lexical class in canvas.min.js, NOT reachable
        // by bare name/window/eval from another script — canvas.min.js exposes it
        // as window.ForgeCanvas for us.
        if (typeof window.ForgeCanvas === 'undefined') return; // canvas.min.js not ready yet
        var args;
        try { args = JSON.parse(el.dataset.forgeInit); } catch (e) { return; }
        if (!document.getElementById('container_' + args[0])) return; // container not mounted yet
        el.dataset.forgeLazyDone = '1';
        try { new window.ForgeCanvas(...args); } catch (e) { console.error('[forge] lazy canvas init failed', e); }
    }
    function scan() {
        document.querySelectorAll('.forge-lazy-canvas').forEach(tryInit);
    }
    function start() {
        scan();
        new MutationObserver(function () { scan(); }).observe(document.body, { childList: true, subtree: true });
    }
    if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);
})();
