// Background-tab-immune timers.
//
// Chrome (and friends) throttle main-thread setTimeout/setInterval to roughly
// once per minute in a hidden tab. Anything cosmetic can live with that, but a
// polling loop that carries CORRECTNESS -- progress updates, live previews, and
// especially the "the job finished" detection that clears the task id and
// restores the Generate button -- silently stops working the moment the user
// switches tabs. Web Worker timers are not throttled, so pace the callbacks
// from a worker and run them on the main thread.
//
//   forgeTimer.setTimeout(fn, ms) -> id
//   forgeTimer.clearTimeout(id)
//
// Falls back to window.setTimeout when workers are unavailable (CSP, etc.).
window.forgeTimer = (function () {
    'use strict';

    var worker = null;          // null = not built yet, false = unavailable
    var seq = 0;
    var pending = {};

    function ensureWorker() {
        if (worker !== null) return worker;
        try {
            var src = 'onmessage=function(e){setTimeout(function(){postMessage(e.data.id);},e.data.ms);};';
            worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'application/javascript' })));
            worker.onmessage = function (e) {
                var fn = pending[e.data];
                delete pending[e.data];
                if (!fn) return;                        // cleared before it fired
                try {
                    fn();
                } catch (err) {
                    console.error('[forgeTimer] callback failed:', err);
                }
            };
        } catch (e) {
            console.warn('[forgeTimer] worker unavailable, falling back to main-thread timers:', e);
            worker = false;
        }
        return worker;
    }

    var api = {
        setTimeout: function (fn, ms) {
            var w = ensureWorker();
            if (!w) return window.setTimeout(fn, ms);
            var id = 'ft' + (++seq);
            pending[id] = fn;
            w.postMessage({ id: id, ms: ms });
            return id;
        },
        clearTimeout: function (id) {
            if (typeof id === 'string' && id.indexOf('ft') === 0) {
                delete pending[id];                     // worker still fires; callback is gone
            } else if (id !== undefined) {
                window.clearTimeout(id);
            }
        },
    };

    // ---- requestAnimationFrame in hidden tabs --------------------------------
    // Browsers PAUSE rAF completely while a tab is hidden -- it does not fire
    // late, it does not fire at all until the tab is looked at again. That is
    // fine for animation, and fatal for anything on a correctness path.
    //
    // Svelte 5's tick() -- which gradio's client awaits while submitting -- is:
    //     new Promise(e => { requestAnimationFrame(() => e()); setTimeout(() => e()); })
    // i.e. it resolves on whichever fires first. In a hidden tab the rAF half
    // never fires, so every await falls back to a THROTTLED timer, and a submit
    // that awaits several ticks stalls for seconds -- or until the tab regains
    // focus and the queued rAF callbacks all fire at once. Reported exactly
    // that way: "hit generate, switched tabs, it didn't start until I came
    // back".
    //
    // So: while the document is hidden, service rAF callbacks from the Worker
    // clock (which browsers do not throttle). Visible tabs keep the real rAF,
    // untouched, so animation timing and vsync alignment are unaffected.
    try {
        var nativeRAF = window.requestAnimationFrame.bind(window);
        var nativeCAF = window.cancelAnimationFrame.bind(window);
        var shimmed = Object.create(null);

        window.requestAnimationFrame = function (cb) {
            if (!document.hidden) return nativeRAF(cb);
            var id = api.setTimeout(function () {
                delete shimmed[id];
                try {
                    cb(performance.now());              // rAF passes a timestamp
                } catch (e) {
                    console.error('[forgeTimer] rAF callback failed:', e);
                }
            }, 16);
            shimmed[id] = true;
            return id;
        };

        window.cancelAnimationFrame = function (id) {
            if (id !== undefined && shimmed[id]) {
                delete shimmed[id];
                return api.clearTimeout(id);
            }
            return nativeCAF(id);
        };
    } catch (e) {
        console.warn('[forgeTimer] could not shim requestAnimationFrame:', e);
    }

    return api;
})();
