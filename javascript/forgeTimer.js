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

    return {
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
})();
