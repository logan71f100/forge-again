// After a generation replaces a gallery's items, gradio 6 keeps (or clamps) the
// old preview selection, which lands on the LAST item — for Replacer that's the
// mask composite instead of the generated result. Whenever a watched gallery's
// item set changes, select the first item so the fresh result is what you see.
(function() {
    'use strict';

    var WATCHED = ['txt2img_gallery', 'img2img_gallery', 'replacer_gallery', 'replacer_video_gallery', 'extras_gallery'];
    var lastKey = {};

    function itemButtons(g) {
        // preview mode: the bottom thumbnail strip; grid mode: the tile buttons
        var thumbs = g.querySelectorAll('.thumbnail-item, .thumbnails button');
        if (thumbs.length) return [...thumbs];
        return [...g.querySelectorAll('button.media-button')].filter(function(b) { return b.querySelector('img'); });
    }

    function tick() {
        for (var i = 0; i < WATCHED.length; i++) {
            var id = WATCHED[i];
            var g = gradioApp().getElementById(id);
            if (!g) continue;
            var imgs = [...g.querySelectorAll('img')];
            // ignore live-preview frames (data:/blob: URIs) — wait for real files
            var files = imgs.map(function(im) { return im.src; }).filter(function(s) { return s.indexOf('file=') !== -1; });
            if (!files.length) { continue; }
            var key = files.slice().sort().join('|');
            if (lastKey[id] === undefined) { lastKey[id] = key; continue; }   // adopt initial state silently
            if (key === lastKey[id]) continue;
            lastKey[id] = key;
            // new item set: select the first item (only matters in preview mode,
            // and clicking the first thumbnail there is exactly what a user would do)
            (function(gallery) {
                setTimeout(function() {
                    var items = itemButtons(gallery);
                    if (items.length > 1 && gallery.querySelector('.preview')) items[0].click();
                }, 250);
            })(g);
        }
    }

    if (typeof onUiLoaded === 'function') onUiLoaded(function() { setInterval(tick, 600); });
})();
