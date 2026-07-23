// Gallery zoom & pan for Forge's full-screen image viewer (lightbox).
// Forge's built-in viewer only toggles fit vs. actual size; this adds smooth
// scroll-to-zoom centred on the cursor, plus click-drag panning — handy for
// inspecting inpaint detail. Scoped to #modalImage only, so it never touches
// the inpaint masking canvas or the AI assistant panel.

(function () {
    'use strict';

    function setup() {
        const app = (typeof gradioApp === 'function') ? gradioApp() : document;
        const modal = app.getElementById ? app.getElementById('lightboxModal') : document.getElementById('lightboxModal');
        const img = app.getElementById ? app.getElementById('modalImage') : document.getElementById('modalImage');
        if (!modal || !img || img.dataset.zoomWired) return;
        img.dataset.zoomWired = '1';

        let scale = 1, tx = 0, ty = 0;
        let dragging = false, moved = false, sx = 0, sy = 0, stx = 0, sty = 0;

        function apply() {
            img.style.transformOrigin = '0 0';
            img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
            img.style.cursor = scale > 1 ? (dragging ? 'grabbing' : 'grab') : 'auto';
        }

        function reset() {
            scale = 1; tx = 0; ty = 0; apply();
        }

        // reset whenever a new image is shown or the modal is reopened
        const mo = new MutationObserver(reset);
        mo.observe(img, { attributes: true, attributeFilter: ['src'] });
        const modalMo = new MutationObserver(() => {
            if (modal.style.display === 'none') reset();
        });
        modalMo.observe(modal, { attributes: true, attributeFilter: ['style'] });

        img.addEventListener('wheel', (e) => {
            e.preventDefault();
            e.stopPropagation();
            // rect already reflects the current transform, so the element's
            // fixed layout origin is rect.(left/top) minus the current pan.
            const rect = img.getBoundingClientRect();
            const baseX = rect.left - tx;
            const baseY = rect.top - ty;
            // cursor in the element's layout space
            const cx = e.clientX - baseX;
            const cy = e.clientY - baseY;
            // image-space point currently under the cursor
            const ix = (cx - tx) / scale;
            const iy = (cy - ty) / scale;
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            let ns = Math.max(1, Math.min(8, scale * factor));
            if (ns === scale) return;
            // solve tx so that ix maps back under the cursor at the new scale
            tx = cx - ix * ns;
            ty = cy - iy * ns;
            scale = ns;
            if (scale === 1) { tx = 0; ty = 0; }
            apply();
        }, { passive: false });

        img.addEventListener('mousedown', (e) => {
            if (scale <= 1) return;   // nothing to pan at fit size
            dragging = true; moved = false;
            sx = e.clientX; sy = e.clientY; stx = tx; sty = ty;
            e.preventDefault();
            apply();
        });

        window.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            tx = stx + (e.clientX - sx);
            ty = sty + (e.clientY - sy);
            if (Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) > 3) moved = true;
            apply();
        });

        window.addEventListener('mouseup', () => {
            if (dragging) dragging = false;
            apply();
        }, true);

        // swallow the click that ends a drag so it doesn't toggle Forge's
        // built-in fit/fullscreen handler
        img.addEventListener('click', (e) => {
            if (moved) { e.preventDefault(); e.stopPropagation(); moved = false; }
        }, true);

        // double-click resets to fit
        img.addEventListener('dblclick', (e) => {
            e.preventDefault(); e.stopPropagation();
            reset();
        }, true);
    }

    // the lightbox img exists early, but re-check on UI loads / mutations
    if (typeof onUiLoaded === 'function') onUiLoaded(setup);
    document.addEventListener('DOMContentLoaded', setup);
    const poll = setInterval(setup, 1500);
    setTimeout(() => clearInterval(poll), 30000);
})();
