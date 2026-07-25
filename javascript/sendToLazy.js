// "Send to img2img / inpaint / extras" from a tab like PNG Info copies its
// params via gradio bindings that are only wired when the DESTINATION tab's
// lazy gr.render body builds. Clicking such a button while the destination has
// never been opened is therefore a silent no-op (no bindings exist yet).
//
// Fix: intercept the click. If the destination isn't built, switch to it (the
// tab select flips the build gate -> gr.render builds the body -> its
// connect_paste_params_buttons() wires every registered binding, including this
// button's), wait for the build to land, then replay the click -- which now
// runs the full wired chain (copy image/params + stay on the destination).
// Once the destination is built the interceptor short-circuits, so the normal
// path is untouched.
(function() {
    var DEST = {
        img2img: {
            built: function() { return !!gradioApp().querySelector('#img2img_prompt textarea'); },
            go: function() { switch_to_img2img(); },
        },
        inpaint: {
            built: function() { return !!gradioApp().querySelector('#img2img_prompt textarea'); },
            go: function() { switch_to_inpaint(); },
        },
        extras: {
            built: function() { return !!gradioApp().querySelector('#tab_extras input'); },
            go: function() { switch_to_extras(); },
        },
        // txt2img is built eagerly -- never needs the dance.
    };

    function arm(btn, dest) {
        if (btn.dataset.forgeSendToArmed) return;
        btn.dataset.forgeSendToArmed = '1';
        btn.addEventListener('click', function() {
            var d = DEST[dest];
            if (!d || d.built()) return;   // wired -- let the normal chain run

            // Destination unbuilt: this click went nowhere. Build it, then replay.
            d.go();
            var tries = 0;
            var poll = setInterval(function() {
                if (d.built()) {
                    clearInterval(poll);
                    // beat for gradio to register the render's new bindings; the
                    // payload can be large (many re-wired paste bindings), so be
                    // generous -- a late replay is invisible, an early one is lost.
                    setTimeout(function() { btn.click(); }, 3000);
                } else if (++tries > 50) {   // ~15 s: build never landed, give up
                    clearInterval(poll);
                }
            }, 300);
        }, true);
    }

    onAfterUiUpdate(function() {
        Object.keys(DEST).forEach(function(dest) {
            gradioApp().querySelectorAll('button[id="' + dest + '_tab"]').forEach(function(b) {
                arm(b, dest);
            });
        });
    });
})();
