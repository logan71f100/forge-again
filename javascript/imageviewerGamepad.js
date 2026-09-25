let gamepads = [];

function lightboxIsOpen() {
    const lb = gradioApp().getElementById('lightboxModal');
    return !!lb && lb.style.display === 'flex';
}

window.addEventListener('gamepadconnected', (e) => {
    const index = e.gamepad.index;
    let isWaiting = false;
    gamepads[index] = setInterval(async() => {
        if (!opts.js_modal_lightbox_gamepad || isWaiting || !lightboxIsOpen()) return;
        const gamepad = navigator.getGamepads()[index];
        if (!gamepad) return; // Chromium reports null slots around a disconnect
        const xValue = gamepad.axes[0];
        if (xValue <= -0.3) {
            modalPrevImage(e);
            isWaiting = true;
        } else if (xValue >= 0.3) {
            modalNextImage(e);
            isWaiting = true;
        }
        if (isWaiting) {
            await sleepUntil(() => {
                const gp = navigator.getGamepads()[index];
                if (!gp) return true;
                const xValue = gp.axes[0];
                if (xValue < 0.3 && xValue > -0.3) {
                    return true;
                }
            }, opts.js_modal_lightbox_gamepad_repeat);
            isWaiting = false;
        }
    }, 10);
});

window.addEventListener('gamepaddisconnected', (e) => {
    clearInterval(gamepads[e.gamepad.index]);
});

/*
Primarily for vr controller type pointer devices.
I use the wheel event because there's currently no way to do it properly with web xr.
 */
let isScrolling = false;
window.addEventListener('wheel', (e) => {
    // only while the lightbox is open: otherwise any horizontal trackpad
    // scroll anywhere on the page re-selected gallery images
    if (!opts.js_modal_lightbox_gamepad || isScrolling || !lightboxIsOpen()) return;
    isScrolling = true;

    if (e.deltaX <= -0.6) {
        modalPrevImage(e);
    } else if (e.deltaX >= 0.6) {
        modalNextImage(e);
    }

    setTimeout(() => {
        isScrolling = false;
    }, opts.js_modal_lightbox_gamepad_repeat);
});

function sleepUntil(f, timeout) {
    return new Promise((resolve) => {
        const timeStart = new Date();
        const wait = setInterval(function() {
            if (f() || new Date() - timeStart > timeout) {
                clearInterval(wait);
                resolve();
            }
        }, 20);
    });
}
