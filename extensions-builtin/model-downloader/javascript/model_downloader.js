// Load the "top models" list automatically the first time the tab is opened —
// the fetch talks to huggingface.co, so it must not run at page load for
// users who never open this tab. (Bound to the tab button directly: the old
// get_uiCurrentTab()/onUiTabChange helpers predate gradio 6's role=tab markup.)
(function () {
    let loaded = false;
    onAfterUiUpdate(function () {
        for (const btn of gradioApp().querySelectorAll('#tabs [role=tab]:not([data-md-hooked])')) {
            if (btn.textContent.trim() !== 'Model Downloader') continue;
            btn.dataset.mdHooked = '1';
            btn.addEventListener('click', function () {
                if (loaded) return;
                setTimeout(function () {
                    const refresh = gradioApp().getElementById('model_downloader_refresh_top');
                    if (refresh) {
                        loaded = true;
                        refresh.click();
                    }
                }, 600);
            });
        }
    });
})();
