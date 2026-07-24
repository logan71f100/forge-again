var isSetupForMobile = false;

function isMobile() {
    for (var tab of ["txt2img", "img2img"]) {
        var imageTab = gradioApp().getElementById(tab + '_results');
        if (imageTab && imageTab.offsetParent && imageTab.offsetLeft == 0) {
            return true;
        }
    }

    return false;
}

function reportWindowSize() {
    if (gradioApp().querySelector('.toprow-compact-tools')) return; // not applicable for compact prompt layout

    var currentlyMobile = isMobile();
    if (currentlyMobile == isSetupForMobile) return;
    isSetupForMobile = currentlyMobile;

    for (var tab of ["txt2img", "img2img"]) {
        var button = gradioApp().getElementById(tab + '_generate_box');
        var target = gradioApp().getElementById(currentlyMobile ? tab + '_results' : tab + '_actions_column');
        var results = gradioApp().getElementById(tab + '_results');
        // img2img is a lazily-built tab: these are null until it's first opened.
        // On a narrow/mobile viewport the early-return above doesn't fire, so
        // without this guard the img2img iteration threw on insertBefore(null).
        // Skip that tab until it's built; reportWindowSize re-runs on resize.
        if (!button || !target || !results) continue;
        target.insertBefore(button, target.firstElementChild);
        results.classList.toggle('mobile', currentlyMobile);
    }
}

window.addEventListener("resize", reportWindowSize);

onUiLoaded(function() {
    reportWindowSize();
});
