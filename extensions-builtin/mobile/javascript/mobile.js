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

    for (var tab of ["txt2img", "img2img"]) {
        var button = gradioApp().getElementById(tab + '_generate_box');
        var target = gradioApp().getElementById(currentlyMobile ? tab + '_results' : tab + '_actions_column');
        var results = gradioApp().getElementById(tab + '_results');
        // img2img is a lazily-built tab: these are null until it's first opened.
        // Skip that tab until it's built; reportWindowSize re-runs on every UI
        // update, so it is picked up as soon as it mounts.
        if (!button || !target || !results) continue;
        // Decide per element rather than from a remembered flag: gradio 6
        // remounts a tab's children on tab switches, which puts the generate
        // box back where the python built it and drops the .mobile class.
        if (button.parentElement === target && results.classList.contains('mobile') === currentlyMobile) continue;
        target.insertBefore(button, target.firstElementChild);
        results.classList.toggle('mobile', currentlyMobile);
    }
}

window.addEventListener("resize", reportWindowSize);

onAfterUiUpdate(reportWindowSize);
