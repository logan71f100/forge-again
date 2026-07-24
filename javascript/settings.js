let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_licenses: 1,
};

function settingsShowAllTabs() {
    gradioApp().querySelectorAll('#settings > div').forEach(function(elem) {
        if (settingsExcludeTabsFromShowAll[elem.id]) return;

        elem.style.display = "block";
    });
}

function settingsShowOneTab() {
    gradioApp().querySelector('#settings_show_one_page').click();
}

function setupSettingsSearch() {
    var edit = gradioApp().querySelector('#settings_search');
    var editTextarea = gradioApp().querySelector('#settings_search > label > input');
    var buttonShowAllPages = gradioApp().getElementById('settings_show_all_pages');
    var settings_tabs = gradioApp().querySelector('#settings div');
    // Settings is a lazily-built tab: none of the above exist at load, which is
    // why this used to throw. Bail until they're present; the observer below
    // re-runs it once the tab is built.
    if (!edit || !editTextarea || !buttonShowAllPages || !settings_tabs) return false;
    if (edit.dataset.forgeSearchWired) return true;
    edit.dataset.forgeSearchWired = '1';

    onEdit('settingsSearch', editTextarea, 250, function() {
        var searchText = (editTextarea.value || "").trim().toLowerCase();

        gradioApp().querySelectorAll('#settings > div[id^=settings_] div[id^=column_settings_] > *').forEach(function(elem) {
            var visible = elem.textContent.trim().toLowerCase().indexOf(searchText) != -1;
            elem.style.display = visible ? "" : "none";
        });

        if (searchText != "") {
            settingsShowAllTabs();
        } else {
            settingsShowOneTab();
        }
    });

    settings_tabs.insertBefore(edit, settings_tabs.firstChild);
    settings_tabs.appendChild(buttonShowAllPages);


    buttonShowAllPages.addEventListener("click", settingsShowAllTabs);
    return true;
}

onUiLoaded(function() {
    if (setupSettingsSearch()) return;
    // Settings tab not built yet -- wire the search box once it appears.
    var obs = new MutationObserver(function() {
        if (setupSettingsSearch()) obs.disconnect();
    });
    obs.observe(gradioApp(), {childList: true, subtree: true});
});


onOptionsChanged(function() {
    if (gradioApp().querySelector('#settings .settings-category')) return;

    var sectionMap = {};
    gradioApp().querySelectorAll('#settings > div > button').forEach(function(x) {
        sectionMap[x.textContent.trim()] = x;
    });

    opts._categories.forEach(function(x) {
        var section = localization[x[0]] ?? x[0];
        var category = localization[x[1]] ?? x[1];

        var span = document.createElement('SPAN');
        span.textContent = category;
        span.className = 'settings-category';

        var sectionElem = sectionMap[section];
        if (!sectionElem) return;

        sectionElem.parentElement.insertBefore(span, sectionElem);
    });
});

