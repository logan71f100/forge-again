// Highlight the input control that caused an error (issue #6).
//
// When a generation fails with a recognizable, attributable error, the backend
// (modules/call_queue.py + modules/error_tips.py) renders the error div with a
// data-field attribute carrying a CSS selector for the responsible control
// ("{tab}" is a placeholder for the active tab, e.g. txt2img). This script
// watches for those divs, resolves the selector against the active tab,
// expands any collapsed ancestor accordions, outlines the control and scrolls
// it into view. The outline clears when the user interacts with the control or
// when the next run starts (the old error div is removed from the DOM).

(function() {
    const HIGHLIGHT_CLASS = 'error-field-highlight';

    function activeTabName() {
        const content = get_uiCurrentTabContent();
        if (content && content.id && content.id.startsWith('tab_')) {
            return content.id.slice(4);
        }
        return 'txt2img';
    }

    function clearHighlights() {
        gradioApp().querySelectorAll('.' + HIGHLIGHT_CLASS).forEach(function(el) {
            el.classList.remove(HIGHLIGHT_CLASS);
        });
    }

    function expandAncestors(el) {
        // gradio accordions collapse their body; open every closed ancestor so
        // the highlighted control is actually visible. The toggle is a child
        // button/label with aria-expanded or .label-wrap(.open) depending on
        // the gradio component.
        for (let p = el.parentElement; p; p = p.parentElement) {
            const toggle = p.querySelector(':scope > button[aria-expanded="false"], :scope > .label-wrap:not(.open)');
            if (toggle && toggle.offsetParent !== null) toggle.click();
        }
    }

    function highlight(selectorTemplate) {
        const selector = selectorTemplate.replaceAll('{tab}', activeTabName());
        let target;
        try {
            target = gradioApp().querySelector(selector);
        } catch (e) {
            return; // malformed selector -- never break the error display
        }
        if (!target) return;
        expandAncestors(target);
        target.classList.add(HIGHLIGHT_CLASS);
        target.scrollIntoView({behavior: 'smooth', block: 'center'});
        const clear = function() {
            target.classList.remove(HIGHLIGHT_CLASS);
            target.removeEventListener('input', clear, true);
            target.removeEventListener('click', clear, true);
        };
        target.addEventListener('input', clear, true);
        target.addEventListener('click', clear, true);
    }

    onUiUpdate(function() {
        gradioApp().querySelectorAll('div.error[data-field]:not([data-field-handled])').forEach(function(div) {
            div.setAttribute('data-field-handled', '1');
            clearHighlights();
            highlight(div.getAttribute('data-field'));
        });
    });
})();
