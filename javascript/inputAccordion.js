function inputAccordionChecked(id, checked) {
    var accordion = gradioApp().getElementById(id);
    accordion.visibleCheckbox.checked = checked;
    accordion.onVisibleCheckboxChange();
}

function setupAccordion(accordion) {
    var labelWrap = accordion.querySelector('.label-wrap');
    var gradioCheckbox = gradioApp().querySelector('#' + accordion.id + "-checkbox input");
    if (!labelWrap || !gradioCheckbox) return false;   // not fully mounted yet — retry on a later UI update
    var extra = gradioApp().querySelector('#' + accordion.id + "-extra");
    var span = labelWrap.querySelector('span');
    // 'input-accordion-unlinked' (on the bridge checkbox block): the checkbox
    // enables/disables independently of the accordion's open state — used by
    // ControlNet units, where opening a unit to look at it must not enable it
    var checkboxBlock = gradioApp().getElementById(accordion.id + "-checkbox");
    var linked = !(checkboxBlock && checkboxBlock.classList.contains('input-accordion-unlinked'));

    var isOpen = function() {
        return labelWrap.classList.contains('open');
    };

    var observerAccordionOpen = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutationRecord) {
            accordion.classList.toggle('input-accordion-open', isOpen());

            if (linked) {
                accordion.visibleCheckbox.checked = isOpen();
                accordion.onVisibleCheckboxChange();
            }
        });
    });
    observerAccordionOpen.observe(labelWrap, {attributes: true, attributeFilter: ['class']});

    if (extra) {
        labelWrap.insertBefore(extra, labelWrap.lastElementChild);
    }

    accordion.onChecked = function(checked) {
        if (linked) {
            if (isOpen() != checked) {
                labelWrap.click();
            }
        } else {
            accordion.visibleCheckbox.checked = checked;
        }
    };

    // Open the accordion WITHOUT changing its checkbox value. Used by
    // session/profile restore: on a linked InputAccordion (Refiner, Hires fix)
    // a plain header click would ENABLE the feature just to peek inside it.
    // Unlinks permanently — same as any manual checkbox interaction would.
    accordion.expandOnly = function() {
        if (!isOpen()) {
            linked = false;
            labelWrap.click();
        }
    };

    var visibleCheckbox = document.createElement('INPUT');
    visibleCheckbox.type = 'checkbox';
    visibleCheckbox.checked = linked ? isOpen() : gradioCheckbox.checked;
    visibleCheckbox.id = accordion.id + "-visible-checkbox";
    visibleCheckbox.className = gradioCheckbox.className + " input-accordion-checkbox";
    span.insertBefore(visibleCheckbox, span.firstChild);

    accordion.visibleCheckbox = visibleCheckbox;
    accordion.onVisibleCheckboxChange = function() {
        if (linked && isOpen() != visibleCheckbox.checked) {
            labelWrap.click();
        }

        // native click, not checked+synthetic event: gradio 6's Svelte checkbox
        // only listens for real 'change' activation, so the old updateInput()
        // path updated the DOM but never the backend — a ControlNet unit toggled
        // off in the UI would still run server-side with its previous state
        if (gradioCheckbox.checked !== visibleCheckbox.checked) {
            gradioCheckbox.click();
        }
    };

    visibleCheckbox.addEventListener('click', function(event) {
        linked = false;
        event.stopPropagation();
    });
    visibleCheckbox.addEventListener('input', accordion.onVisibleCheckboxChange);
    return true;
}

// gradio 6 only mounts the active tab / open accordions, so input-accordions can
// appear long after page load (ControlNet units inside the ControlNet section,
// anything in a lazily-built tab). Set each one up when it first appears instead
// of only once at load; the data marker keeps this idempotent and cheap.
onAfterUiUpdate(function() {
    for (var accordion of gradioApp().querySelectorAll('.input-accordion:not([data-ia-setup])')) {
        if (setupAccordion(accordion)) {
            accordion.dataset.iaSetup = '1';
        }
    }
});
