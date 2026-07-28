let promptTokenCountUpdateFunctions = {};

function update_txt2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("txt2img_token_button");
    update_token_counter("txt2img_negative_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

function update_img2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("img2img_token_button");
    update_token_counter("img2img_negative_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

function update_token_counter(button_id) {
    promptTokenCountUpdateFunctions[button_id]?.();
}


function recalculatePromptTokens(name) {
    promptTokenCountUpdateFunctions[name]?.();
}

function recalculate_prompts_txt2img() {
    // Called from Gradio
    recalculatePromptTokens('txt2img_prompt');
    recalculatePromptTokens('txt2img_neg_prompt');
    return Array.from(arguments);
}

function recalculate_prompts_img2img() {
    // Called from Gradio
    recalculatePromptTokens('img2img_prompt');
    recalculatePromptTokens('img2img_neg_prompt');
    return Array.from(arguments);
}

function setupTokenCounting(id, id_counter, id_button) {
    var prompt = gradioApp().getElementById(id);
    var counter = gradioApp().getElementById(id_counter);
    // gradio 6 puts a div.input-container between the label and the textarea,
    // so the old `> label > textarea` selector no longer matches
    var textarea = gradioApp().querySelector(`#${id} textarea`);

    // img2img prompts live in a lazily-built tab, so these are null at load.
    // Bail quietly; setupTokenCounting is retried once img2img mounts.
    if (!prompt || !counter || !textarea) {
        return;
    }

    if (counter.parentElement == prompt.parentElement) {
        return;
    }

    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = "relative";

    var func = onEdit(id, textarea, 800, function() {
        if (counter.classList.contains("token-counter-visible")) {
            gradioApp().getElementById(id_button)?.click();
        }
    });
    promptTokenCountUpdateFunctions[id] = func;
    promptTokenCountUpdateFunctions[id_button] = func;
}

function toggleTokenCountingVisibility(id, id_counter, id_button) {
    var counter = gradioApp().getElementById(id_counter);
    if (!counter) return;   // lazy tab not mounted yet

    counter.style.display = opts.disable_token_counters ? "none" : "block";
    counter.classList.toggle("token-counter-visible", !opts.disable_token_counters);
}

function runCodeForTokenCounters(fun) {
    fun('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button');
    fun('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
    fun('img2img_prompt', 'img2img_token_counter', 'img2img_token_button');
    fun('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');
}

onUiLoaded(function() {
    function wireUpTokenCounters() {
        runCodeForTokenCounters(setupTokenCounting);
        // visibility is normally applied by onOptionsChanged, which may have
        // already run before a counter mounted -- re-apply once opts exist
        if (Object.keys(opts).length) {
            runCodeForTokenCounters(toggleTokenCountingVisibility);
        }
    }
    wireUpTokenCounters();
    // gradio 6 mounts components progressively (the counters can appear after
    // #txt2img_prompt triggers uiLoaded), builds img2img lazily, and remounts a
    // tab's children on tab switches, so counters can (re)appear unwired at any
    // time. Keep observing; setupTokenCounting no-ops when already wired.
    new MutationObserver(wireUpTokenCounters).observe(gradioApp(), {childList: true, subtree: true});
});

onOptionsChanged(function() {
    runCodeForTokenCounters(toggleTokenCountingVisibility);
});
