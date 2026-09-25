// Stable Diffusion WebUI - Bracket checker
// By Hingashi no Florin/Bwin4L & @akx
// Counts open and closed brackets (round, square, curly) in the prompt and negative prompt text boxes in the txt2img and img2img tabs.
// If there's a mismatch, the keyword counter turns red and if you hover on it, a tooltip tells you what's wrong.

function checkBrackets(textArea, counterElt) {
    var counts = {};
    (textArea.value.match(/[(){}[\]]/g) || []).forEach(bracket => {
        counts[bracket] = (counts[bracket] || 0) + 1;
    });
    var errors = [];

    function checkPair(open, close, kind) {
        if (counts[open] !== counts[close]) {
            errors.push(
                `${open}...${close} - Detected ${counts[open] || 0} opening and ${counts[close] || 0} closing ${kind}.`
            );
        }
    }

    checkPair('(', ')', 'round brackets');
    checkPair('[', ']', 'square brackets');
    checkPair('{', '}', 'curly brackets');
    counterElt.title = errors.join('\n');
    counterElt.classList.toggle('error', errors.length !== 0);
}

function setupBracketChecking(id_prompt, id_counter) {
    // gradio 6 puts a div.input-container between the label and the textarea,
    // so the old `> label > textarea` selector no longer matches
    var textarea = gradioApp().querySelector("#" + id_prompt + " textarea");
    var counter = gradioApp().getElementById(id_counter);

    if (!textarea || !counter) return;   // lazy tab (img2img) not mounted yet
    if (textarea.dataset.bracketChecker) return; // already wired this element
    textarea.dataset.bracketChecker = '1';

    // look the counter up on each keystroke: gradio 6 can remount a tab's
    // children on tab switches, which would leave a captured element stale
    textarea.addEventListener("input", function() {
        var elt = gradioApp().getElementById(id_counter);
        if (elt) checkBrackets(textarea, elt);
    });
    checkBrackets(textarea, counter);
}

// img2img is built lazily and gradio 6 remounts tab contents, so the prompt
// boxes can appear (or be replaced) at any time after load. Re-run on every
// UI update; setupBracketChecking no-ops once a textarea is wired.
onAfterUiUpdate(function() {
    setupBracketChecking('txt2img_prompt', 'txt2img_token_counter');
    setupBracketChecking('txt2img_neg_prompt', 'txt2img_negative_token_counter');
    setupBracketChecking('img2img_prompt', 'img2img_token_counter');
    setupBracketChecking('img2img_neg_prompt', 'img2img_negative_token_counter');
});
