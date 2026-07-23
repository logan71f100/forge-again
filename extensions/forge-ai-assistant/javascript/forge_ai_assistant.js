// Forge AI Assistant — floating chat panel wired to a local LLM (text-gen-webui).
// The LLM can read every visible control on the current tab, change values,
// trigger generation (juggling VRAM automatically), and look at images.

(function () {
    'use strict';

    let messages = [];        // OpenAI-format chat history (system prompt is rebuilt every call)
    let pendingImages = [];   // dataURLs attached to the next user message
    let busy = false;
    let statusTimer = null;
    let autoSee = true;   // auto-attach source/result images
    let lastSeen = {};    // hashes of images already in the chat, keyed per tab
    let detectionLocked = true;   // LOCKED BY DEFAULT — a good mask is protected until the AI proves it needs changing
    let sawMask = false;          // must inspect the mask (get_image mask) before unlocking detection
    let referenceImage = null;    // user-uploaded reference/target image, pinned in context
    let sentImageUrls = new Set(); // images already in the server's KV cache (survives the VRAM juggle) — status shows only NEW images as "reading"

    // start (or hibernate-wake) the LLM. A fresh process (pid in the response)
    // means the KV cache is cold — forget which images the server has seen.
    async function startLLM() {
        const st = await apiJSON('/forge-ai/textgen/start', { model: currentModel() }, 300000);
        if (st && st.pid) sentImageUrls = new Set();
        return st;
    }

    // Detection settings are the SAM search params — locking them prevents the
    // model from wrecking a good mask by re-tuning it (a repeated failure mode).
    function isDetectionControl(label) {
        return /detection prompt|box threshold|mask expand|sam|dino/i.test(label);
    }

    // Images and generation history are tracked PER TAB — switching tabs must
    // not make the other tab's gallery look like a brand-new result.
    function seenKey(which) { return currentTabName() + ':' + which; }

    function hashStr(s) {
        let h = 5381;
        for (let i = 0; i < s.length; i += 97) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
        return h + ':' + s.length;
    }

    // Capture source + result and return parts for any image not yet seen by the LLM.
    async function autoImageParts() {
        const parts = [];
        for (const which of ['source', 'result']) {
            let img = null;
            try { img = await captureImage(which); } catch (e) { /* ignore */ }
            if (!img) continue;
            const h = hashStr(img);
            if (h === lastSeen[seenKey(which)]) continue;
            lastSeen[seenKey(which)] = h;
            if (which === 'source') {
                // new source image: re-arm the lock (protect the default detection)
                // and remind the AI to verify before touching it
                detectionLocked = true;
                parts.push({ type: 'text', text: '[auto-attached: current source/inpaint image] (detection is LOCKED — verify the mask first; only unlock_detection if the gallery shows it is actually wrong.)' });
            } else {
                // a result we haven't seen = the user ran a generation themselves
                const tabNow = currentTabName();
                const rec = recordGeneration(snapshotSettings());
                const runEntry = settingsHistory.find(x => x.n === rec.n);
                if (runEntry) runEntry.img = img;
                parts.push({
                    type: 'text',
                    text: `[auto-attached: generation #${rec.n} result (run by the user, tab: ${tabNow})]` +
                        (rec.diff.length ? ` Settings changed vs the previous run: ${rec.diff.join('; ')}.` : '')
                });
            }
            parts.push({ type: 'image_url', image_url: { url: img } });
        }
        return parts;
    }

    // ------------------------------------------------------------- utils

    // Browsers throttle page timers hard in unfocused tabs (Chrome: down to
    // 1/minute), which stalls generation polling and LLM restarts when the
    // user switches away. Worker timers are exempt — route all sleeps there.
    let timerWorker = null;
    const timerCallbacks = new Map();
    let timerSeq = 0;
    try {
        const src = 'onmessage=e=>setTimeout(()=>postMessage(e.data.id),e.data.ms)';
        timerWorker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
        timerWorker.onmessage = (e) => {
            const cb = timerCallbacks.get(e.data);
            if (cb) { timerCallbacks.delete(e.data); cb(); }
        };
    } catch (err) {
        console.warn('[forge-ai] worker timers unavailable, falling back to page timers', err);
    }

    function sleep(ms) {
        if (!timerWorker) return new Promise(r => setTimeout(r, ms));
        return new Promise(r => {
            const id = ++timerSeq;
            timerCallbacks.set(id, r);
            timerWorker.postMessage({ id: id, ms: ms });
        });
    }

    // Every request gets a timeout — a hanging fetch (e.g. a chat request
    // orphaned by a mid-flight model unload) must fail loudly, never freeze
    // the queue forever.
    async function apiJSON(url, body, timeoutMs) {
        const ctl = new AbortController();
        const tmo = setTimeout(() => ctl.abort(), timeoutMs || 30000);
        try {
            const opt = body !== undefined
                ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctl.signal }
                : { signal: ctl.signal };
            const r = await fetch(url, opt);
            if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
            return await r.json();
        } catch (e) {
            if (e.name === 'AbortError') throw new Error(url + ' timed out after ' + Math.round((timeoutMs || 30000) / 1000) + 's');
            throw e;
        } finally {
            clearTimeout(tmo);
        }
    }

    function visible(el) {
        return el && el.offsetParent !== null && el.getBoundingClientRect().width > 0;
    }

    // Controls inside collapsed accordions (e.g. Replacer's "Advanced options")
    // are hidden but still fully settable — include them in the scan. Inactive
    // top-level sub-tabs (img2img vs inpaint vs batch) must stay excluded, but
    // unit tabs nested inside an accordion (ControlNet Unit 0/1/2) are included.
    function scannable(el) {
        if (!el || el.closest('#fai-panel')) return false;
        if (visible(el)) return true;
        let n = el.parentElement;
        while (n && n !== document.body) {
            if (getComputedStyle(n).display === 'none') {
                return !!n.closest('.gradio-accordion');
            }
            n = n.parentElement;
        }
        return false;
    }

    function inCollapsedSection(el) {
        return !visible(el);
    }

    // Name of the sub-tab (e.g. "ControlNet Unit 1") an element sits in, and
    // the nav button that activates it. Buttons and panes align by index.
    function tabInfoFor(el) {
        const ti = el.closest('.tabitem');
        if (!ti) return null;
        const tabs = ti.closest('.tabs');
        if (!tabs) return null;
        // gradio 4: every pane is mounted; buttons and panes align by index
        const nav = tabs.querySelector('.tab-nav');
        if (nav) {
            const items = [...tabs.children].filter(c => c.classList.contains('tabitem'));
            const btns = [...nav.querySelectorAll('button')];
            const idx = items.indexOf(ti);
            if (idx < 0 || !btns[idx]) return null;
            return {
                name: btns[idx].textContent.trim(),
                button: btns[idx],
                insideAccordion: !!ti.closest('.gradio-accordion'),
            };
        }
        // gradio 6: no .tab-nav; a pane's button can be found by the Forge/extension
        // id convention (#<paneId>-button), else: only the ACTIVE pane of a tab group
        // is mounted, so a mounted pane's button is the aria-selected one.
        const strip = (b) => b.closest('.tabs') === tabs;   // exclude nested tab strips
        let btn = null;
        if (ti.id) btn = gradioApp().getElementById(ti.id + '-button');
        if (!btn) btn = [...tabs.querySelectorAll('[role=tab]')].filter(strip)
            .find(b => b.getAttribute('aria-selected') === 'true');
        if (!btn) return null;
        return {
            name: btn.textContent.trim(),
            button: btn,
            insideAccordion: !!ti.closest('.gradio-accordion'),
        };
    }

    // Bring a control on-screen: expand its collapsed accordion and select its
    // sub-tab. Needed for dropdowns, whose option lists only render when shown.
    async function revealControl(el) {
        const acc = el.closest('.gradio-accordion');
        if (acc && !visible(el)) {
            const head = acc.querySelector('.label-wrap');
            if (head && !visible(el)) { head.click(); await sleep(300); }
        }
        if (!visible(el)) {
            const info = tabInfoFor(el);
            if (info) { info.button.click(); await sleep(300); }
        }
    }

    function activeTabRoot() {
        if (typeof get_uiCurrentTabContent === 'function') {
            const t = get_uiCurrentTabContent();
            if (t) return t;
        }
        return gradioApp();
    }

    // gradio 4 used  #tabs > .tab-nav button  with a .selected class; gradio 6 uses a
    // .tab-wrapper strip with role=tab + aria-selected. Support both, and exclude
    // nested tab strips (img2img sub-tabs, ControlNet units) by requiring the button's
    // enclosing .tabs container to be #tabs itself.
    function topTabButtons() {
        const tabs = gradioApp().getElementById('tabs');
        if (!tabs) return [];
        return [...tabs.querySelectorAll('[role=tab], .tab-nav button')]
            .filter(b => b.closest('.tabs, #tabs') === tabs);
    }

    function tabBtnSelected(b) {
        return b.classList.contains('selected') || b.getAttribute('aria-selected') === 'true';
    }

    function currentTabName() {
        const sel = topTabButtons().find(tabBtnSelected);
        return sel ? sel.textContent.trim() : 'unknown';
    }

    async function switchTab(tabName, subtabName) {
        if (tabName) {
            const btns = topTabButtons();
            const want = String(tabName).toLowerCase();
            const btn = btns.find(b => b.textContent.trim().toLowerCase() === want)
                || btns.find(b => b.textContent.trim().toLowerCase().includes(want));
            if (!btn) return `no top-level tab named "${tabName}" (tabs: ${btns.map(b => b.textContent.trim()).join(', ')})`;
            btn.click();
            await sleep(400);
        }
        if (subtabName) {
            const subBtns = [...activeTabRoot().querySelectorAll('.tab-nav button, [role=tab]')].filter(visible);
            const want = String(subtabName).toLowerCase();
            const sbtn = subBtns.find(b => b.textContent.trim().toLowerCase() === want)
                || subBtns.find(b => b.textContent.trim().toLowerCase().includes(want));
            if (!sbtn) return `no sub-tab named "${subtabName}" here (sub-tabs: ${subBtns.map(b => b.textContent.trim()).join(', ')})`;
            sbtn.click();
            await sleep(400);
        }
        return null;
    }

    // ------------------------------------------------------ control scan

    // Builds a registry of every visible input on the current tab:
    // sliders, number fields, textareas, dropdowns, checkboxes, radio groups.
    function scanControls() {
        const root = activeTabRoot();
        const controls = [];
        const seenBlocks = new Set();
        const seenLabels = {};

        function labelFor(block, fallback) {
            let txt = '';
            const lab = block.querySelector('label span, label');
            if (lab) txt = lab.textContent.trim().split('\n')[0].trim();
            if (!txt) txt = fallback || '';
            return txt;
        }

        // strip session-dependent decorations so labels stay STABLE across
        // boots: "ControlNet Unit 1[Depth]" -> "ControlNet Unit 1",
        // "ControlNet Integrated1 unit" (live badge) -> "ControlNet Integrated"
        function stableName(s) {
            return s ? s.replace(/\[[^\]]*\]/g, '').replace(/\d+\s*units?$/i, '').replace(/\s+/g, ' ').trim() : s;
        }

        function register(c) {
            if (!c.label) return;
            c.label = stableName(c.label);
            // prefix with section/unit so short labels like "Enable" are unambiguous:
            // "ControlNet Unit 1 > Enable", "Advanced options > Denoising", ...
            const acc = c.el && c.el.closest('.gradio-accordion');
            let accTitle = acc ? acc.querySelector('.label-wrap span')?.textContent.trim().split('\n')[0].trim() : null;
            accTitle = stableName(accTitle);
            const tabRaw = c.el ? tabInfoFor(c.el) : null;
            const tab = tabRaw && tabRaw.insideAccordion ? tabRaw : null;   // only unit-style tabs, not top-level ones
            const tabName = tab ? stableName(tab.name) : null;
            let prefix = null;
            if (tabName && accTitle && tabName.split(' ')[0] === accTitle.split(' ')[0]) prefix = tabName;
            else if (tabName) prefix = (accTitle ? accTitle + ' > ' : '') + tabName;
            else if (accTitle) prefix = accTitle;
            if (prefix && !c.label.startsWith(prefix)) c.label = prefix + ' > ' + c.label;
            const n = (seenLabels[c.label] || 0) + 1;
            seenLabels[c.label] = n;
            if (n > 1) c.label = c.label + ' #' + n;
            c.id = controls.length;
            c.hidden = c.el ? inCollapsedSection(c.el) : false;
            controls.push(c);
        }

        // sliders (range + paired number input). A block can hold TWO range
        // inputs (double-ended sliders like ControlNet's Timestep Range) —
        // register every thumb, not just the first one in the block.
        const rangeBlocks = [];
        root.querySelectorAll('input[type="range"]').forEach(r => {
            if (!scannable(r)) return;
            const blk = r.closest('.block') || r.parentElement;
            if (!rangeBlocks.includes(blk)) rangeBlocks.push(blk);
        });
        rangeBlocks.forEach(block => {
            if (seenBlocks.has(block)) return;
            seenBlocks.add(block);
            const ranges = [...block.querySelectorAll('input[type="range"]')].filter(scannable);
            const nums = [...block.querySelectorAll('input[type="number"]')];
            ranges.forEach((range, idx) => {
                const num = ranges.length === 1 ? nums[0] : nums[idx];
                let label = labelFor(block);
                if (ranges.length > 1) label += idx === 0 ? ' (start)' : (idx === 1 ? ' (end)' : ' #' + (idx + 1));
                register({
                    kind: 'slider',
                    el: range,
                    label: label,
                    min: range.min, max: range.max, step: range.step,
                    get: () => range.value,
                    set: (v) => {
                        range.value = v;
                        if (num) { num.value = v; updateInput(num); }
                        updateInput(range);
                    }
                });
            });
        });

        // standalone number inputs (seed etc.)
        root.querySelectorAll('input[type="number"]').forEach(num => {
            if (!scannable(num)) return;
            const block = num.closest('.block') || num.parentElement;
            if (seenBlocks.has(block)) return;
            seenBlocks.add(block);
            register({
                kind: 'number',
                el: num,
                label: labelFor(block, num.placeholder),
                get: () => num.value,
                set: (v) => { num.value = v; updateInput(num); }
            });
        });

        // textareas (prompts etc.)
        root.querySelectorAll('textarea').forEach(ta => {
            if (!scannable(ta)) return;
            const block = ta.closest('.block') || ta.parentElement;
            let label = labelFor(block, ta.placeholder);
            const bid = block.id || '';
            if (/neg/i.test(bid) || /negative/i.test(label)) label = label || 'Negative prompt';
            else if (/prompt/i.test(bid)) label = label || 'Prompt';
            register({
                kind: 'text',
                el: ta,
                label: label,
                get: () => ta.value,
                set: (v) => { ta.value = v; updateInput(ta); }
            });
        });

        // single-line text inputs (some extensions use these instead of textareas)
        root.querySelectorAll('input[type="text"]').forEach(inp => {
            if (!scannable(inp)) return;
            if (inp.closest('.gradio-dropdown, .dropdown')) return;   // dropdown filter box, not a field
            // image-editor internals (brush color "foreground"/"background",
            // upload paths) are not settings — keep them out of snapshots
            if (inp.closest('.image-container, .gradio-image, [data-testid*="image"]')) return;
            const block = inp.closest('.block') || inp.parentElement;
            if (seenBlocks.has(block)) return;
            seenBlocks.add(block);
            register({
                kind: 'text',
                el: inp,
                label: labelFor(block, inp.placeholder),
                get: () => inp.value,
                set: (v) => { inp.value = v; updateInput(inp); }
            });
        });

        // color pickers
        root.querySelectorAll('input[type="color"]').forEach(inp => {
            if (!scannable(inp)) return;
            if (inp.closest('.image-container, .gradio-image, [data-testid*="image"]')) return;
            const block = inp.closest('.block') || inp.parentElement;
            if (seenBlocks.has(block)) return;
            seenBlocks.add(block);
            register({
                kind: 'color',
                el: inp,
                label: labelFor(block),
                get: () => inp.value,
                set: (v) => { inp.value = v; updateInput(inp); }
            });
        });

        // gradio-4 dropdowns (custom combobox; multiselects like Styles show chips).
        // Matched STRUCTURALLY as well as by class — some builds don't put
        // .gradio-dropdown on the block, so also catch elem_id *_dropdown and
        // blocks with a .wrap-inner/ul.options combobox skeleton.
        root.querySelectorAll('.gradio-dropdown, .dropdown, [id*="_dropdown"], .block:has(.wrap-inner)').forEach(node => {
            const block = node.classList && node.classList.contains('block') ? node : (node.closest('.block') || node);
            if (!scannable(block) || seenBlocks.has(block)) return;
            // a WRAPPER block containing a dropdown deeper inside also matches
            // :has(.wrap-inner) — it would register as a phantom dropdown labeled by
            // whatever control happens to come first inside it. Only accept the block
            // that immediately owns the combobox skeleton.
            const wi = block.querySelector('.wrap-inner');
            if (wi && wi.closest('.block') !== block) return;
            const input = block.querySelector('input:not([type="checkbox"]):not([type="radio"]):not([type="number"]):not([type="range"]):not([type="color"])');
            if (!input) return;
            // must actually be a combobox, not a random block with an input
            if (!block.querySelector('.wrap-inner, ul.options')
                && input.getAttribute('role') !== 'listbox'
                && !/dropdown/i.test(block.id || '')
                && !block.classList.contains('gradio-dropdown')) return;
            seenBlocks.add(block);
            const chipTexts = () => [...block.querySelectorAll('.token')]
                .map(t => t.textContent.replace(/[×✕]/g, '').trim()).filter(Boolean);
            // gradio 6: the combobox only opens on keyboard interaction (synthetic
            // click/focus does nothing), and option labels carry a leading "✓ ".
            const openList = async () => {
                input.focus();
                input.click();
                if (input.getAttribute('aria-expanded') !== 'true') {
                    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
                }
                await sleep(150);
            };
            const optText = o => o.textContent.replace(/^[✓✔\s]+/, '').trim();
            const pickOption = async (o) => {
                o.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
                o.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
                o.click();
                await sleep(150);
            };
            register({
                kind: 'dropdown',
                el: input,
                // ControlNet's Preprocessor/Model dropdowns have no <label> —
                // fall back to aria/placeholder, else a positional name (the
                // #N dedupe keeps "…> Dropdown" / "…> Dropdown #2" stable)
                label: labelFor(block, input.getAttribute('aria-label') || input.placeholder || 'Dropdown'),
                get: () => {
                    const chips = chipTexts();
                    return chips.length ? chips.join(' || ') : input.value;
                },
                set: async (v) => {
                    // multiselect (chips): sync the chip set to the wanted list
                    if (String(v).includes(' || ') || chipTexts().length) {
                        await revealControl(input);
                        const want = String(v).split(' || ').map(s => s.trim()).filter(Boolean);
                        for (const tok of [...block.querySelectorAll('.token')]) {
                            const txt = tok.textContent.replace(/[×✕]/g, '').trim();
                            if (!want.includes(txt)) {
                                const x = tok.querySelector('.token-remove, button, svg');
                                if (x) { x.click(); await sleep(120); }
                            }
                        }
                        const have = chipTexts();
                        for (const item of want) {
                            if (have.includes(item)) continue;
                            await openList();
                            input.value = item;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            await sleep(250);
                            const opts = [...block.querySelectorAll('ul li, .options .item, [role="option"]')];
                            const hit = opts.find(o => optText(o) === item)
                                || opts.find(o => optText(o).toLowerCase() === item.toLowerCase())
                                || opts.find(o => optText(o).toLowerCase().includes(item.toLowerCase()));
                            if (hit) { await pickOption(hit); }
                        }
                        input.blur();
                        return;
                    }
                    // Reveal (accordion/unit tab), open, then TYPE the value to
                    // filter — long lists are virtualized, so the target option
                    // often isn't in the DOM until the list is filtered down.
                    await revealControl(input);
                    const want = String(v).toLowerCase().trim();
                    const findHit = () => {
                        const opts = [...block.querySelectorAll('ul li, .options .item, [role="option"]')];
                        return {
                            opts,
                            hit: opts.find(o => optText(o).toLowerCase() === want)
                                || opts.find(o => optText(o).toLowerCase().includes(want))
                                || opts.find(o => optText(o) && want.includes(optText(o).toLowerCase())),
                        };
                    };
                    await openList();
                    input.value = String(v);
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    await sleep(250);
                    let r = findHit();
                    if (!r.hit && want.length > 6) {
                        // retry with a short fragment in case the full string over-filters
                        input.value = String(v).split(/[\s\[_]/)[0].slice(0, 12);
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        await sleep(250);
                        r = findHit();
                    }
                    if (r.hit) {
                        await pickOption(r.hit);
                        return;
                    }
                    // nothing matched — close cleanly and tell the LLM what exists
                    const avail = r.opts.slice(0, 12).map(o => optText(o)).filter(Boolean);
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                    input.blur();
                    throw new Error(`no option matching "${v}"` +
                        (avail.length ? ` — visible options: ${avail.join(' | ')}` : ' (empty list — it may need its 🔄 refresh button)'));
                }
            });
        });

        // checkboxes
        root.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            if (!scannable(cb)) return;
            const block = cb.closest('label') || cb.closest('.block') || cb.parentElement;
            // InputAccordion header checkboxes (Hires fix, Refiner, ...) sit inside the
            // accordion's .label-wrap, whose full text includes session-dependent info
            // ("from 768x640 to 1536x1280") — label them by the accordion TITLE span so
            // the label is short and stable across sessions.
            const wrap = cb.closest('.label-wrap');
            let label;
            if (wrap) label = (wrap.querySelector('span')?.textContent || '').trim().split('\n')[0].trim();
            if (!label) label = (block.textContent || '').trim().split('\n')[0].trim();
            register({
                kind: 'checkbox',
                el: cb,
                label: label,
                get: () => cb.checked ? 'true' : 'false',
                set: async (v) => {
                    const want = (v === true || String(v).toLowerCase() === 'true');
                    // reveal collapsed section first so the toggle registers and
                    // any dependent sub-options appear (e.g. Soft inpainting)
                    if (!visible(cb)) { await revealControl(cb); }
                    if (cb.checked !== want) { cb.click(); await sleep(120); }
                    if (cb.checked !== want) {   // retry once if it didn't take
                        cb.checked = want;
                        cb.dispatchEvent(new Event('change', { bubbles: true }));
                        await sleep(120);
                    }
                }
            });
        });

        // radio groups (fieldsets: Masked content, Inpaint area, ...)
        root.querySelectorAll('fieldset').forEach(fs => {
            if (!scannable(fs)) return;
            const radios = [...fs.querySelectorAll('input[type="radio"]')];
            if (!radios.length) return;
            const block = fs.closest('.block') || fs;
            const options = radios.map(r => (r.closest('label')?.textContent || r.value).trim());
            register({
                kind: 'radio',
                el: fs,
                label: labelFor(block),
                options: options,
                get: () => {
                    const r = radios.find(x => x.checked);
                    return r ? (r.closest('label')?.textContent || r.value).trim() : '';
                },
                set: (v) => {
                    const want = String(v).toLowerCase();
                    const i = options.findIndex(o => o.toLowerCase() === want)
                        >= 0 ? options.findIndex(o => o.toLowerCase() === want)
                             : options.findIndex(o => o.toLowerCase().includes(want));
                    if (i >= 0) radios[i].closest('label')?.click() || radios[i].click();
                }
            });
        });

        return controls;
    }

    // ---------------------------------------- generation settings history

    // Snapshot of every control's value per generation, so the AI knows what
    // changed between results and can revert changes that made things worse.
    let genCounter = 0;
    const settingsHistory = [];   // [{n, snap: {label: value}}]

    function snapshotSettings() {
        const snap = {};
        for (const c of scanControls()) {
            let v = String(c.get() ?? '');
            if (v.length > 300) v = v.slice(0, 300);
            snap[c.label] = v;
        }
        return snap;
    }

    function diffSettings(prevSnap, curSnap) {
        const changes = [];
        for (const k of Object.keys(curSnap)) {
            if (prevSnap && k in prevSnap && prevSnap[k] !== curSnap[k]) {
                changes.push(`${k}: ${prevSnap[k]} → ${curSnap[k]}`);
            }
        }
        return changes;
    }

    function recordGeneration(snap) {
        genCounter++;
        const tab = currentTabName();
        const prev = [...settingsHistory].reverse().find(e => e.tab === tab);
        const diff = prev ? diffSettings(prev.snap, snap) : [];
        settingsHistory.push({ n: genCounter, tab: tab, snap: snap, diff: diff, verdict: null, note: '' });
        if (settingsHistory.length > 20) settingsHistory.shift();
        persistRuns();
        return { n: genCounter, diff: diff };
    }

    function persistRuns() {
        botAutosave();
        // images stay in-memory only (too big for sessionStorage)
        try {
            sessionStorage.setItem('fai_runs', JSON.stringify(
                { genCounter: genCounter, settingsHistory: settingsHistory },
                (k, v) => k === 'img' ? undefined : v
            ));
        } catch (e) { /* quota */ }
    }

    // Compact per-run history injected into every system prompt so the AI can
    // see the whole trajectory: what changed each run and how it was judged.
    function runLogSection() {
        if (!settingsHistory.length) return '';
        const rows = settingsHistory.slice(-12).map(e => {
            const changes = (e.diff && e.diff.length) ? e.diff.join('; ') : 'baseline (first run on this tab)';
            const verdict = e.verdict ? `${e.verdict.toUpperCase()}${e.note ? ' — ' + e.note : ''}` : 'NOT JUDGED YET';
            return `#${e.n} [${e.tab}] changed: ${changes} => ${verdict}`;
        });
        return 'RUN LOG — every generation, what changed, and your recorded verdict. Keep it accurate with {"tool":"verdict"}; fetch any run\'s full settings with {"tool":"get_settings","gen":N}:\n' + rows.join('\n');
    }

    // Restore the settings that produced generation #n (default: the one
    // before the latest — i.e. undo whatever the latest generation changed).
    async function revertTo(n) {
        let entry;
        if (n) entry = settingsHistory.find(s => s.n === Number(n));
        else entry = settingsHistory[settingsHistory.length - 2] || settingsHistory[settingsHistory.length - 1];
        if (!entry) return 'no settings snapshot available to revert to';
        if (entry.tab && entry.tab !== currentTabName()) {
            return `generation #${entry.n}'s settings belong to the "${entry.tab}" tab, but you are on "${currentTabName()}" — switch_tab there first, then revert`;
        }
        const controls = scanControls();
        let reverted = 0;
        for (const [label, val] of Object.entries(entry.snap)) {
            const c = controls.find(x => x.label === label);
            if (!c) continue;
            let cur = String(c.get() ?? '');
            if (cur.length > 300) cur = cur.slice(0, 300);
            if (cur !== val) {
                try {
                    await c.set(val);
                    reverted++;
                    sysMsg(`↩ ${label} → ${val}`);
                } catch (e) { /* skip unsettable */ }
            }
        }
        return reverted ? null : 'nothing differed from that snapshot';
    }

    // Split a prompt into normalized comma/newline phrases (strip weights/parens).
    function promptTerms(s) {
        return String(s || '').toLowerCase().split(/[,\n]/)
            .map(t => t.replace(/[()]/g, '').replace(/:\s*[\d.]+/g, '').trim())
            .filter(Boolean);
    }

    // Returns a reason string if the proposed prompt is a no-op vs the current
    // one (identical, reordered, or only re-adds words already present), else null.
    function promptRedundant(current, next) {
        if (String(current).trim().toLowerCase() === String(next).trim().toLowerCase()) return 'it is identical to the current prompt';
        const cur = new Set(promptTerms(current));
        const nxt = promptTerms(next);
        const added = [...new Set(nxt)].filter(t => !cur.has(t));
        const dupes = nxt.filter((t, i) => nxt.indexOf(t) !== i);
        if (!added.length) return 'every phrase you wrote is ALREADY in the current prompt — you added nothing new';
        if (dupes.length) return 'it repeats phrases that are already present: ' + [...new Set(dupes)].slice(0, 3).join(', ');
        return null;
    }

    function findControl(controls, label) {
        const want = String(label).toLowerCase();
        // "Section > Control" entries must never fuzzy-match the Section's own
        // toggle ("Hires. fix > Denoising strength" vs the "Hires. fix" checkbox)
        const sectionOf = want.split(' > ')[0].trim();
        const lastSeg = want.split(' > ').pop().trim();
        return controls.find(c => c.label.toLowerCase() === want)
            || controls.find(c => c.label.toLowerCase().includes(want))
            // reverse-substring is collision-prone with section-prefixed labels
            // ("Hires. fix > Resize width to" must NEVER match "Width") — only
            // allow it for labels long enough to be distinctive, and only when
            // the control label is a PREFIX of the final path segment: mid/
            // suffix hits are how "… Do not use detection prompt if …" once
            // wrote "true" into the Detection prompt box and "… > Override
            // positive prompt" wiped the main Positive prompt
            || controls.find(c => c.label.length >= 10
                && c.label.toLowerCase() !== sectionOf
                && lastSeg.startsWith(c.label.toLowerCase()));
    }

    function controlsListing(controls) {
        return controls.map(c => {
            let v = String(c.get() ?? '');
            if (v.length > 200) v = v.slice(0, 200) + '…';
            let extra = '';
            if (c.kind === 'slider') extra = ` (slider ${c.min}–${c.max}, step ${c.step})`;
            if (c.kind === 'radio') extra = ` (options: ${c.options.join(' | ')})`;
            if (c.kind === 'checkbox') extra = ' (true/false)';
            if (c.hidden) extra += ' [in a collapsed section — you can still set it]';
            return `- "${c.label}" = ${JSON.stringify(v)}${extra}`;
        }).join('\n');
    }

    // ------------------------------------------------------ image capture

    async function toDataURL(source, maxDim) {
        maxDim = maxDim || 1024;
        let w, h, draw;
        if (source instanceof HTMLCanvasElement) {
            w = source.width; h = source.height;
            draw = (ctx, W, H) => ctx.drawImage(source, 0, 0, W, H);
        } else {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = source.src; });
            w = img.naturalWidth; h = img.naturalHeight;
            draw = (ctx, W, H) => ctx.drawImage(img, 0, 0, W, H);
        }
        const scale = Math.min(1, maxDim / Math.max(w, h));
        const c = document.createElement('canvas');
        c.width = Math.round(w * scale); c.height = Math.round(h * scale);
        draw(c.getContext('2d'), c.width, c.height);
        return c.toDataURL('image/jpeg', 0.9);
    }

    // All images in the visible gallery (result + greyscale mask + masked
    // composite when the return-mask options are on), deduped, in order.
    async function captureGallery() {
        const root = activeTabRoot();
        const seen = new Set();
        const out = [];
        for (const im of [...root.querySelectorAll('[id*="gallery"] img')].filter(visible)) {
            if (seen.has(im.src)) continue;
            seen.add(im.src);
            try { out.push(await toDataURL(im, 768)); } catch (e) { /* skip broken */ }
            if (out.length >= 6) break;
        }
        return out;
    }

    async function captureImage(which) {
        const root = activeTabRoot();
        const area = el => { const r = el.getBoundingClientRect(); return r.width * r.height; };

        if (which === 'result') {
            const imgs = [...root.querySelectorAll('[id*="gallery"] img')].filter(visible);
            if (imgs.length) return toDataURL(imgs[0]);
        }
        // source: prefer the inpaint canvas
        const maskCanvases = [...root.querySelectorAll('[id*="maskimg"] canvas, [id*="image"] canvas')].filter(visible);
        const canvases = maskCanvases.length ? maskCanvases : [...root.querySelectorAll('canvas')].filter(visible);
        if (which === 'source' && canvases.length) {
            canvases.sort((a, b) => area(b) - area(a));
            return toDataURL(canvases[0]);
        }
        const anyImgs = [...root.querySelectorAll('img')].filter(v => visible(v) && !v.closest('#fai-panel'));
        if (anyImgs.length) {
            anyImgs.sort((a, b) => area(b) - area(a));
            return toDataURL(anyImgs[0]);
        }
        return null;
    }

    // ------------------------------------------------------ generation

    function findGenerateButton() {
        const root = activeTabRoot();
        let btn = [...root.querySelectorAll('button[id$="_generate"]')].find(visible);
        if (!btn) {
            btn = [...root.querySelectorAll('button')].find(b =>
                visible(b) && /^(generate|run)$/i.test(b.textContent.trim()));
        }
        return btn;
    }

    // Returns true if we actually observed the job running. Forge can spend
    // 60s+ loading model weights and running detection BEFORE progress
    // registers — declaring "done" too early restarts the LLM mid-generation
    // and the two fight over VRAM.
    async function waitForGeneration() {
        const started = Date.now();
        let sawActive = false;
        setActivity('🎨 waiting for the generation to start (Forge may be cold-loading its model)…');
        // phase 1: wait up to 240s for the job to appear. The FIRST generation
        // after a VRAM juggle makes Forge reload the whole checkpoint from disk,
        // which on a big SDXL/Flux model can take minutes.
        while (Date.now() - started < 240000) {
            try {
                const p = await apiJSON('/sdapi/v1/progress?skip_current_image=true');
                if (p.state && (p.state.job_count > 0 || p.progress > 0)) { sawActive = true; break; }
                const secs = Math.round((Date.now() - started) / 1000);
                if (secs >= 10) setActivity(`🎨 Forge loading its model / preparing… (${secs}s)`);
            } catch (e) { /* forge busy — keep waiting */ }
            await sleep(1000);
        }
        // phase 2: wait for it to finish (max 30 min); transient fetch errors
        // count as "still running", never as done
        while (Date.now() - started < 1800000) {
            try {
                const p = await apiJSON('/sdapi/v1/progress?skip_current_image=true');
                const active = p.state && (p.state.job_count > 0 || p.progress > 0);
                if (active) {
                    sawActive = true;
                    setActivity(`🎨 generating… ${Math.round((p.progress || 0) * 100)}%`);
                } else if (sawActive) {
                    setActivity('');
                    return true;
                } else {
                    setActivity('');
                    return false;   // 90s and it never started
                }
            } catch (e) { /* keep waiting */ }
            await sleep(1000);
        }
        setActivity('');
        return sawActive;
    }

    async function waitForLLM(timeoutMs) {
        const started = Date.now();
        while (Date.now() - started < (timeoutMs || 300000)) {
            try {
                const s = await apiJSON('/forge-ai/status');
                if (s.textgen_api_ready && s.model_loaded) { setActivity(''); return true; }
                const secs = Math.round((Date.now() - started) / 1000);
                if (s.textgen_api_ready) setActivity(`⏳ loading LLM model… (${secs}s)`);
                else if (s.textgen_proc) setActivity(`⏳ text-gen booting… (${secs}s)`);
                else setActivity(`⏳ waiting for text-gen… (${secs}s)`);
            } catch (e) { /* forge busy, keep polling */ }
            await sleep(2000);
        }
        setActivity('');
        return false;
    }

    // ----------------------------------------------- auto-restore watchdog

    // When the user runs a generation themselves, the server-side hook kills
    // text-gen to free VRAM and sets stopped_for_gen. This watchdog notices,
    // waits for the job to finish (two consecutive idle checks), and restarts
    // the LLM.
    let cycleInProgress = false;
    let restoring = false;
    let idleChecks = 0;
    let lastLoaded = null;   // tracks server-side restores so we can announce them

    async function watchdog() {
        try {
            const s = await apiJSON('/forge-ai/status');
            let genActive = false, genProgress = 0;
            try {
                const p = await apiJSON('/sdapi/v1/progress?skip_current_image=true');
                genActive = !!(p.state && (p.state.job_count > 0 || p.progress > 0));
                genProgress = p.progress || 0;
            } catch (e) { /* progress unavailable — treat as idle */ }

            // steady state bar
            if (genActive) setSteady(`🎨 Forge generating… ${Math.round(genProgress * 100)}%`);
            else if (s.sleeping) setSteady('🌙 LLM hibernated — VRAM free, memory intact (~1.5s wake)');
            else if (s.textgen_api_ready && s.model_loaded) setSteady('🟢 LLM running — ready');
            else if (s.stopped_for_gen && s.auto_restore) setSteady('⏸ LLM parked for generation — auto-restore pending');
            else if (s.textgen_api_ready) setSteady('🌙 server warm — model unloaded (fast start)');
            else if (s.textgen_proc) setSteady('⏳ llama-server starting…');
            else setSteady('⚪ LLM off');

            // chat notices on state transitions — but not during our own
            // cycles/turns, whose messages already narrate what's happening
            const loadedNow = !!(s.textgen_api_ready && s.model_loaded);
            if (!cycleInProgress && !restoring && !busy) {
                if (lastLoaded === false && loadedNow) sysMsg('LLM is back (auto-restored after your generation).');
                else if (lastLoaded === true && !loadedNow && s.stopped_for_gen) sysMsg('Your generation is using the VRAM — LLM parked, will auto-restore after.');
            }
            lastLoaded = loadedNow;

            // pull any live operator guidance (cheap, keeps takeover responsive)
            refreshGuidance();

            // stranded queued messages (e.g. after a transient failure): drain
            // whenever we're idle and not deliberately paused
            if (sendQueue.length && !busy && !restoring && !cycleInProgress && !queuePaused) processQueue();

            // the state bar above always updates; everything below only when idle
            if (cycleInProgress || restoring || busy) return;

            // client-side restore fallback (server usually beats us to it)
            if (!s.stopped_for_gen || !s.auto_restore) { idleChecks = 0; return; }
            if (genActive) { idleChecks = 0; return; }
            idleChecks++;
            if (idleChecks < 2) return;   // debounce: user may queue another gen
            idleChecks = 0;
            restoring = true;
            sysMsg('Your generation freed the VRAM — restarting the LLM…');
            await startLLM();
            const ok = await waitForLLM(300000);
            sysMsg(ok ? 'LLM is back. Send a message and I\'ll look at the new result.'
                      : 'LLM restart failed — check the text-gen console.');
        } catch (e) {
            /* forge busy or mid-restart — try again next tick */
        } finally {
            restoring = false;
        }
    }

    // ----------------------------------------------------- blind judging

    // The model rates its own changes too favorably when it judges inside the
    // working conversation. So every new result is compared against the best
    // so far by a SEPARATE clean-context request: two images in random order,
    // no knowledge of what changed. That verdict is authoritative.
    let bestResult = null;   // { url, gen }
    let tieStreak = 0;       // consecutive judge ties → escalate to the user
    let cnFailCount = 0;     // ControlNet-caused generation failures this session

    // was a ControlNet unit enabled at generation time?
    function anyControlNetEnabled() {
        return scanControls().some(c => /controlnet/i.test(c.label) && /enable/i.test(c.label) && String(c.get()).toLowerCase() === 'true');
    }

    function recentGoalText() {
        const texts = [];
        for (let i = messages.length - 1; i >= 0 && texts.length < 3; i--) {
            const m = messages[i];
            if (m.role !== 'user') continue;
            const t = Array.isArray(m.content)
                ? m.content.filter(p => p.type === 'text' && !p.text.startsWith('[')).map(p => p.text).join(' ')
                : String(m.content || '');
            if (t && !t.startsWith('[')) texts.unshift(t);
        }
        return texts.join(' | ').slice(0, 600) || 'improve the inpainted region realistically';
    }

    // Re-compress a dataURL to a smaller size — judge calls send 3 images at
    // once, and full-size images spike llama-server's vision memory enough to
    // crash it (observed: connection reset mid-judge).
    function shrinkDataUrl(url, maxDim, quality) {
        const q = quality || 0.85;
        return new Promise((resolve) => {
            const im = new Image();
            im.onload = () => {
                const scale = Math.min(1, maxDim / Math.max(im.width, im.height));
                if (scale >= 1) return resolve(url);
                const c = document.createElement('canvas');
                c.width = Math.round(im.width * scale);
                c.height = Math.round(im.height * scale);
                c.getContext('2d').drawImage(im, 0, 0, c.width, c.height);
                resolve(c.toDataURL('image/jpeg', q));
            };
            im.onerror = () => resolve(url);
            im.src = url;
        });
    }

    async function judgeOnce(newUrl, bestUrl, anchorImg, anchorIsReference, newIsA) {
        const content = [{
            type: 'text',
            text: 'You are a STRICT quality inspector comparing two attempts (A and B) at the same image edit. GOAL: "' + recentGoalText() + '".'
                + (anchorImg ? (anchorIsReference
                    ? ' The FIRST image is the user\'s REFERENCE — the TARGET to match. The winning attempt should look more like this reference (for the attributes in the goal) AND have fewer artifacts.'
                    : ' The first image is the ORIGINAL (before any edit) — reference only, do not judge it.') : '')
                + ' Look CLOSELY and SPECIFICALLY for ARTIFACTS in each attempt — examine the edited region and its edges carefully:'
                + ' • seams or hard edges where the edit meets the original'
                + ' • warped/asymmetric/deformed anatomy, wrong proportions'
                + ' • extra or missing body parts, fingers, limbs'
                + ' • mushy, smeared, plastic, or low-detail texture; blur'
                + ' • skin tone / lighting that does not match the surrounding body'
                + ' • melted, doubled, or nonsensical shapes.'
                + ' For EACH attempt, LIST the specific artifacts you actually see (say "none" only if truly clean). Then: an attempt WITH artifacts LOSES to one that is cleaner, even if both achieve the goal. Be decisive and critical — do not call things equal to avoid choosing. End EXACTLY with:\nWINNER: A or B or TIE\nREASON: name the deciding artifact/difference in one sentence'
        }];
        if (anchorImg) {
            content.push({ type: 'text', text: anchorIsReference ? 'REFERENCE (the target to match):' : 'ORIGINAL (reference):' });
            content.push({ type: 'image_url', image_url: { url: anchorImg } });
        }
        content.push({ type: 'text', text: 'Attempt A:' });
        content.push({ type: 'image_url', image_url: { url: newIsA ? newUrl : bestUrl } });
        content.push({ type: 'text', text: 'Attempt B:' });
        content.push({ type: 'image_url', image_url: { url: newIsA ? bestUrl : newUrl } });
        try {
            const resp = await apiJSON('/forge-ai/chat', {
                messages: [{ role: 'user', content: content }],
                max_tokens: 4000,   // Thinking model reasons before the WINNER line — don't cut it off
                temperature: 0.1,
                judge: true,        // routes to slot 1 — never evicts the conversation's cached images in slot 0
            }, 600000);
            const w = /WINNER:\s*(A|B|TIE)/i.exec(resp.reply || '');
            if (!w) return null;
            const reason = ((/REASON:\s*([\s\S]+)/i.exec(resp.reply || '') || [])[1] || '').trim().slice(0, 200);
            if (/tie/i.test(w[1])) return { pick: 'tie', reason };
            return { pick: (w[1].toUpperCase() === 'A') === newIsA ? 'new' : 'best', reason };
        } catch (e) {
            return null;
        }
    }

    // Two passes with swapped image order — vision models have strong position
    // bias (a logged session picked "image 2" five times out of five). Only an
    // order-independent agreement counts; disagreement means no real difference.
    async function blindJudge(newUrl) {
        if (!bestResult) return null;
        // The judge is a pure QUALITY inspector — it does NOT see the reference
        // (matching the reference is the main agent's job). Anchor = the source,
        // small, just so it knows what the original looked like.
        let sourceImg = null;
        try { sourceImg = await captureImage('source'); } catch (e) { /* none */ }
        if (sourceImg) sourceImg = await shrinkDataUrl(sourceImg, 512);
        // the two ATTEMPTS go near-full-res (1024) so artifacts are visible.
        const newSmall = await shrinkDataUrl(newUrl, 1024);
        const bestSmall = await shrinkDataUrl(bestResult.url, 1024);
        setActivity('⚖ blind-judging (pass 1 of 2)…');
        const p1 = await judgeOnce(newSmall, bestSmall, sourceImg, false, true);
        setActivity('⚖ blind-judging (pass 2 of 2)…');
        const p2 = await judgeOnce(newSmall, bestSmall, sourceImg, false, false);
        setActivity('');
        if (!p1 && !p2) return null;
        if (p1 && p2) {
            if (p1.pick === 'new' && p2.pick === 'new') return { outcome: 'better', reason: p1.reason };
            if (p1.pick === 'best' && p2.pick === 'best') return { outcome: 'worse', reason: p1.reason };
            return { outcome: 'tie', reason: 'order-swapped passes disagreed — no clear difference between the two' };
        }
        const p = p1 || p2;
        return { outcome: p.pick === 'new' ? 'better' : (p.pick === 'best' ? 'worse' : 'tie'), reason: p.reason };
    }

    // Read a real Forge/Python generation error from the UI (Gradio error
    // toasts, alert boxes). These signatures never appear in normal UI, so
    // they're a reliable failure signal — and the actual message tells the AI
    // exactly what went wrong (e.g. a wrong-arch ControlNet shape mismatch).
    const ERROR_SIG = /RuntimeError|OutOfMemoryError|CUDA (?:error|out of memory)|AssertionError|Traceback|shapes cannot be multiplied|mat1 and mat2|size mismatch|ValueError|TypeError|KeyError|Expected .* but got/;

    function scanForgeErrors() {
        const app = gradioApp();
        const nodes = app.querySelectorAll('.toast-wrap *, .toast-body, [class*="toast"], [role="alert"], .error, .gradio-html, #footer ~ *');
        const hits = [];
        nodes.forEach(n => {
            if (n.closest && n.closest('#fai-panel')) return;
            const t = (n.textContent || '').trim();
            if (t && t.length < 1200 && ERROR_SIG.test(t)) hits.push(t.replace(/\s+/g, ' '));
        });
        return hits;
    }

    function interpretForgeError(text) {
        if (/shapes cannot be multiplied|mat1 and mat2|size mismatch/.test(text)) {
            return 'This is a ControlNet ARCHITECTURE MISMATCH (SD1.5 ControlNet on an SDXL checkpoint, or vice versa) — the dims 768 vs 2048 confirm it. DISABLE the ControlNet unit (Enable=false) or pick the matching-arch model. Do NOT touch denoise/CFG.';
        }
        if (/out of memory|OutOfMemoryError|CUDA error/i.test(text)) {
            return 'Forge ran out of VRAM. Lower the resolution or disable ControlNet/extra features, then retry.';
        }
        return 'Fix the cause named in the error before regenerating; do not just tweak sliders.';
    }

    // Detect a broken/garbage result (near-uniform fill, or a saturated neon
    // wash — the signature of a misconfigured ControlNet) so we can handle it
    // as an ERROR before wasting judge calls on it. Returns a reason or null.
    function detectBrokenImage(dataUrl) {
        return new Promise((resolve) => {
            const im = new Image();
            im.onload = () => {
                try {
                    const n = 40;
                    const c = document.createElement('canvas');
                    c.width = n; c.height = n;
                    const ctx = c.getContext('2d');
                    ctx.drawImage(im, 0, 0, n, n);
                    const d = ctx.getImageData(0, 0, n, n).data;
                    let sum = 0, sum2 = 0, satHigh = 0, cnt = 0;
                    for (let i = 0; i < d.length; i += 4) {
                        const r = d[i], g = d[i + 1], b = d[i + 2];
                        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
                        sum += lum; sum2 += lum * lum; cnt++;
                        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
                        const sat = mx === 0 ? 0 : (mx - mn) / mx;
                        if (sat > 0.6 && mx > 120) satHigh++;
                    }
                    const mean = sum / cnt;
                    const std = Math.sqrt(sum2 / cnt - mean * mean);
                    if (std < 10) return resolve('the result is nearly a flat single color (no image detail) — the generation produced garbage, not a photo');
                    if (satHigh / cnt > 0.85) return resolve('the result is a saturated neon wash — the classic signature of a broken/misconfigured ControlNet');
                    resolve(null);
                } catch (e) { resolve(null); }
            };
            im.onerror = () => resolve(null);
            im.src = dataUrl;
        });
    }

    // Find the main Seed number field on the current tab (not variation/resize).
    function findSeedInput() {
        const root = activeTabRoot();
        return [...root.querySelectorAll('input[type="number"]')].find(inp => {
            if (!visible(inp) && !inCollapsedSection(inp)) return false;
            const block = inp.closest('.block') || inp.parentElement;
            const lab = (block && block.querySelector('label')?.textContent || '').toLowerCase().trim();
            return /(^|\b)seed\b/.test(lab) && !/variation|resize|extra/.test(lab);
        });
    }

    // Impose a FIXED seed before the first generation so every comparison is
    // apples-to-apples. Setting a concrete integer is bulletproof — it does not
    // depend on finding Forge's reuse-seed button (which the Replacer tab lacks).
    async function ensureFixedSeed() {
        const seedInput = findSeedInput();
        if (!seedInput) return ' NOTE: could not find the Seed field to lock — set Seed to a fixed integer manually so comparisons are valid.';
        const cur = String(seedInput.value ?? '').trim();
        if (cur !== '' && cur !== '-1') {
            return ` Seed is fixed at ${cur} — comparisons are valid. Set it to -1 only for a fresh composition.`;
        }
        const seed = Math.floor(Math.random() * 2147483647);
        seedInput.value = String(seed);
        updateInput(seedInput);
        await sleep(150);
        const now = String(seedInput.value ?? '').trim();
        if (now === String(seed)) {
            sysMsg(`🎲 seed locked to ${seed} — setting changes are now measurable`);
            return ` Seed is now LOCKED to ${seed}: future comparisons reflect only the setting change, not seed noise. Keep it fixed while tuning; set Seed to -1 for a fresh composition.`;
        }
        return ' WARNING: could not set a fixed seed — set the Seed field to an integer yourself before comparing settings.';
    }

    // The full VRAM-juggle cycle: stop LLM -> generate -> restart LLM -> show result.
    // Never throws: every failure becomes a [tool error] the LLM can react to.
    async function generateCycle() {
        cycleInProgress = true;
        try {
            return await generateCycleInner();
        } catch (e) {
            sysMsg('Generate cycle error: ' + e.message);
            return [{ type: 'text', text: '[tool error] the generate cycle failed: ' + e.message + '. The generation may or may not have completed. Diagnose before retrying — do not just repeat the same action.' }];
        } finally {
            cycleInProgress = false;
        }
    }

    async function generateCycleInner() {
        // fingerprint the current result so we can tell if generation produced anything
        let before = null;
        try { before = await captureImage('result'); } catch (e) { /* no result yet */ }

        // lock the seed BEFORE the first generation so the baseline and every
        // iteration share one seed — otherwise setting comparisons are noise.
        let seedNote = '';
        if (!bestResult) seedNote = await ensureFixedSeed();

        const snapAtClick = snapshotSettings();   // the settings this generation will use

        // identical settings + fixed seed = the exact same image; catch it
        // before wasting a whole VRAM juggle (observed live: post-revert regen)
        const lastEntry = settingsHistory[settingsHistory.length - 1];
        if (lastEntry && lastEntry.tab === currentTabName() && !diffSettings(lastEntry.snap, snapAtClick).length) {
            const seedKey = Object.keys(snapAtClick).find(k => /(^|> )Seed(\s|$)/i.test(k));
            const seedVal = seedKey ? String(snapAtClick[seedKey]).trim() : '-1';
            if (seedVal !== '' && seedVal !== '-1') {
                return [{ type: 'text', text: `[tool error] settings are IDENTICAL to generation #${lastEntry.n} and the seed is fixed (${seedVal}) — generating would reproduce the exact same image. Change a setting first, or set Seed to -1 for variation. Note: after a revert, settings already match that generation.` }];
            }
        }

        // Claude (API or Claude Code) is off-GPU, so no VRAM juggle: Forge keeps
        // its model loaded and we generate directly. Local model needs the juggle.
        const juggle = provider === 'local';
        if (juggle) {
            sysMsg('Stopping LLM to free VRAM for Forge…');
            setActivity('♻ freeing VRAM for Forge…');
            await apiJSON('/forge-ai/textgen/stop', {}, 90000);
            await sleep(2000);
        }

        const btn = findGenerateButton();
        if (!btn) {
            sysMsg('Could not find a Generate button on this tab.');
            return [{ type: 'text', text: '[tool error] no Generate button found on the current tab.' }];
        }
        const errsBefore = new Set(scanForgeErrors());
        btn.click();
        sysMsg(juggle ? 'Generating… (LLM is offline meanwhile)' : 'Generating…');
        const jobRan = await waitForGeneration();
        await sleep(1500);

        // ERROR HANDLING FIRST: if Forge threw a Python error, the generation
        // failed — read the real message and stop, before any judging.
        const newErrs = scanForgeErrors().filter(e => !errsBefore.has(e));
        if (newErrs.length) {
            const err = newErrs[0].slice(0, 400);
            if (anyControlNetEnabled() || /controlnet|mat1 and mat2|shapes cannot/i.test(err)) cnFailCount++;
            sysMsg('⚠ Forge error: ' + err.slice(0, 120));
            await startLLM();
            await waitForLLM(300000);
            return [{ type: 'text', text: `[tool error] the generation FAILED with a Forge error (not judged, not recorded): "${err}". ${interpretForgeError(newErrs[0])}` }];
        }

        // Poll the result gallery to SETTLE — its DOM can update a beat after the
        // job finishes, so a single early capture may grab the stale previous
        // image and be misread as "no change / failed". Retry until it differs
        // from `before` or we've given it enough time.
        let img = null;
        for (let i = 0; i < 8; i++) {
            try { img = await captureImage('result'); } catch (e) { /* retry */ }
            if (img && (!before || hashStr(img) !== hashStr(before))) break;
            await sleep(800);
        }
        const changed = img && (!before || hashStr(img) !== hashStr(before));

        if (!jobRan && !img) {
            sysMsg('⚠ The generation never started within 240s.');
            if (juggle) { await startLLM(); await waitForLLM(300000); }
            return [{ type: 'text', text: '[tool error] the generation never started (no job appeared within 240s and there is no image). The Generate click may not have registered — check the Forge tab.' }];
        }

        if (juggle) {
            sysMsg('Generation done. Reloading the LLM (warm reload — should be quick)…');
            await startLLM();
            const ok = await waitForLLM(300000);
            if (!ok) { sysMsg('LLM did not come back up in time. Check the text-gen console.'); return null; }
            sysMsg('LLM is back — analyzing the result.');
        } else {
            sysMsg('Generation done — analyzing the result…');
        }

        // A completed job (no Forge error above) with a real image is a SUCCESS.
        // Only genuinely-empty gallery is a failure — the fragile hash-compare is
        // NO LONGER used to declare failure (it caused false "generation failed").
        if (!img) {
            const cnEnabled = anyControlNetEnabled();
            if (cnEnabled) cnFailCount++;
            return [{ type: 'text', text: '[tool error] the gallery has no result image after the job ran.' + (cnEnabled ? ' A ControlNet unit is enabled — disable it and retry.' : ' Check the Forge tab for an error.') }];
        }

        // broken/garbage result (neon or flat) — a misconfigured ControlNet
        const broken = await detectBrokenImage(img);
        if (broken) {
            sysMsg('⚠ broken result detected — not judging it');
            const cnEnabled = scanControls().filter(c => /controlnet/i.test(c.label) && /enable/i.test(c.label) && String(c.get()).toLowerCase() === 'true');
            const cnHint = cnEnabled.length
                ? ` A ControlNet unit is ENABLED (${cnEnabled.map(c => (c.label.match(/unit\s*\d+/i) || [''])[0]).join(', ')}) — DISABLE it now (set Enable false) and regenerate. This is almost certainly the cause.`
                : ' Check the prompt/settings for what could produce a flat or neon image.';
            return [{ type: 'text', text: `[tool error] the generation produced a BROKEN image: ${broken}. It was NOT judged and NOT recorded as a result.${cnHint} Do not tune denoise/CFG — fix the cause first.` }];
        }

        lastSeen[seenKey('result')] = hashStr(img);
        const rec = recordGeneration(snapAtClick);
        // seedNote was set before the generation (ensureFixedSeed)

        // independent blind comparison against the best result so far
        let judgeLine = '';
        const entry = settingsHistory.find(x => x.n === rec.n);
        if (entry) entry.img = img;   // kept so the user can re-point "best" later
        if (!bestResult) {
            bestResult = { url: img, gen: rec.n };
            judgeLine = ` This is the baseline result (best so far: #${rec.n}).` + seedNote;
        } else {
            const j = await blindJudge(img);
            if (j) {
                if (entry) {
                    entry.verdict = j.outcome === 'tie' ? 'mixed' : j.outcome;
                    entry.note = ('blind judge: ' + j.reason).slice(0, 150);
                    persistRuns();
                }
                if (j.outcome === 'better') {
                    sysMsg(`⚖ blind judge: #${rec.n} BEATS #${bestResult.gen} — ${j.reason}`);
                    judgeLine = ` INDEPENDENT BLIND JUDGE (authoritative, already recorded): this result BEATS the previous best (${j.reason}). Best so far is now #${rec.n}. Continue refining from here.`;
                    bestResult = { url: img, gen: rec.n };
                } else if (j.outcome === 'worse') {
                    sysMsg(`⚖ blind judge: #${rec.n} is WORSE than best #${bestResult.gen} — ${j.reason}`);
                    judgeLine = ` INDEPENDENT BLIND JUDGE (authoritative, already recorded): this result is WORSE than best generation #${bestResult.gen} (${j.reason}). Do NOT rationalize it as better. Use {"tool":"revert","to":${bestResult.gen}} and try a different, smaller change.`;
                } else {
                    tieStreak++;
                    sysMsg(`⚖ blind judge: #${rec.n} ≈ tie with best #${bestResult.gen}${tieStreak >= 2 ? ' (judge is stuck — ask the user)' : ''}`);
                    if (tieStreak >= 2) {
                        judgeLine = ` INDEPENDENT BLIND JUDGE: TIE again (#${tieStreak} in a row) — the judge CANNOT resolve this difference. STOP relying on it. Show the user this result and ASK them which is better, starting your reply with "NEED INPUT:" and a specific question (e.g. "is #${rec.n} better than #${bestResult.gen} for roundness?"). Use their answer as the authoritative verdict and best.`;
                    } else {
                        judgeLine = ` INDEPENDENT BLIND JUDGE: essentially a TIE with best #${bestResult.gen} (${j.reason}). The last change did not clearly help; try a DIFFERENT axis (a different lever than last time).`;
                    }
                }
                if (j.outcome !== 'tie') tieStreak = 0;
            } else {
                judgeLine = ` (Blind judge unavailable — evaluate yourself and record a verdict with {"tool":"verdict","gen":${rec.n},...}; be skeptical of your own change.)`;
            }
        }

        const tooMany = rec.diff.length > 2
            ? ` ⚠ You changed ${rec.diff.length} settings at once — you CANNOT tell which one caused this result. Change only 1-2 settings per generation so each change is measurable.`
            : '';
        const identical = !changed
            ? ` Note: this result looks identical to the previous one — if you changed a setting it may not have registered (verify the CONTROLS list), or you regenerated the same settings+seed. It is still recorded as generation #${rec.n}.`
            : '';
        return [
            {
                type: 'text',
                text: `[tool result] Generation #${rec.n} finished — image attached.` +
                    (rec.diff.length ? ` Settings changed vs the previous run: ${rec.diff.join('; ')}.` : '') +
                    tooMany + identical + judgeLine
            },
            { type: 'image_url', image_url: { url: img } },
        ];
    }

    // ------------------------------------------------- checkpoint notes

    // Author-recommended settings per checkpoint, distilled from Civitai pages
    // and author notes. Matched against the loaded checkpoint name; the hit is
    // injected into the system prompt so the AI tunes for THIS model.
    const MODEL_NOTES = [
        {
            match: 'cyberrealistic_v80inpainting',
            notes: 'CyberRealistic v8 Inpainting (SD1.5, purpose-built for inpainting). DPM++ SDE Karras or DPM++ 2M Karras, ~30 steps, CFG 5, clip skip 1, 512x768 base. Natural short prompts, no quality-tag spam; neg: "worst quality, low quality, bad anatomy, bad hands, watermark, blurry". Denoise ~0.3 for refinement, 0.6-0.75 only when replacing content; keep masked region near 512px. VAE baked in — never load an external VAE. CFG above ~7 degrades realism.'
        },
        {
            match: 'cyberrealistic_v90',
            notes: 'CyberRealistic v9 (SD1.5). DPM++ SDE Karras or DPM++ 2M Karras, 30 steps, CFG 5, clip skip 1, 512x768 portrait native. Simple natural prompts, no "masterpiece/8k" prefixes; neg: "lowres, bad anatomy, bad hands, text, watermark, worst quality, low quality, jpeg artifacts, blurry". Keep CFG ~5 — higher loses realism. VAE baked in. img2img/hires denoise ~0.3.'
        },
        {
            match: 'epicrealism_pureevolution',
            notes: 'epiCRealism pureEvolution V5 (SD1.5). DPM++ SDE Karras for max realism / DPM++ 2M Karras for softer look, 20-30 steps, CFG 5, clip skip 1, 512x768. Short simple prompts — author says quality tags are unnecessary; neg: "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)". CFG above 5-7 loses realism. img2img denoise ~0.35; artifacts → raise steps.'
        },
        {
            match: 'v1-5-pruned',
            notes: 'Vanilla SD 1.5 base. Euler a or DPM++ 2M Karras, 20-30 steps, CFG 7-7.5, 512x512 native (subject duplication above ~576px/side). Natural-language captions. NOT an inpainting model — masked inpaint at high denoise makes seams; suggest a dedicated inpainting checkpoint instead. Weak hands/faces vs finetunes.'
        },
        {
            match: 'cyberrealisticpony',
            notes: 'CyberRealistic Pony (SDXL Pony). DPM++ SDE Karras / DPM++ 2M Karras / Euler a, 30+ steps, CFG 5, clip skip 2, 896x1152 or 832x1216. Tag-style prompting with positive prefix "score_9, score_8_up, score_7_up," (recommended, optional); no source_/rating_ tags needed. Neg: "(worst quality:1.2), (low quality:1.2), (normal quality:1.2), lowres, bad anatomy, bad hands, watermarks". Keep CFG ~5. VAE baked in.'
        },
        {
            match: 'ponyrealism',
            notes: 'Pony Realism (SDXL Pony). Euler a or DPM2 a (DPM2 a best for detail) — author says AVOID DPM++ 2M Karras. 30+ steps, CFG 6-7, clip skip 2, >1024px. Danbooru tag style with prefix "score_9, score_8_up, score_7_up, BREAK"; use "female/male" not "woman/man". Neg: "score_4, score_5, score_6". Inpainting: author recommends denoise 0.77-0.79, mask blur 6. Keep prompt weights <=1.5.'
        },
        {
            match: 'epicrealismxl',
            notes: 'epiCRealism XL vXVII Crystal Clear (SDXL). DPM++ 2M SDE + Exponential, 30 steps, CFG 5, no clip skip, 1024x1024 / 832x1216. Natural-language photo prompts, no prefix; author advises starting with an EMPTY negative prompt on SDXL. Inpainting: works best with small denoise and multiple passes; poor at large-area replacement.'
        },
        {
            match: 'realvisxl',
            notes: 'RealVisXL V5 (SDXL). DPM++ SDE Karras 30+ steps or DPM++ 2M Karras 50+, CFG ~5-7, no clip skip, VAE baked. Natural language, photorealism focus. Neg: "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth". Hires: denoise 0.1-0.3, 1.1-1.5x.'
        },
        {
            match: 'juggernautxl',
            notes: 'Juggernaut XL (SDXL). DPM++ 2M SDE, 30-40 steps, CFG 3-6 (author: "less is more realistic"), no clip skip, 832x1216 portrait. Natural language; booru tags also work. Author: start with NO negative, add only what you actually see and don\'t want. Hires: denoise 0.3, 1.5x. Weak at text and distant faces.'
        },
    ];

    // User-defined per-checkpoint notes from Settings > AI Assistant (JSON array
    // of {match, notes}); kept out of the public code so personal model choices
    // stay in the local config. Malformed input is ignored.
    function userCheckpointNotes() {
        try {
            const raw = (window.opts && opts.forge_ai_checkpoint_notes) || '';
            if (!String(raw).trim()) return [];
            const arr = JSON.parse(raw);
            return Array.isArray(arr) ? arr.filter(x => x && x.match && x.notes) : [];
        } catch (e) { return []; }
    }

    function checkpointNotes(ckpt) {
        const c = (ckpt || '').toLowerCase();
        // user notes take priority over the built-in public defaults
        const hit = userCheckpointNotes().concat(MODEL_NOTES)
            .find(m => c.includes(String(m.match).toLowerCase()));
        return hit ? hit.notes : null;
    }

    // ---------------------------------------------------- learned memory

    // Notes the AI wrote for itself in past sessions (what worked, what
    // didn't) — persisted server-side, injected into every system prompt.
    let aiMemory = [];

    async function refreshMemory() {
        try { aiMemory = (await apiJSON('/forge-ai/memory')).notes || []; } catch (e) { /* keep old */ }
    }

    // Live guidance the operator (you, via this session, or the user) can set
    // mid-run to steer or take over the assistant — injected at the TOP of the
    // system prompt with highest priority. Cleared = normal behavior resumes.
    let liveGuidance = '';
    async function refreshGuidance() {
        try {
            const g = (await apiJSON('/forge-ai/guidance')).guidance || '';
            if (g !== liveGuidance) {
                liveGuidance = g;
                if (g) sysMsg('📌 live guidance updated by operator');
            }
        } catch (e) { /* keep old */ }
    }

    function memorySection() {
        if (!aiMemory.length) return '';
        const recent = aiMemory.slice(-40);
        return 'LEARNED NOTES from your past sessions (trust these — you wrote them when something was confirmed to work):\n'
            + recent.map(n => `- [${n.ts}${n.checkpoint ? ' | ' + n.checkpoint : ''}${n.tab ? ' | ' + n.tab : ''}] ${n.note}`).join('\n');
    }

    // Checkpoint architecture — from Forge's mode preset, falling back to the
    // checkpoint name. Drives which ControlNet models are allowed.
    let forgePreset = null;   // 'sd' | 'xl' | 'flux', fetched once at load
    let provider = 'local';   // 'local' (text-gen) | 'claude' (Anthropic API)

    function archOfCheckpoint() {
        if (forgePreset === 'sd' || forgePreset === 'xl' || forgePreset === 'flux') return forgePreset;
        const ck = (gradioApp().querySelector('#setting_sd_model_checkpoint input')?.value || '').toLowerCase();
        if (/flux|gguf/.test(ck)) return 'flux';
        if (/xl|pony|illustrious/.test(ck)) return 'xl';
        return 'sd';
    }

    const CN_MODELS = {
        xl: 'xinsir-canny-sdxl (canny), xinsir-depth-sdxl (depth), controlnet++_openpose_sdxl (pose), Kataragi_inpaintXL (inpaint), controlnet-union-promax-sdxl_… (universal — one model for canny/depth/openpose/tile/scribble/lineart/softedge/seg/normal/inpaint; its long filename lists them, match it by the "union-promax" part)',
        sd: 'control_v11p_sd15_canny, control_v11f1p_sd15_depth, control_v11p_sd15_openpose, control_v11p_sd15_inpaint, control_v11f1e_sd15_tile, control_v11p_sd15_normalbae, controlnet++_canny/depth/seg_sd15',
    };

    // Returns an error string if this value is a wrong-architecture ControlNet
    // model for the current checkpoint, else null.
    function cnArchViolation(value) {
        const arch = archOfCheckpoint();
        const v = String(value).toLowerCase();
        const looksSd15 = /sd15|v11/.test(v);
        const looksXl = /xl|union|kataragi/.test(v);
        if (arch === 'flux') return 'no Flux ControlNet models are installed — do not enable ControlNet in flux mode.';
        if (arch === 'xl' && looksSd15) return `"${value}" is an SD1.5 ControlNet but the current checkpoint is SDXL. Use one of: ${CN_MODELS.xl}.`;
        if (arch === 'sd' && looksXl && !looksSd15) return `"${value}" is an SDXL ControlNet but the current checkpoint is SD1.5. Use one of: ${CN_MODELS.sd}.`;
        return null;
    }

    // ------------------------------------------------------ agent loop

    function systemPrompt(controls) {
        const ckpt = gradioApp().querySelector('#setting_sd_model_checkpoint input')?.value || 'unknown';
        return [
            (liveGuidance ? '★★★ OPERATOR LIVE DIRECTIVE (highest priority — overrides your other instincts, follow it exactly): ' + liveGuidance + ' ★★★' : ''),
            (referenceImage ? 'A REFERENCE IMAGE is loaded (labeled [REFERENCE IMAGE] in the conversation). Treat it as the TARGET the user wants to move toward — compare your results to it for whatever the user specified (pose, lighting, style, anatomy, colour) and steer edits to match it.' : ''),
            'You are an assistant embedded in Stable Diffusion Forge, helping the user get the image they want. You work on whichever tab is active: txt2img (generate from a prompt), img2img (transform an input image), or Replacer (prompt-guided inpainting/replacement). Detection/mask and Replacer-specific instructions in this prompt apply ONLY when the current tab is Replacer; on txt2img/img2img just tune the prompt and the standard generation settings.',
            'You can change the UI controls yourself. To use a tool, output a fenced block exactly like:',
            '```json',
            '{"tool":"set","label":"Denoising strength","value":0.55}',
            '```',
            'Tools:',
            '- {"tool":"set","label":"<exact label from CONTROLS>","value":<value>} — change a control. Use several blocks for several changes.',
            '- {"tool":"get_image","which":"result"} — re-fetch an image ("source" = the input/inpaint image). Usually unnecessary: the source image and latest result are attached to the user\'s messages automatically whenever they change.',
            '- {"tool":"get_image","which":"gallery"} — fetch ALL gallery images: the result, then the greyscale selection MASK, then the masked composite showing exactly which region was inpainted. USE THIS to verify the mask/selection is correct.',
            '- {"tool":"get_image","which":"mask"} — fetch ONLY the mask + composite (works even while detection is locked). REQUIRED before you can unlock_detection: verify the mask actually misses an area before touching detection.',
            '- {"tool":"generate"} — run the generation. This temporarily parks you to free VRAM and brings you back with the result attached. USE IT PROACTIVELY: settings changes accomplish nothing until a generation runs.',
            '- {"tool":"revert"} — restore the settings that produced the PREVIOUS generation (undo what the latest generation changed). {"tool":"revert","to":3} restores the exact settings of generation #3. Use this whenever a result got WORSE.',
            '- {"tool":"lock_detection"} / {"tool":"unlock_detection"} — protect/release the mask detection settings. Lock the instant the mask is verified good.',
            '- {"tool":"remember","note":"<short reusable insight>"} — save a note to your permanent memory for future sessions. Use when the user CONFIRMS something worked ("that\'s great") or you discover a reliable model/workflow insight. Write it general and reusable ("epicrealismXL inpainting: denoise 0.75 + Kataragi_inpaintXL = clean blends"), never session chatter.',
            '- {"tool":"verdict","gen":4,"verdict":"better|worse|mixed","note":"<why>","best":3} — record a judgement in the RUN LOG. The optional "best" field re-points the best-so-far baseline that blind judging compares against.',
            '- {"tool":"get_settings","gen":3} — fetch the complete settings snapshot used for generation #3.',
            '- {"tool":"switch_tab","tab":"txt2img","subtab":"..."} — go to another UI tab (subtab optional). You have full access to every tab: txt2img, img2img (with its Inpaint/Sketch sub-tabs), Replacer, Extras, etc. Work on whichever tab the user is using or asks for. Only switch when the user wants a different tab than the current one.',
            '- {"tool":"click","label":"<button text or tooltip>"} — click a UI button, e.g. "Send image and generation parameters to img2img inpaint tab" under a result gallery to carry the image over before switching tabs.',
            'THINK BEFORE YOU ACT. Before ANY tool call, write a short analysis in these exact labeled steps, then the tool block(s):',
            'OBSERVE: describe what the LATEST result image actually shows, concretely, vs the user\'s goal — what is already good, and the single most important thing still wrong.',
            'DIAGNOSE: name the ROOT CAUSE of that one problem and which specific lever controls it (prompt for content/shape, Inpaint area/padding for proportions, denoise for how-much-changes, CFG for prompt adherence, mask for selection). Refer to the RUN LOG: what have you already tried, and did the judge say it helped?',
            'PLAN: state the ONE change you will make, why it targets the diagnosed cause, and what you expect it to do. If unsure between two levers, pick the one you have not tried yet.',
            'Then output the tool block(s) for that one change plus {"tool":"generate"}. This reasoning is shown to the user and makes your choices far better — do not skip it. (You may be terse, but you must do all three steps.)',
            'DIAGNOSE↔ACTION MUST MATCH: your tool call must act on the EXACT lever your DIAGNOSE named. If you diagnosed the PROMPT as the problem, your action must be a set on the prompt text — NOT a slider. Repeatedly saying "the prompt needs refinement" while only changing Steps/CFG is a failure loop: either actually edit the prompt this turn, or pick a different diagnosis. Do not say one lever and pull another.',
            'HOW TO EDIT THE PROMPT (do this, do not just talk about it): use the EXACT prompt-field label from the CONTROLS list of the CURRENT tab — the label differs per tab. On txt2img and img2img the main field is labeled "Prompt" (plus "Negative prompt"); on Replacer it is "Positive prompt" (plus "Detection prompt" and "Negative prompt"). Always read the CONTROLS list to get the right label for the tab you are on, then write the FULL new text, e.g. on txt2img: {"tool":"set","label":"Prompt","value":"a woman in a red wool sweater, soft knit texture, natural window lighting"}. To block unwanted things: {"tool":"set","label":"Negative prompt","value":"deformed, extra fingers, watermark"}. If your diagnosis is about content/shape, THIS is the action — not a slider.',
            (window.opts && String(opts.forge_ai_task_guidance || '').trim())
                ? 'USER TASK GUIDANCE (the user\'s own workflow — follow it when deciding what to write into the prompt fields):\n'
                  + String(opts.forge_ai_task_guidance).trim()
                : '',
            'A PROMPT EDIT MUST MEANINGFULLY DIFFER from the current prompt — re-adding words already there does NOTHING. Read the current prompt in the CONTROLS list first. To actually change the output: ADD a genuinely new descriptive term you have not used, REMOVE a term that is not helping, or increase EMPHASIS on an existing term with weight syntax (term:1.3) — do not just type the same words again.',
            'STAY IN SCOPE: only work toward what the USER actually asked for. Do NOT invent goals they did not state. If the user is fine with the current setting/lighting/background/pose, never try to "fix" it. Judge each result ONLY against the user\'s stated goal, not against things you personally think could be different.',
            'REFERENCE IMAGE SCOPE: when the user gives a reference for a SPECIFIC attribute (e.g. "reference the jacket style"), use it ONLY for that attribute. Do NOT try to match the reference\'s background, setting, lighting, or composition — those are not the target. A reference photo of a jacket shot in a forest, given for the jacket style, does NOT mean move the subject into a forest.',
            'Respect slider min/max. Never pretend to see an image you were not given.',
            'Errors: when a [tool error] comes back, DIAGNOSE it — explain to the user what failed and why, then either fix the cause (different model/control/value) or ask the user. NEVER silently repeat the exact same failing action.',
            'LOOK-THEN-ACT: if you request an image ({"tool":"get_image"}), put NO settings changes in that same reply — you have not seen the image yet. Inspect first, act in your next turn.',
            'ONE AXIS PER RUN: change at most 1-2 settings between generations — multi-axis changes make it impossible to know what helped. NEVER mix detection changes and generation changes in the same run (outside the initial mask-fixing phase).',
            'After {"tool":"revert"}: the settings already EQUAL that generation\'s — do not re-set them; only set what you intend to be DIFFERENT.',
            'ITERATION LOOP: work autonomously — make your change(s) AND include {"tool":"generate"} in the SAME reply, evaluate, record the verdict, adjust, generate again, indefinitely (see CHASE MODE). The user stops you with ⏹. NEVER end a turn having changed settings without generating. Exception: if the user asked a plain question or is just discussing (not requesting image work), answer without generating.',
            'Images in this chat: you keep the ORIGINAL source image plus the most recent results (older intermediates are dropped to save context). Use the sequence to track how each settings change affected the output and whether you are converging.',
            'Prompt style: match the checkpoint. SD 1.5 / SDXL checkpoints (e.g. epicRealism) want comma-separated tag prompts and benefit from a negative prompt. Flux checkpoints want plain natural-language sentences, low CFG (~1), and ignore the negative prompt.',
            'Inpainting: prefer small targeted changes (denoising strength, mask blur, inpaint area/padding, masked content) over rewriting everything at once, so the user can see what each change does.',
            'WORKFLOW: work on whichever tab the user is currently using — you always operate on the ACTIVE tab (shown as "Current tab" below), and the CONTROLS list reflects that tab. Do not assume Replacer. On txt2img you generate from a prompt; on img2img you transform an input image; on Replacer you do prompt-guided inpainting/replacement. Use the semantics that match the current tab. The Replacer-specific guidance below applies ONLY when the current tab is Replacer.',
            'REPLACER FIELD SEMANTICS (only when on the Replacer tab) — never mix these up:',
            '- DETECTION PROMPT = what EXISTS in the SOURCE image that you want masked/selected (e.g. "sunglasses", "red shirt"). It is a SEARCH query over the original photo. NEVER put the desired outcome there — searching a photo for the desired RESULT finds nothing and breaks the mask.',
            '- POSITIVE PROMPT = what should appear INSTEAD in the masked area. Outcome words (the new garment, "bare face", "blue denim jacket") belong HERE.',
            '- NEGATIVE PROMPT = what must NOT appear in the result.',
            'WHICH LEVER DOES WHAT — this is the most important thing to get right:',
            '- The PROMPT decides WHAT appears in the masked area. DENOISING decides HOW MUCH the area is allowed to change. They are NOT interchangeable.',
            '- If the WRONG CONTENT keeps appearing (e.g. you want a leather jacket but keep getting a recolored hoodie), the PROMPT is wrong — DENOISING WILL NOT FIX IT. Set the positive prompt to describe what SHOULD be there and the negative prompt to block what should not. Nudging denoise by 0.05 when the content is wrong is the classic mistake — do not do it.',
            '- Example (replacing a garment): positive prompt should describe the NEW garment and its texture; negative prompt should list the old item ("hoodie, fleece, drawstrings"); and denoising must be HIGH (0.85–1.0) so the old garment is fully replaced, not reinterpreted. At denoise 0.5–0.7 the model keeps the old garment\'s shape and just recolors it.',
            '- Only tune denoise/CFG once the CONTENT is correct and you are refining quality/blend. Content wrong → fix prompt. Content right but blend/detail off → then denoise/CFG.',
            'SHAPE / SIZE / STYLE / POSE / COLOR requests from the user ("make it slimmer", "longer", "different pose", "warmer tone") are PROMPT edits, NOT slider edits. CFG/Steps/Denoise cannot change the shape of what is generated. Add the wanted attribute to the POSITIVE prompt (e.g. "slim-fit, tailored") and the unwanted one to the NEGATIVE prompt (e.g. "baggy, oversized"). Changing Steps or CFG to fix a shape is the fishing mistake — do not.',
            'When the user asks for a specific visual change, your FIRST move is the relevant prompt edit + generate — one focused change, not a bundle of slider tweaks.',
            'DO NOT RE-FETCH THE RESULT: the latest generated result is ALREADY attached to every generation tool-result and to the user\'s messages — you can see it. Never call get_image to "look at" the result; that wastes a whole round-trip. Judge the attached image directly. Use get_image ONLY to inspect the mask, and only when detection is unlocked and you suspect it is wrong (rare).',
            'MASK VERIFICATION: when a result looks wrong, check the SELECTION before blaming generation settings — {"tool":"get_image","which":"gallery"} shows the greyscale mask and the masked composite. Mask covers the wrong area / too little / too much → fix detection prompt, box threshold, or mask expand. Mask is correct but content is bad → then tune denoise/CFG/prompt. This order saves wasted generations.',
            'BASELINE FIRST — your standing procedure on a new image/task:',
            '- Your FIRST action is a baseline generation with the user\'s current settings UNCHANGED: {"tool":"generate"} and NOTHING else. Do not touch ANY setting — especially not the detection prompt — before seeing the baseline. The user\'s existing settings are usually good. Sole exception: the user asked for a specific content change (e.g. a new replacement subject) — set exactly that, nothing more.',
            '- Detection starts LOCKED BY DEFAULT and detection settings are physically BLOCKED. The user\'s existing detection is usually already correct — leave it alone. Changing a working detection prompt is the #1 way you have ruined sessions.',
            '- After the baseline, check the mask in the gallery. If it covers the right region (it usually does), do NOTHING to detection — it is already locked and protected. Move on to generation settings.',
            '- A missing / incomplete / poorly-rendered area in the OUTPUT is almost ALWAYS a GENERATION problem, NOT a mask problem: the mask covers it, the model just rendered it badly. Fix it with denoise / steps / prompt — do NOT touch detection. Detection controls WHICH region is regenerated, never HOW WELL it renders.',
            '- CRITICAL — how a GOOD mask looks: the mask covers the ITEM being replaced (the shirt, the sunglasses, the hair), NOT the desired result area. That is CORRECT and expected. The masked region is exactly what gets regenerated into the replacement content. Do NOT conclude "the result is not masked so detection is wrong" — a mask over the item being replaced IS how replacement works. A mask over the target item = a GOOD mask; leave detection locked.',
            '- Before you may unlock detection you MUST first look at the mask itself with {"tool":"get_image","which":"mask"} and confirm the mask genuinely fails to cover the TARGET item. If it covers the target, the mask is CORRECT — keep detection locked and fix denoise/steps/prompt instead.',
            '- Only after verifying the mask truly misses (or wrongly includes) an area: {"tool":"unlock_detection"}, make the minimal fix (usually Mask Expand), re-verify, {"tool":"lock_detection"}. Never unlock just to re-type a detection prompt that already selects the right thing.',
            '- Detection is deterministic (same settings, same mask), so re-tuning a good mask cannot help. Iterate on generation settings (denoise, CFG, prompts, ControlNet, sampler), not detection.',
            'Convergence strategy — scale your changes to how close the result is:',
            '- FAR from the goal: bigger moves are fine (denoise ±0.15+, rewrite the prompt, enable ControlNet, different approach).',
            '- CLOSE (composition right, details off): change ONE or TWO things, in small steps (denoise ±0.05, CFG ±0.5-1, ControlNet weight ±0.1, add/remove single prompt phrases). Lock the seed first with {"tool":"click","label":"Reuse seed from last generation"} so the next run is directly comparable — otherwise a new random seed masks the effect of your change.',
            '- NEARLY THERE: micro-adjustments only (denoise ±0.02-0.03, single word prompt edits). Keep the seed locked. Do NOT rework things that already look right.',
            '- State which regime you think you are in when you make changes. When the user says it looks good except one thing, fix only that thing.',
            '- Set Seed back to -1 when the user wants variety or a fresh direction.',
            'SEED DISCIPLINE (critical): comparisons are only valid with a FIXED seed. After the baseline the seed is auto-locked. If you see repeated "mixed"/"tie" verdicts, the changes are too small to matter at this seed — do NOT keep nudging the same slider; either make a bigger change or move to a different lever. Never tune settings while the seed is -1.',
            'DISTORTION/PROPORTIONS on inpainting: warped anatomy or elongated bodies come from the inpaint region, NOT from denoise/CFG. FIRST FIX, in order: (1) "Inpaint area" = "Only masked" (regenerates the masked region at native detail — "Whole picture" stretches it), then (2) raise "Only masked padding" (e.g. 32→64). These fix proportions the vast majority of the time. Do NOT reach for ControlNet for proportions — ControlNet frequently fails in this setup (crashes the generation) and is rarely needed for inpainting. Only consider ControlNet if Inpaint area + padding genuinely did not help, and even then expect it may fail.',
            'CHASE MODE — you run continuously toward the best possible result; the user stops you with the ⏹ button, so NEVER stop on your own or declare it "good enough". Keep going.',
            'EXPLORE vs EXPLOIT — this is how "chase forever" actually improves instead of spinning:',
            '- EXPLOIT: at the locked seed, refine settings toward the best (small, measurable changes). When you beat the best, keep refining from there.',
            '- PLATEAU: if ~3 tries at the current seed do not beat the best, you have found what this seed can do. Do NOT keep nudging — RE-ROLL: set Seed to -1, generate a fresh composition, then re-lock the new seed and exploit it. A new seed can have a higher ceiling.',
            '- The global BEST is always kept as the floor — a bad exploration never loses your best result; revert to it anytime with {"tool":"revert","to":N}.',
            '- Vary WHICH lever you explore across rounds: denoise, CFG, prompt wording, ControlNet weight, inpaint area, sampler — not the same slider repeatedly.',
            '- Only pause for the user if the GOAL itself is ambiguous (not for permission to continue). To pause, reply starting with "NEED INPUT:" and your question.',
            'EVALUATION PROTOCOL — every generation is numbered, diffed, and BLIND-JUDGED:',
            '- Each new result is compared to the best-so-far by an independent blind judge (clean context, random image order). Its verdict is AUTHORITATIVE over YOUR opinion and pre-recorded in the run log. You are biased toward your own changes — when the judge says worse, it is worse; never argue it into "better".',
            '- THE USER OUTRANKS THE JUDGE — always. If the user says a verdict was wrong, or that a result is the best/good/perfect/"that\'s it", immediately make that the best: {"tool":"verdict","gen":<the current/named generation>,"verdict":"better","note":"user approved","best":<that generation number>}. This re-points the baseline so all future blind judging compares against the user-approved image. Never argue with the user about quality. When the user approves a result, also {"tool":"remember"} the settings/prompt that produced it.',
            '- JUDGE UNRELIABLE ON FINE DETAILS: the blind judge often returns TIE on subtle differences (small anatomy/texture changes) because its two passes disagree. If you get 2+ TIEs in a row, STOP trusting the judge for this distinction — show the user your latest result and ASK them directly which is better (reply starting with "NEED INPUT:"), then use their answer as the authoritative verdict and best. Do not keep spinning on a judge that cannot tell.',
            '- Judge says WORSE → revert to the best generation and try a different, smaller change. Never pile new changes on top of a regression.',
            '- Judge says TIE → the change did nothing; try a different axis instead of pushing the same slider further.',
            '- Two failed attempts in a row against the same best → revert to best, tell the user what you tried, and ask for direction.',
            'ControlNet: all units are listed, prefixed "ControlNet Unit 0/1/2" — you can set any of them even while their section is collapsed. To lock composition/pose/edges while inpainting at high denoise, set a unit\'s Enable to true and pick a preprocessor+model (canny/depth for structure, openpose for people); tune Control Weight and start/end steps. Stack units for combined constraints (e.g. Unit 0 depth + Unit 1 openpose). In img2img each unit uses the source image automatically.',
            'ControlNet TROUBLESHOOTING — if a result is garbage (neon/green/rainbow colors, melted/duplicated content, or it ignores the prompt) right after you ENABLED or CHANGED ControlNet, the CONTROLNET is the cause — NOT denoise or the prompt. Do not tune denoise/CFG to fix it. FIRST disable it: {"tool":"set","label":"ControlNet Unit 0 > Enable","value":false} and regenerate to confirm. Only re-enable with a VALID preprocessor + matching-arch model + Pixel Perfect on. A unit enabled with no model, a mismatched preprocessor, or wrong-arch model produces exactly this garbage. Most inpainting needs NO ControlNet — when unsure, leave it OFF.',
            'Before enabling any ControlNet unit, confirm from the CONTROLS list that a valid model is selected for it (not "None"). Enabling with model "None" corrupts output.',
            (function () {
                const arch = archOfCheckpoint();
                if (arch === 'flux') return 'ControlNet MODELS: current mode is FLUX — no Flux ControlNet models are installed. Do NOT enable ControlNet; Flux Fill conditions on the mask natively.';
                if (arch === 'xl') return 'ControlNet MODELS: current checkpoint is SDXL — you may ONLY use these models (sd15/v11 models are blocked and corrupt output): ' + CN_MODELS.xl + '. If the Model dropdown does not list them, tell the user to click the 🔄 refresh icon next to it.';
                return 'ControlNet MODELS: current checkpoint is SD 1.5 — you may ONLY use the sd15 models: ' + CN_MODELS.sd + '. Never the sdxl/union/Kataragi ones.';
            })(),
            '',
            'Current SD checkpoint: ' + ckpt,
            (checkpointNotes(ckpt) ? 'AUTHOR-RECOMMENDED SETTINGS for this checkpoint (prefer these over generic defaults): ' + checkpointNotes(ckpt) : ''),
            'The tabs are: ' + topTabButtons().map(b => b.textContent.trim()).join(', ') + '. The img2img tab has sub-tabs: img2img, Sketch, Inpaint, Inpaint sketch, Inpaint upload, Batch.',
            memorySection(),
            // NOTE: the volatile state (current tab, run log, live CONTROLS values)
            // is sent as a TRAILING message, not here — so it changing does not
            // invalidate the cached KV of the (expensive) images before it.
        ].filter(Boolean).join('\n');
    }

    // Volatile per-turn state — appended AFTER the conversation so the images
    // stay in the stable cached prefix. This is the only part that churns each
    // turn (control values, run log), so only it gets recomputed.
    function liveContext(controls) {
        return [
            '[CURRENT STATE — reference for your next action; respond to the conversation above using these live values]',
            'Current tab: ' + currentTabName() + '.',
            runLogSection(),
            'CONTROLS on the current tab (live values):',
            controlsListing(controls),
        ].filter(Boolean).join('\n');
    }

    // Brace-scan every top-level {...} span in a string (string-literal aware).
    function extractJsonObjects(text) {
        const objs = [];
        let depth = 0, start = -1, inStr = false, esc = false;
        for (let i = 0; i < text.length; i++) {
            const ch = text[i];
            if (inStr) {
                if (esc) esc = false;
                else if (ch === '\\') esc = true;
                else if (ch === '"') inStr = false;
                continue;
            }
            if (ch === '"') { inStr = true; continue; }
            if (ch === '{') { if (depth === 0) start = i; depth++; }
            else if (ch === '}') {
                if (depth > 0) depth--;
                if (depth === 0 && start >= 0) { objs.push(text.slice(start, i + 1)); start = -1; }
            }
        }
        return objs;
    }

    // Tolerant tool extraction: any fenced block (multiple objects per block
    // allowed), plus bare inline {"tool": ...} objects outside fences, with
    // trailing-comma repair. Models fumble formats — dropping their tool calls
    // silently makes them think they changed things when nothing happened.
    function extractTools(text) {
        const candidates = [];
        const re = /```\w*\s*([\s\S]*?)```/g;
        let m;
        while ((m = re.exec(text)) !== null) candidates.push(...extractJsonObjects(m[1]));
        const outside = text.replace(/```[\s\S]*?```/g, '');
        for (const o of extractJsonObjects(outside)) {
            if (/"tool"\s*:/.test(o)) candidates.push(o);
        }
        const tools = [];
        for (const c of candidates) {
            let obj = null;
            try { obj = JSON.parse(c); }
            catch (e) {
                try { obj = JSON.parse(c.replace(/,\s*([}\]])/g, '$1')); } catch (e2) { /* unparseable */ }
            }
            if (obj && obj.tool) tools.push(obj);
        }
        return tools;
    }

    async function executeTools(tools, controls) {
        // apply ControlNet MODEL sets before ControlNet ENABLE sets, so if the
        // model does both in one reply the model lands first and the enable
        // guardrail passes (stable order otherwise preserved)
        const rank = (t) => {
            if (t.tool !== 'set') return 1;
            const l = String(t.label || '');
            if (/controlnet/i.test(l) && /model/i.test(l)) return 0;
            if (/controlnet/i.test(l) && /enable/i.test(l)) return 2;
            return 1;
        };
        tools = tools.map((t, i) => [t, i]).sort((a, b) => (rank(a[0]) - rank(b[0])) || (a[1] - b[1])).map(p => p[0]);

        const feedback = [];        // text results for the follow-up turn
        const followupImages = [];  // images requested via get_image
        let fullReturn = null;      // generate cycle supplies complete content
        let didSet = false;         // did any setting change actually apply?

        async function runOne(t) {
            if (t.tool === 'set') {
                const c = findControl(controls, t.label);
                // guardrail: detection is locked — refuse changes to detection settings
                if (c && detectionLocked && isDetectionControl(c.label)) {
                    sysMsg(`🔒 blocked change to "${c.label}" — detection is locked`);
                    feedback.push(`[tool error] detection is LOCKED — "${c.label}" is protected. But if the USER reported a mask coverage problem (a missing/uncovered area, wrong region, or too much masked), that is a valid reason: call {"tool":"unlock_detection"} first, then fix it. Otherwise leave detection alone and tune generation settings (denoise, prompt, CFG).`);
                    return;
                }
                // guardrail: the detection prompt is a SEARCH over the source
                // image — outcome words there break the mask entirely
                if (c && /detection prompt/i.test(c.label)) {
                    const bad = String(t.value).toLowerCase().match(/\b(nude|naked|bare skin|nothing|removed|topless|bottomless)\b/);
                    if (bad) {
                        sysMsg(`🚫 blocked outcome-word "${bad[1]}" in the detection prompt`);
                        feedback.push(`[tool error] "${bad[1]}" is an OUTCOME, not something that exists in the source photo — the detection prompt is a search query over the ORIGINAL image and must name the objects to mask (e.g. "sunglasses", "red shirt"). Put outcome words in the positive prompt instead.`);
                        return;
                    }
                }
                // guardrail (fail-closed): a ControlNet unit may ONLY be enabled
                // when a valid, arch-matching model is already selected for it.
                // Missing model / "None" / wrong-arch / model-control-not-found
                // all block the enable — an unmatched ControlNet crashes Forge
                // (shape-mismatch RuntimeError) or produces garbage.
                if (c && /controlnet/i.test(c.label) && /enable/i.test(c.label) &&
                    (t.value === true || String(t.value).toLowerCase() === 'true')) {
                    // circuit-breaker: ControlNet has crashed generations repeatedly
                    if (cnFailCount >= 2) {
                        sysMsg('🚫 ControlNet disabled for the session (failed ' + cnFailCount + '×)');
                        feedback.push(`[tool error] ControlNet has already crashed ${cnFailCount} generations this session — it is not working in this setup. STOP trying to enable it. Fix proportions with "Inpaint area" = "Only masked" and "Only masked padding" instead. (The user can re-enable ControlNet manually if they truly need it.)`);
                        return;
                    }
                    const unit = (c.label.match(/unit\s*\d+/i) || [''])[0];
                    const modelCtl = controls.find(x => /controlnet/i.test(x.label) && (unit ? new RegExp(unit, 'i').test(x.label) : true) && /model/i.test(x.label));
                    let mv = modelCtl ? String(modelCtl.get() ?? '').trim() : '';
                    const want = archOfCheckpoint();

                    if (want === 'flux') {
                        sysMsg('🚫 blocked enabling ControlNet — Flux has no ControlNet installed');
                        feedback.push(`[tool error] refused to enable ${unit || 'the ControlNet unit'}: this is FLUX mode and no Flux ControlNet model is installed. Do not use ControlNet in Flux mode.`);
                        return;
                    }

                    // if no valid arch-matching model is selected, AUTO-SELECT the
                    // correct one so ControlNet always has a working model — the
                    // union SDXL model is universal; sd15 gets a sane default.
                    const needsModel = !modelCtl || mv === '' || mv.toLowerCase() === 'none' || cnArchViolation(mv);
                    if (needsModel && modelCtl) {
                        const target = want === 'xl' ? 'union-promax-sdxl' : 'control_v11p_sd15_openpose';
                        sysMsg(`⚙ auto-selecting a ${want.toUpperCase()} ControlNet model (${target})…`);
                        try { await modelCtl.set(target); await sleep(400); } catch (e) { /* verified below */ }
                        mv = String(modelCtl.get() ?? '').trim();
                    }

                    // re-verify: only allow the enable if a valid arch model is now set
                    if (!modelCtl || mv === '' || mv.toLowerCase() === 'none' || cnArchViolation(mv)) {
                        const reason = !modelCtl ? `no Model dropdown found for ${unit || 'the unit'}`
                            : (mv === '' || mv.toLowerCase() === 'none') ? 'could not auto-select a valid model'
                                : `"${mv}" is the wrong architecture for this ${want.toUpperCase()} checkpoint`;
                        sysMsg('🚫 blocked enabling ControlNet — ' + reason);
                        feedback.push(`[tool error] refused to enable ${unit || 'the ControlNet unit'}: ${reason}. Set the Model dropdown to a valid ${want.toUpperCase()} ControlNet FIRST (${CN_MODELS[want] || ''}), THEN enable. Enabling with no/wrong model crashes the generation. Or leave ControlNet OFF.`);
                        return;
                    }
                    sysMsg(`✓ ControlNet model confirmed: ${mv}`);
                }
                // hard guardrail: refuse wrong-architecture ControlNet models
                if (c && c.kind === 'dropdown' && /controlnet/i.test(c.label) && /model/i.test(c.label)) {
                    const violation = cnArchViolation(t.value);
                    if (violation) {
                        sysMsg('🚫 blocked wrong-arch ControlNet model');
                        feedback.push('[tool error] ' + violation);
                        return;
                    }
                }
                if (!c) {
                    const words = String(t.label).toLowerCase().split(/\s+/).filter(w => w.length > 2);
                    const near = controls.filter(x => words.some(w => x.label.toLowerCase().includes(w)))
                        .slice(0, 6).map(x => `"${x.label}"`).join(', ');
                    feedback.push(`[tool error] no control labelled "${t.label}" found.` + (near ? ` Similar controls: ${near}` : ''));
                    return;
                }
                // reject a no-op prompt edit — re-adding words already present
                // does nothing and just produces another tie
                if (/prompt/i.test(c.label) && !/detection/i.test(c.label) && c.kind === 'text') {
                    const red = promptRedundant(c.get(), t.value);
                    if (red) {
                        sysMsg('🚫 redundant prompt edit blocked');
                        feedback.push(`[tool error] that prompt edit does nothing — ${red}. Current "${c.label}": "${String(c.get()).slice(0, 200)}". For a REAL change: add genuinely NEW descriptive terms (a different attribute, lighting, texture, angle), REMOVE terms that aren't helping, or change EMPHASIS with weights like (soft window light:1.3). Do not re-add words already there.`);
                        return;
                    }
                }
                try {
                    await c.set(t.value);
                    didSet = true;
                    sysMsg(`⚙ ${c.label} → ${t.value}`);
                } catch (e) {
                    feedback.push(`[tool error] failed to set "${c.label}": ${e.message}`);
                }
            } else if (t.tool === 'get_image') {
                // explicit mask inspection — works even while detection is locked,
                // so the model can verify coverage BEFORE deciding to unlock
                if (t.which === 'mask') {
                    const imgs = await captureGallery();
                    const maskImgs = imgs.slice(1);   // [0]=result, rest are mask/composite
                    if (!maskImgs.length) { feedback.push('[tool error] no mask/composite images in the gallery — generate first, or the mask preview is off'); return; }
                    maskImgs.forEach(u => followupImages.push({ type: 'image_url', image_url: { url: u } }));
                    sawMask = true;
                    sysMsg('🔍 mask/composite attached for verification');
                    feedback.push('[tool result] the greyscale MASK and masked composite are attached. Check ONLY whether the mask covers the intended region. If it covers the area the user mentioned, the mask is FINE — the problem is generation (denoise/steps/prompt), keep detection locked. Only if the mask genuinely misses/over-covers may you unlock_detection.');
                    return;
                }
                if (t.which === 'gallery') {
                    // once detection is locked, the mask is verified — don't waste
                    // tokens/attention re-reading mask images; return only the result
                    if (detectionLocked) {
                        const r = await captureImage('result');
                        if (r) { followupImages.push({ type: 'image_url', image_url: { url: r } }); }
                        feedback.push('[tool result] detection is LOCKED (mask already verified good) — only the result image is attached; the mask is not re-sent. Judge the result, do not re-inspect the mask.');
                        sysMsg('📷 result attached (mask skipped — detection locked)');
                        return;
                    }
                    const imgs = await captureGallery();
                    if (!imgs.length) { feedback.push('[tool error] no gallery images found on this tab'); return; }
                    sysMsg(`📷 attached ${imgs.length} gallery image(s) — result + mask/composite`);
                    imgs.forEach(u => followupImages.push({ type: 'image_url', image_url: { url: u } }));
                    feedback.push(`[tool result] ${imgs.length} gallery images attached IN ORDER as shown in the carousel: generated result(s) first, then the selection/mask images (greyscale mask, masked composite). Verify the selection covers the intended target and nothing else, then LOCK detection.`);
                    return;
                }
                const which = t.which === 'source' ? 'source' : 'result';
                const img = await captureImage(which);
                if (!img) { feedback.push('[tool error] no image found on this tab'); return; }
                lastSeen[seenKey(which)] = hashStr(img);
                sysMsg(`📷 attached ${which} image`);
                followupImages.push({ type: 'image_url', image_url: { url: img } });
                feedback.push(`[tool result] ${which} image attached.`);
            } else if (t.tool === 'generate') {
                const parts = await generateCycle();
                if (parts) fullReturn = parts;
                else feedback.push('[tool error] generation cycle failed — the LLM restart did not complete');
            } else if (t.tool === 'lock_detection') {
                detectionLocked = true;
                sysMsg('🔒 detection locked — mask settings are now protected');
                feedback.push('[tool result] detection LOCKED. Detection settings (prompt, box threshold, mask expand) are now protected from changes. Iterate only on generation settings from here.');
            } else if (t.tool === 'unlock_detection') {
                // gate: must verify the mask first — stops the model from
                // unlocking to "fix" what is actually a generation problem
                if (!sawMask) {
                    sysMsg('🚫 unlock blocked — verify the mask first');
                    feedback.push('[tool error] you have not looked at the mask yet, so you cannot know it is wrong. A missing/incomplete area in the OUTPUT is usually a GENERATION problem (denoise/steps/prompt), not a mask problem. First {"tool":"get_image","which":"mask"} and confirm the mask genuinely misses the area. Only then unlock.');
                    return;
                }
                detectionLocked = false;
                sawMask = false;   // require a fresh check next time
                sysMsg('🔓 detection unlocked');
                feedback.push('[tool result] detection unlocked — make the MINIMAL mask fix (usually Mask Expand), verify, then re-lock. Do not re-type a detection prompt that already works.');
            } else if (t.tool === 'verdict') {
                const e = settingsHistory.find(x => x.n === Number(t.gen));
                if (!e) { feedback.push(`[tool error] no generation #${t.gen} in the run log`); return; }
                const v = String(t.verdict || '').toLowerCase();
                if (!['better', 'worse', 'mixed'].includes(v)) { feedback.push('[tool error] verdict must be "better", "worse", or "mixed"'); return; }
                e.verdict = v;
                e.note = String(t.note || '').slice(0, 150);
                persistRuns();
                sysMsg(`⚖ generation #${e.n}: ${v}${e.note ? ' — ' + e.note : ''}`);
                if (t.best !== undefined) {
                    const b = settingsHistory.find(x => x.n === Number(t.best));
                    if (!b) {
                        feedback.push(`[tool error] cannot set best: no generation #${t.best} in the run log`);
                    } else if (!b.img) {
                        feedback.push(`[tool error] generation #${t.best} has no stored image (lost in a page reload) — regenerate it (revert to its settings + generate) to re-establish it as the baseline`);
                    } else {
                        bestResult = { url: b.img, gen: b.n };
                        sysMsg(`⭐ best-so-far re-pointed to generation #${b.n} (user correction)`);
                        feedback.push(`[tool result] best is now generation #${b.n}; future blind judging compares against it.`);
                    }
                }
            } else if (t.tool === 'get_settings') {
                const e = settingsHistory.find(x => x.n === Number(t.gen));
                if (!e) { feedback.push(`[tool error] no generation #${t.gen} in the run log`); return; }
                let lines = Object.entries(e.snap).map(([k, v]) => `${k} = ${v}`).join('\n');
                if (lines.length > 2500) lines = lines.slice(0, 2500) + '…';
                feedback.push(`[tool result] full settings used for generation #${e.n} (tab: ${e.tab}):\n${lines}`);
            } else if (t.tool === 'remember') {
                const note = String(t.note || '').trim();
                if (!note) { feedback.push('[tool error] remember needs a non-empty "note"'); return; }
                const ckpt = gradioApp().querySelector('#setting_sd_model_checkpoint input')?.value || '';
                const r = await apiJSON('/forge-ai/memory', { note: note, checkpoint: ckpt, tab: currentTabName() });
                if (r.ok) {
                    sysMsg('📝 remembered: ' + note);
                    await refreshMemory();
                } else {
                    feedback.push('[tool error] could not save note: ' + (r.error || 'unknown'));
                }
            } else if (t.tool === 'revert') {
                const err = await revertTo(t.to);
                if (err) feedback.push('[tool error] ' + err);
                else feedback.push(`[tool result] settings reverted${t.to ? ' to those of generation #' + t.to : " to the previous generation's"}. Verify the CONTROLS list next turn.`);
            } else if (t.tool === 'switch_tab') {
                // Tab switching is unrestricted: the assistant works on whichever
                // tab the user is using (txt2img, img2img, Replacer, ...).
                const err = await switchTab(t.tab, t.subtab);
                if (err) {
                    feedback.push('[tool error] ' + err);
                } else {
                    controls = scanControls();   // later tools in this same reply target the NEW tab
                    const where = t.tab + (t.subtab ? ' → ' + t.subtab : '');
                    sysMsg('📑 switched to ' + where);
                    feedback.push(`[tool result] now on "${where}". Controls are rescanned — sets in this same reply now target the new tab; the full CONTROLS list arrives next turn.`);
                }
            } else if (t.tool === 'click') {
                const want = String(t.label || '').toLowerCase();
                const btns = [...gradioApp().querySelectorAll('button')].filter(b => visible(b) && !b.closest('#fai-panel'));
                const hit = btns.find(b => b.textContent.trim().toLowerCase() === want)
                    || btns.find(b => (b.title || '').trim().toLowerCase() === want)
                    || btns.find(b => b.textContent.trim().toLowerCase().includes(want) || (b.title || '').toLowerCase().includes(want));
                if (!hit) {
                    feedback.push(`[tool error] no visible button matching "${t.label}"`);
                } else {
                    hit.click();
                    await sleep(400);
                    sysMsg(`🖱 clicked "${t.label}"`);
                    feedback.push(`[tool result] clicked "${t.label}".`);
                }
            } else {
                feedback.push(`[tool error] unknown tool "${t.tool}"`);
            }
        }

        for (const t of tools) {
            try {
                await runOne(t);
            } catch (e) {
                feedback.push(`[tool error] ${t.tool || 'tool'} crashed: ${e.message}`);
            }
            if (fullReturn) return fullReturn;
        }

        // Auto-generate: if the model changed settings but did NOT ask to
        // generate (common now that it reasons first, then stops), run the
        // generation itself — a settings change is pointless without it, and
        // the user expects the loop to keep moving.
        const askedGenerate = tools.some(t => t.tool === 'generate');
        if (didSet && !askedGenerate && !followupImages.length && !stopRequested) {
            sysMsg('▶ auto-generating (you changed settings but didn\'t generate)');
            const parts = await generateCycle();
            if (parts) return parts;
            feedback.push('[tool error] auto-generation after your settings change failed');
        }

        if (followupImages.length) {
            return [{ type: 'text', text: feedback.join('\n') }, ...followupImages];
        }
        if (feedback.length) return [{ type: 'text', text: feedback.join('\n') }];
        return null;   // plain sets need no follow-up turn
    }

    // Cap images sent to the LLM — each costs 1000+ tokens, and if the prompt
    // ever gets truncated server-side, orphaned images crash llama.cpp
    // ("bitmaps does not match markers"). The ORIGINAL source image is always
    // kept so the AI can compare the starting point against the progression;
    // beyond that, the newest images win. Middle ones become placeholders.
    const MAX_IMAGES_SENT = 6;

    function messagesForSend() {
        // locate the very first image in history (the original source)
        let firstImg = null;
        outer:
        for (let i = 0; i < messages.length; i++) {
            const c = messages[i].content;
            if (!Array.isArray(c)) continue;
            for (let j = 0; j < c.length; j++) {
                if (c[j].type === 'image_url') { firstImg = i + ':' + j; break outer; }
            }
        }

        let kept = 0;
        const out = [];
        for (let i = messages.length - 1; i >= 0; i--) {
            const m = messages[i];
            if (!Array.isArray(m.content)) { out.unshift(m); continue; }
            const parts = m.content.map((p, j) => {
                if (p.type !== 'image_url') return p;
                const isOriginal = (i + ':' + j) === firstImg;
                const isReference = referenceImage && p.image_url && p.image_url.url === referenceImage;
                kept++;
                if (isOriginal || isReference || kept <= MAX_IMAGES_SENT - 1) return p;
                return { type: 'text', text: '[an older intermediate image was here — removed to save context]' };
            });
            out.unshift({ role: m.role, content: parts });
        }
        return out;
    }

    // Persist the (image-pruned) chat so a page reload doesn't lose the
    // conversation — restored in buildUI.
    function persistChat() {
        try { sessionStorage.setItem('fai_chat', JSON.stringify(messagesForSend())); } catch (e) { /* quota — skip */ }
        botAutosave();
    }

    // The bot's own persistent state: survives Forge/browser restarts, cleared
    // ONLY by the 🗑 button. Separate from the ↺ settings snapshot on purpose.
    let botSaveTimer = null;
    function botAutosave() {
        clearTimeout(botSaveTimer);
        botSaveTimer = setTimeout(async () => {
            // serializing the chat (with images) costs several MB of JSON —
            // don't do it repeatedly mid-turn; wait until the agent is idle
            if (busy) { botAutosave(); return; }
            try {
                // run-log images only for the newest 3 runs — keeps the file small
                const runs = settingsHistory.map((r, i) => {
                    const copy = Object.assign({}, r);
                    if (i < settingsHistory.length - 3) delete copy.img;
                    return copy;
                });
                await apiJSON('/forge-ai/botstate/save', {
                    v: 1, ts: Date.now(),
                    messages: messagesForSend(),
                    settingsHistory: runs,
                    genCounter: genCounter,
                    bestResult: bestResult,
                    referenceImage: referenceImage,
                    detectionLocked: detectionLocked,
                    sawMask: sawMask,
                }, 60000);
            } catch (e) { /* best-effort */ }
        }, 3000);
    }

    // Silently bring the bot's memory back on page load (no LLM start, no
    // auto-send) — used when per-tab sessionStorage has nothing (new browser).
    async function botRestoreLatest() {
        try {
            const r = await apiJSON('/forge-ai/botstate/latest', undefined, 60000);
            if (!r || !r.exists || !r.state) return false;
            const s = r.state;
            messages = Array.isArray(s.messages) ? s.messages : [];
            settingsHistory.length = 0;
            settingsHistory.push(...(Array.isArray(s.settingsHistory) ? s.settingsHistory : []));
            genCounter = s.genCounter || 0;
            bestResult = s.bestResult || null;
            referenceImage = s.referenceImage || null;
            detectionLocked = s.detectionLocked !== false;
            sawMask = !!s.sawMask;
            sentImageUrls = new Set();   // a fresh server has nothing cached yet
            for (const m of messages) {
                const text = Array.isArray(m.content)
                    ? m.content.filter(p => p.type === 'text').map(p => p.text).join(' ')
                    : String(m.content || '');
                if (!text) continue;
                if (m.role === 'assistant') renderAssistant(text);
                else renderUser(text.length > 400 ? text.slice(0, 400) + '…' : text, 0, false);
            }
            if (messages.length) sysMsg('(bot memory restored from ' + (s.ts ? new Date(s.ts).toLocaleString() : 'last session') + ' — 🗑 clears it for a fresh start)');
            return messages.length > 0;
        } catch (e) {
            return false;
        }
    }

    // ---- server-side session snapshot: survives Forge AND browser restarts ----
    // Autosaved (debounced) to extensions/forge-ai-assistant/last_session.json;
    // brought back by the "↺ Restore session" button next to the UI mode radio.
    let sessionSaveTimer = null;

    // UI state (sliders/prompts/dropdowns/checkboxes/radios) per top-level tab.
    // Captured for every tab the user visits, so Restore puts back settings in
    // ALL windows they worked in — not just the chat.
    let uiSnapshots = {};        // { topTabName: { label: value } }
    let uiActiveTab = null;      // tab that was active at save time
    let uiRestoring = false;     // don't capture while we're mid-restore
    let lastUiSnapJson = '';
    let lastScanKinds = {};      // {slider: N, dropdown: N, ...} from the last capture — save diagnostics
    const UI_SKIP_TABS = /^(settings|extensions)$/i;   // never bulk-write Forge settings

    function captureUiSnapshot(force) {
        if (!force && (uiRestoring || busy)) return;   // AI mid-turn moves controls constantly — skip
        try {
            const tab = currentTabName();
            if (!tab || tab === 'unknown' || UI_SKIP_TABS.test(tab)) return;
            const snap = {};
            lastScanKinds = {};
            for (const c of scanControls()) {
                try {
                    const v = c.get();
                    // settings only — never image data or absurd blobs
                    if (typeof v === 'string' && (v.length > 1500 || v.startsWith('data:'))) continue;
                    snap[c.label] = v;
                    lastScanKinds[c.kind] = (lastScanKinds[c.kind] || 0) + 1;
                } catch (e) { /* unreadable control */ }
            }
            // remember which SUB-tab was active (img2img vs Inpaint vs Sketch…):
            // controls inside inactive sub-tabs aren't scannable, so restore
            // must return to the same sub-tab before applying values
            try {
                const st = [...activeTabRoot().querySelectorAll('.tab-nav button.selected, [role=tab][aria-selected="true"]')].find(visible);
                if (st) snap['__subtab'] = st.textContent.trim();
            } catch (e) { /* no sub-tabs here */ }
            if (Object.keys(snap).length) {
                // MERGE over the previous snapshot instead of replacing it: gradio 6
                // unmounts the contents of closed accordions (gradio 4 kept them in
                // the DOM hidden), so a scan taken while e.g. Hires fix is collapsed
                // no longer sees its sub-controls — replacing wholesale would erase
                // their previously captured values. Stale keys are harmless: apply
                // type-guards them and unmatched ones are skipped.
                uiSnapshots[tab] = Object.assign({}, uiSnapshots[tab] || {}, snap);
                uiActiveTab = tab;
            }
        } catch (e) { /* DOM mid-rebuild */ }
    }

    async function applyUiSnapshots(snaps, activeTab) {
        let applied = 0, failed = 0;
        const unmatched = [];
        const expandedAccordions = new Set();   // one expand-click per accordion per restore
        for (const tab of Object.keys(snaps || {})) {
            setActivity('↺ restoring settings on "' + tab + '"…');
            const pending = new Map(Object.entries(snaps[tab]));
            const subtab = pending.get('__subtab');
            pending.delete('__subtab');
            const err = await switchTab(tab);
            if (err) { failed += pending.size; continue; }
            if (subtab) await switchTab(null, subtab);   // best-effort — apply what's visible either way
            await sleep(300);

            // Pass 1: checkboxes & radios only — enabling a feature (Soft
            // inpainting, a ControlNet unit) is what makes its sub-controls
            // usable. Later passes: everything else, RESCANNING between passes
            // so controls revealed by pass 1 are found and set.
            // normalized fallback: profiles saved before label-stabilization
            // contain decorations like "Unit 1[Depth]" that no longer appear, and
            // pre-gradio-6 profiles carry whole accordion headers ("Hires. fix ▼
            // Upscaler …") for InputAccordion toggles — cut at the ▼ marker.
            const normLabel = s => String(s).toLowerCase()
                .split('▼')[0]
                .replace(/\[[^\]]*\]/g, '').replace(/\d+\s*units?/g, '').replace(/\s+/g, ' ').trim();
            const clickedTabs = new Set();   // one click per inner tab per restore
            for (let pass = 1; pass <= 10 && pending.size; pass++) {
                const controls = scanControls();
                const normMap = new Map();
                for (const x of controls) {
                    const n = normLabel(x.label);
                    if (!normMap.has(n)) normMap.set(n, x);
                }
                // resolve what we can this pass, then apply in dependency order:
                // toggles first (reveal sections), dropdowns second (ControlNet
                // RENAMES its sliders when a preprocessor is picked), rest last —
                // unresolved labels stay pending for the next rescan
                const found = [];
                for (const [label, value] of [...pending]) {
                    const c = controls.find(x => x.label === label)
                        || normMap.get(normLabel(label))
                        || findControl(controls, label);
                    if (c) found.push([label, value, c]);
                }
                const prio = { checkbox: 0, radio: 1, dropdown: 2 };
                found.sort((a, b) => (prio[a[2].kind] ?? 3) - (prio[b[2].kind] ?? 3));
                for (const [label, value, c] of found) {
                    if (pass === 1 && c.kind !== 'checkbox' && c.kind !== 'radio') continue;
                    // type guard: a fuzzy label match must never write a value of the
                    // wrong shape (e.g. "true" into a slider — range inputs silently
                    // reset to their MIDPOINT on invalid values, which is how Width
                    // once became 1056). Keep the entry PENDING instead of consuming
                    // it: the right control may mount in a later pass (accordion
                    // expansion); leftovers surface as 'not found' at the end.
                    if ((c.kind === 'slider' || c.kind === 'number') && !isFinite(parseFloat(value))) continue;
                    if (c.kind === 'checkbox' && String(value) !== 'true' && String(value) !== 'false') continue;
                    let cur;
                    try { cur = c.get(); } catch (e) { cur = undefined; }
                    if (String(cur) === String(value)) { pending.delete(label); continue; }
                    try {
                        await c.set(value);
                        applied++;
                        pending.delete(label);
                    } catch (e) {
                        // hard failure (e.g. dropdown option gone) — don't retry
                        pending.delete(label);
                        failed++;
                        unmatched.push(tab + ': ' + label + ' (' + e.message + ')');
                    }
                }
                if (pending.size) {
                    // gradio 6 mounts a closed accordion's contents only when it opens.
                    // If pending labels are section-prefixed ("Hires. fix > Denoising
                    // strength"), expand the matching accordion so the next rescan can
                    // find them. Click each accordion at most ONCE per restore (the
                    // body can take >1s to mount — a second click would toggle it shut
                    // again), then poll until its inputs actually appear.
                    let revealed = false;
                    const pathSegs = new Set();
                    for (const l of pending.keys()) {
                        for (const seg of String(l).split(' > ').slice(0, -1)) {
                            const s = seg.trim();
                            if (s) pathSegs.add(normLabel(s));
                        }
                    }
                    for (const acc of activeTabRoot().querySelectorAll('.gradio-accordion')) {
                        const title = (acc.querySelector('.label-wrap span')?.textContent || '')
                            .trim().split('\n')[0].trim();
                        const nt = normLabel(title);
                        if (!nt || ![...pathSegs].some(s => nt.startsWith(s) || s.startsWith(nt))) continue;
                        const bodyHasInputs = () => [...acc.querySelectorAll('input, textarea, select')]
                            .some(el => !el.closest('.label-wrap'));
                        if (!bodyHasInputs() && !expandedAccordions.has(title)) {
                            expandedAccordions.add(title);
                            const head = acc.querySelector('.label-wrap');
                            if (head) {
                                // InputAccordions (Refiner, Hires fix…) treat a header
                                // click as ENABLE — expand those without toggling the value
                                if (acc.classList.contains('input-accordion') && acc.expandOnly) acc.expandOnly();
                                else head.click();
                                revealed = true;
                                for (let w = 0; w < 10 && !bodyHasInputs(); w++) await sleep(250);
                            }
                        }
                    }
                    // gradio 6 also unmounts INACTIVE inner-tab panes (ControlNet
                    // Unit 0/1/2, Segment Anything's sub-tabs, Replacer's Generation/
                    // Detection/Inpainting). Click ONE not-yet-selected inner tab whose
                    // name matches a pending path segment per pass — successive passes
                    // walk through all needed tabs, applying each pane's values.
                    if (!revealed) {
                        // one click per TAB STRIP per pass: sibling tabs are mutually
                        // exclusive (clicking a second would unmount the first before
                        // its values apply), but tabs in different strips can be
                        // opened in the same pass.
                        const clickedStrips = new Set();
                        const innerTabs = [...activeTabRoot().querySelectorAll('[role=tab], .tab-nav button')]
                            .filter(b => visible(b) && !tabBtnSelected(b));
                        for (const b of innerTabs) {
                            const strip = b.closest('.tabs') || b.parentElement;
                            if (clickedStrips.has(strip)) continue;
                            const name = normLabel(b.textContent);
                            if (!name || clickedTabs.has(name)) continue;
                            if (![...pathSegs].some(s => s === name || s.startsWith(name) || name.startsWith(s))) continue;
                            clickedTabs.add(name);
                            clickedStrips.add(strip);
                            b.click();
                            revealed = true;
                            await sleep(600);
                        }
                    }
                    // nothing left to reveal and nothing applied this pass -> give up early
                    if (!revealed && pass > 1) {
                        const controls2 = scanControls();
                        const stillFindable = [...pending.keys()].some(l =>
                            !!(controls2.find(x => x.label === l) || findControl(controls2, l)));
                        if (!stillFindable) break;
                    }
                    await sleep(500);   // let Gradio reveal/rename dependents
                }
            }
            failed += pending.size;
            for (const label of pending.keys()) unmatched.push(tab + ': ' + label + ' (not found)');
        }
        if (activeTab) await switchTab(activeTab);
        setActivity('');
        if (unmatched.length) console.log('[forge-ai] restore unmatched:', unmatched);
        window.__faiLastUnmatched = unmatched;   // diagnosable from the console/devtools
        return { applied, failed, unmatched };
    }
    // NOTE: only Forge UI state (settings & prompts per tab) is saved. The AI
    // bot itself (chat, run log, best/reference images) deliberately starts
    // FRESH each session — restore never brings the old conversation back.
    function autosaveSession() {
        clearTimeout(sessionSaveTimer);
        sessionSaveTimer = setTimeout(async () => {
            try {
                captureUiSnapshot();   // fold in the latest control values
                if (!Object.keys(uiSnapshots).length) return;   // never clobber a good save with nothing
                await apiJSON('/forge-ai/session/save', {
                    v: 2, ts: Date.now(),
                    uiSnapshots: uiSnapshots,
                    uiActiveTab: uiActiveTab,
                }, 60000);
            } catch (e) { /* autosave is best-effort */ }
        }, 3000);
    }

    // Restores Forge UI state ONLY (settings & prompts across all saved tabs).
    // The AI bot's chat/run log is never restored — it starts fresh each session.
    async function restoreSession() {
        try {
            const r = await apiJSON('/forge-ai/session/latest', undefined, 60000);
            if (!r || !r.exists || !r.state || !r.state.uiSnapshots || !Object.keys(r.state.uiSnapshots).length) {
                sysMsg('No saved settings snapshot to restore yet.');
                return;
            }
            const s = r.state;
            uiRestoring = true;
            try {
                const res = await applyUiSnapshots(s.uiSnapshots, s.uiActiveTab);
                uiSnapshots = s.uiSnapshots;
                uiActiveTab = s.uiActiveTab || uiActiveTab;
                panel.style.display = 'flex';
                sysMsg('↺ settings & prompts restored across ' + Object.keys(s.uiSnapshots).length
                    + ' tab(s) from ' + (s.ts ? new Date(s.ts).toLocaleString() : 'unknown time')
                    + ': ' + res.applied + ' value(s) applied'
                    + (res.failed ? ' (' + res.failed + ' could not be matched — a model/extension may have changed)' : ''));
            } finally {
                uiRestoring = false;
            }
        } catch (e) {
            uiRestoring = false;
            sysMsg('Restore failed: ' + e.message);
        }
    }

    // "↺ Restore session" + the profile dropdown/💾 live in Forge's top bar,
    // next to the UI mode radio. Gradio can rebuild that area, so this is
    // re-run periodically (id-guarded).
    async function refreshProfileList(sel) {
        try {
            const r = await apiJSON('/forge-ai/profiles');
            const names = (r && r.profiles) || [];
            sel.innerHTML = '';
            const ph = document.createElement('option');
            ph.value = '';
            ph.textContent = names.length ? '— apply a profile —' : '— no saved profiles —';
            sel.appendChild(ph);
            for (const n of names) {
                const o = document.createElement('option');
                o.value = n; o.textContent = n;
                sel.appendChild(o);
            }
        } catch (e) { /* backend not up yet */ }
    }

    function injectRestoreBtn() {
        try {
            const root = gradioApp();
            if (root.querySelector('#fai-session-tools')) return;
            const anchor = root.querySelector('#forge_ui_preset') || root.querySelector('#quicksettings');
            if (!anchor) return;

            const wrap = document.createElement('div');
            wrap.id = 'fai-session-tools';

            const b = document.createElement('button');
            b.id = 'fai-restore-session';
            b.textContent = '↺ Restore session';
            b.title = 'Put back the settings & prompts from your last session, in every tab you used. The AI chat itself always starts fresh.';
            b.onclick = () => restoreSession();

            const row = document.createElement('div');
            row.className = 'fai-profile-row';
            const sel = document.createElement('select');
            sel.id = 'fai-profile-select';
            sel.title = 'Saved settings profiles — pick one to apply it to all tabs';
            const saveBtn = document.createElement('button');
            saveBtn.id = 'fai-profile-save';
            saveBtn.textContent = '💾';
            saveBtn.title = 'Save the current settings & prompts (all tabs visited this session) as a named profile';
            row.append(sel, saveBtn);
            wrap.append(b, row);
            anchor.insertAdjacentElement('afterend', wrap);

            refreshProfileList(sel);

            saveBtn.onclick = async () => {
                const name = (window.prompt('Save current settings as profile name:') || '').trim();
                if (!name) return;
                captureUiSnapshot(true);   // force — grab the freshest values now
                try {
                    const r = await apiJSON('/forge-ai/profiles/save', {
                        name: name,
                        state: { ts: Date.now(), uiSnapshots: uiSnapshots, uiActiveTab: uiActiveTab, kinds: lastScanKinds },
                    }, 60000);
                    if (r && r.ok) {
                        const kindStr = Object.entries(lastScanKinds).map(([k, n]) => n + ' ' + k).join(', ');
                        sysMsg('💾 profile "' + name + '" saved (' + Object.keys(uiSnapshots).length + ' tab(s): ' + kindStr + ').');
                        refreshProfileList(sel);
                    } else {
                        sysMsg('Profile save failed: ' + ((r && r.error) || 'unknown error'));
                    }
                } catch (e) { sysMsg('Profile save failed: ' + e.message); }
            };

            sel.onchange = async () => {
                const name = sel.value;
                sel.value = '';
                if (!name) return;
                try {
                    const r = await apiJSON('/forge-ai/profiles/get?name=' + encodeURIComponent(name), undefined, 60000);
                    if (!r || !r.ok || !r.state || !r.state.uiSnapshots) {
                        sysMsg('Profile load failed: ' + ((r && r.error) || 'empty profile'));
                        return;
                    }
                    uiRestoring = true;
                    try {
                        const res = await applyUiSnapshots(r.state.uiSnapshots, r.state.uiActiveTab);
                        sysMsg('📂 profile "' + name + '" applied: ' + res.applied + ' value(s)'
                            + (res.failed ? ' (' + res.failed + ' could not be matched)' : ''));
                    } finally {
                        uiRestoring = false;
                    }
                } catch (e) {
                    uiRestoring = false;
                    sysMsg('Profile load failed: ' + e.message);
                }
            };
        } catch (e) { /* DOM not ready yet */ }
    }

    let stopRequested = false;   // set by the ⏹ send-button while busy
    let chaseMode = true;        // keep optimizing until the user stops
    let emptyNudges = 0;         // consecutive turns where the model did nothing
    let lastReplyKey = '';       // for the repeat/stuck-loop detector
    let repeatCount = 0;

    async function agentTurn() {
        for (let round = 0; round < 100000; round++) {   // effectively unlimited; ⏹ is the terminator
            if (stopRequested) { sysMsg('⏹ stopped by you.'); return; }
            // fold in any message the user sent mid-chase so they can steer live
            await drainQueueLive();
            if (stopRequested) { sysMsg('⏹ stopped by you.'); return; }
            const controls = scanControls();
            // stable system prompt + conversation (images cached) + volatile state
            // as a trailing message (only this recomputes when settings change)
            const payload = { messages: [
                { role: 'system', content: systemPrompt(controls) },
                ...messagesForSend(),
                { role: 'user', content: liveContext(controls) },
            ] };
            // Status label: only NEW images cost vision-encode time — everything
            // sent on a previous turn is already in the server's KV cache (which
            // now survives the VRAM juggle). Count new vs remembered separately
            // so "4 images" doesn't read like a full re-read.
            let nImg = 0, nNew = 0;
            const urlsThisTurn = new Set();
            for (const m of payload.messages) {
                if (!Array.isArray(m.content)) continue;
                for (const p of m.content) {
                    if (p.type !== 'image_url' || !p.image_url) continue;
                    nImg++;
                    urlsThisTurn.add(p.image_url.url);
                    if (!sentImageUrls.has(p.image_url.url)) nNew++;
                }
            }
            const busyLabel = nNew > 0
                ? `👁 reading ${nNew} new image${nNew > 1 ? 's' : ''}${nImg > nNew ? ` (${nImg - nNew} already in memory)` : ''} + reasoning…`
                : (nImg > 0 ? `🧠 thinking… (${nImg} image${nImg > 1 ? 's' : ''} in memory)` : '🧠 thinking…');
            // Resilient chat: keep reconnecting until it succeeds or the user
            // stops. text-gen can 500 (model warming), park mid-generation, or
            // die outright — each is recoverable, so never give up on a hiccup.
            let resp = null;
            for (let attempt = 1; !stopRequested; attempt++) {
                try {
                    setActivity(attempt === 1 ? busyLabel : `🔄 reconnecting to the LLM (try ${attempt})…`);
                    resp = await apiJSON('/forge-ai/chat', payload, 900000);   // 15 min — Thinking model reasons long
                    if (resp.error) throw new Error(resp.error);
                    break;   // success
                } catch (e) {
                    if (attempt === 1) sysMsg('Chat hiccup (' + e.message + ') — recovering the LLM…');
                    // is text-gen actually down, or just reloading? recover accordingly
                    let up = false;
                    try { const s = await apiJSON('/forge-ai/status'); up = !!(s.textgen_api_ready && s.model_loaded); } catch (e2) { /* status unreachable */ }
                    if (!up) {
                        setActivity('🔄 restarting the LLM…');
                        try { await startLLM(); } catch (e3) { /* keep trying */ }
                        await waitForLLM(300000);
                    }
                    await sleep(Math.min(2000 * attempt, 8000));   // backoff
                    // after ~6 tries with no luck, pause rather than spin forever
                    if (attempt >= 6) {
                        setActivity('');
                        sysMsg('⚠ Could not reach the LLM after several tries. It may need a manual restart (⏻) or a Forge restart. Your chat is saved — send a message to retry.');
                        return;
                    }
                }
            }
            if (stopRequested) { sysMsg('⏹ stopped by you.'); return; }
            if (!resp) return;
            setActivity('');
            // reply arrived → every image in this payload is now in the server's
            // KV cache; future turns only pay for images not in this set
            sentImageUrls = urlsThisTurn;

            const raw = resp.reply || '';
            // Thinking models wrap reasoning in <think>…</think>. Separate it:
            // the reasoning is shown dimmed but NOT used for tool extraction (it
            // may contain example JSON) and is NOT kept in history (it's ephemeral).
            let think = '', reply = raw;
            const tm = raw.match(/<think>([\s\S]*?)<\/think>/i);
            if (tm) {
                think = tm[1].trim();
                reply = (raw.slice(0, tm.index) + raw.slice(tm.index + tm[0].length)).trim();
            } else if (/<think>/i.test(raw) && !/<\/think>/i.test(raw)) {
                // <think> opened but never closed = reasoning got cut off
                think = raw.replace(/<think>/i, '').trim();
                reply = '';
            }
            if (think) renderThinking(think);
            messages.push({ role: 'assistant', content: reply || '(thinking…)' });
            persistChat();
            renderAssistant(reply);

            const tools = extractTools(reply);
            const truncated = resp.finish_reason === 'length' || (think && !reply);
            if (truncated) sysMsg('⚠ reply hit the token limit — asking the model to finish what was cut off');

            // LOOP-BREAKER: if the model emits the same reply twice in a row it
            // is stuck (usually fighting a guardrail) — stop and hand to the user
            // rather than spinning until text-gen OOMs.
            const replyKey = reply.replace(/\s+/g, ' ').trim().slice(0, 300);
            if (replyKey && replyKey === lastReplyKey) {
                repeatCount++;
                if (repeatCount >= 2) {
                    repeatCount = 0; lastReplyKey = '';
                    sysMsg('⏸ the assistant is repeating itself (stuck) — I stopped it. It likely wants to change something a guardrail is protecting (e.g. the mask). Tell it what to do, or say the mask/detection is fine.');
                    return;
                }
            } else { repeatCount = 0; lastReplyKey = replyKey; }

            // FORCE FOLLOW-THROUGH: if it said it would change the prompt but did
            // not actually edit a prompt field, make it do so (it habitually
            // talks about the prompt then pulls a slider instead). BUT never fire
            // when it made an inspection/navigation call (get_image, unlock, etc.)
            // — those are legitimate steps and must be allowed to execute.
            const navOrInspect = tools.some(t => ['get_image', 'unlock_detection', 'lock_detection', 'switch_tab', 'revert', 'click', 'verdict', 'remember'].includes(t.tool));
            const saidPrompt = /(adjust|refin|change|updat|modif|edit|rewrit|improv|tweak|emphasi|add to)[^.]{0,45}(positive|negative)[^.]{0,10}prompt/i.test(reply);
            const didPromptSet = tools.some(t => t.tool === 'set' && /positive prompt|negative prompt/i.test(String(t.label)) && !/detection/i.test(String(t.label)));
            if (saidPrompt && !didPromptSet && !navOrInspect && !stopRequested) {
                sysMsg('↩ it described a prompt change but did not make one — requiring it');
                messages.push({ role: 'user', content: '[system] You said the PROMPT is the problem but you did not edit it — you changed a slider instead. That is the failure loop. NOW issue the actual edit: {"tool":"set","label":"Positive prompt","value":"<the full new positive prompt>"} (and/or "Negative prompt"), then {"tool":"generate"}. Write the real prompt text; do not adjust a slider this turn.' });
                persistChat();
                continue;
            }
            if (!tools.length) {
                // the model is deliberately asking the user — STOP and wait, do
                // NOT nudge it (nudging just makes it repeat the same question)
                if (/NEED INPUT:/i.test(reply)) {
                    emptyNudges = 0;
                    sysMsg('⏸ waiting for your answer.');
                    return;
                }
                // the model TRIED to call tools but its JSON was unparseable —
                // tell it instead of silently doing nothing
                if (/"tool"\s*:/.test(reply)) {
                    sysMsg('⚠ the model\'s tool JSON was malformed — asking it to re-issue');
                    messages.push({ role: 'user', content: '[tool error] your tool call JSON could not be parsed. Re-issue each tool call as its OWN fenced ```json block containing exactly one object, double quotes, no trailing commas, no comments.' });
                    persistChat();
                    continue;
                }
                if (truncated) {
                    messages.push({ role: 'user', content: '[system] your reply was cut off at the token limit. Continue where it stopped — re-issue any unfinished tool calls, keep prose minimal.' });
                    persistChat();
                    continue;
                }
                // chase mode: the model talked but took no action — nudge it to
                // keep optimizing rather than ending the turn. Cap consecutive
                // nudges so a genuine question to the user isn't steamrolled.
                if (chaseMode && !stopRequested) {
                    emptyNudges++;
                    if (emptyNudges <= 2) {
                        messages.push({ role: 'user', content: '[system] keep chasing a better result: make your next change (or re-roll the seed to explore a new composition) and {"tool":"generate"}. If you are genuinely blocked and need my input, say exactly "NEED INPUT:" followed by your question.' });
                        persistChat();
                        continue;
                    }
                    sysMsg('⏸ paused — the assistant stopped acting. Send a message or it will resume on the next.');
                    emptyNudges = 0;
                }
                return;
            }
            emptyNudges = 0;
            if (stopRequested) { sysMsg('⏹ stopped by you — reply shown, tools not executed.'); return; }
            setActivity('⚙ applying changes…');
            let followup = await executeTools(tools, controls);
            setActivity('');
            if (truncated) {
                const note = { type: 'text', text: '[system] your previous reply was cut off at the token limit — if any tool calls were truncated, re-issue the missing ones now, compactly.' };
                if (!followup) followup = [note];
                else followup.push(note);
            }
            if (!followup) return;
            messages.push({ role: 'user', content: followup });
            persistChat();
        }
        sysMsg('(safety stop after 50 tool rounds — send a message to continue)');
    }

    // ------------------------------------------------------------- UI

    let panel, msgsEl, inputEl, statusDot, vramEl, startBtn, stopBtn, attachPreview, activityEl, sendBtnEl, queueEl, providerBtn;
    let queuePaused = false;   // set when auto-start fails or user hits ⏹ — next send resumes

    // Permanent state bar. Steady state (running/off/parked/generating) is
    // always shown; transient activity (thinking, booting, applying) overrides
    // it with a pulsing style while in progress.
    let transientActivity = '';
    let steadyState = '⚪ LLM off';

    function renderActivity() {
        if (!activityEl) return;
        activityEl.textContent = transientActivity || steadyState;
        activityEl.classList.toggle('fai-active', !!transientActivity);
    }

    function setActivity(text) {
        transientActivity = text || '';
        renderActivity();
    }

    function setSteady(text) {
        steadyState = text;
        renderActivity();
    }

    // model selection moved to Settings → AI Assistant (forge_ai_model); the
    // backend uses that default whenever no model is passed
    function currentModel() { return null; }

    function el(tag, cls, text) {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined) e.textContent = text;
        return e;
    }

    function sysMsg(text) {
        const d = el('div', 'fai-msg fai-sys', text);
        msgsEl.appendChild(d);
        msgsEl.scrollTop = msgsEl.scrollHeight;
    }

    function renderThinking(text) {
        const d = el('div', 'fai-msg fai-think', '🧠 ' + text);
        msgsEl.appendChild(d);
        msgsEl.scrollTop = msgsEl.scrollHeight;
    }

    function renderAssistant(text) {
        // hide tool blocks (they show as ⚙ lines when executed) but keep any
        // non-tool code blocks visible so nothing disappears silently
        const clean = text.replace(/```\w*\s*([\s\S]*?)```/g,
            (full, inner) => /"tool"\s*:/.test(inner) ? '' : full).trim();
        if (!clean) return;
        const d = el('div', 'fai-msg fai-assistant', clean);
        msgsEl.appendChild(d);
        msgsEl.scrollTop = msgsEl.scrollHeight;
    }

    function renderUser(text, nImages, queued) {
        const d = el('div', 'fai-msg fai-user' + (queued ? ' fai-queued' : ''),
            text + (nImages ? `  [${nImages} image${nImages > 1 ? 's' : ''}]` : ''));
        // queued messages wait in their own strip at the bottom; they move
        // into the conversation the moment the agent consumes them
        ((queued && queueEl) ? queueEl : msgsEl).appendChild(d);
        msgsEl.scrollTop = msgsEl.scrollHeight;
        return d;
    }

    // Messages queue while the assistant is mid-revision and are processed in
    // order once it frees up (including after a VRAM-juggle LLM restart).
    const sendQueue = [];

    // Queued-but-not-yet-processed messages survive a page reload too — without
    // this, the FIRST message of a session (queued while the LLM cold-boots)
    // vanished if the page was refreshed before the turn started.
    function persistQueue() {
        try { sessionStorage.setItem('fai_queue', JSON.stringify(sendQueue.map(it => it.text))); } catch (e) { /* quota — skip */ }
    }

    function onSend() {
        const text = inputEl.value.trim();
        if (!text) return;
        inputEl.value = '';
        inputEl.style.height = '';   // collapse the auto-grown box back to 2 rows
        queuePaused = false;   // a new send always resumes the queue
        emptyNudges = 0;
        const item = { text, images: pendingImages.slice() };
        pendingImages = [];
        attachPreview.textContent = '';
        item.bubble = renderUser(text, item.images.length, busy || sendQueue.length > 0);
        sendQueue.push(item);
        persistQueue();
        processQueue();
    }

    // If the LLM stack isn't up, bring it up (unload Forge weights, boot or
    // warm-reload text-gen) so the user can just type without pressing ⏻.
    async function ensureLLM() {
        try {
            const s = await apiJSON('/forge-ai/status');
            provider = s.provider || provider;
            if (provider === 'claude') {
                if (s.claude_ready) return true;
                sysMsg('☁ Claude selected but no API key found. Set ANTHROPIC_API_KEY or create extensions/forge-ai-assistant/anthropic_key.txt.');
                return false;
            }
            if (s.textgen_api_ready && s.model_loaded) return true;
        } catch (e) { /* fall through to start */ }
        if (provider === 'claude') return false;
        sysMsg('LLM is off — starting it for you… (cold boot can take a minute, warm reload ~10s)');
        try {
            await startLLM();
        } catch (e) {
            sysMsg('Failed to start the LLM: ' + e.message);
            return false;
        }
        const ok = await waitForLLM(300000);
        sysMsg(ok ? 'LLM ready.' : 'LLM did not come up — check the text-gen console window.');
        return ok;
    }

    function setBusyUI(on) {
        if (!sendBtnEl) return;
        sendBtnEl.textContent = on ? '⏹' : '➤';
        sendBtnEl.title = on ? 'Stop the assistant (finishes the current step, keeps the chat)' : 'Send';
    }

    // Move one queued message into the live conversation (bubble + content).
    async function consumeQueueItem(item) {
        item.bubble.classList.remove('fai-queued');
        if (queueEl && item.bubble.parentElement === queueEl) msgsEl.appendChild(item.bubble);
        msgsEl.scrollTop = msgsEl.scrollHeight;
        const autoParts = autoSee ? await autoImageParts() : [];
        const nImages = item.images.length + autoParts.filter(p => p.type === 'image_url').length;
        let content;
        if (nImages) {
            content = [{ type: 'text', text: item.text }];
            item.images.forEach(u => content.push({ type: 'image_url', image_url: { url: u } }));
            content.push(...autoParts);
        } else {
            content = item.text;
        }
        messages.push({ role: 'user', content: content });
        persistChat();
        persistQueue();   // it's in the chat now — drop it from the pending set
    }

    // Called each round of the chase loop so a message sent mid-run steers the
    // chase live instead of sitting stuck behind the never-ending turn.
    async function drainQueueLive() {
        let drained = 0;
        while (sendQueue.length && !stopRequested) {
            await consumeQueueItem(sendQueue.shift());
            drained++;
        }
        if (drained) sysMsg('↪ folding in your message…');
        return drained;
    }

    async function processQueue() {
        if (busy) return;   // the running loop below will pick up new items
        busy = true;
        stopRequested = false;
        setBusyUI(true);
        try {
            while (sendQueue.length && !stopRequested) {
                while (restoring) await sleep(1000);   // LLM coming back up — wait
                if (!(await ensureLLM())) {
                    queuePaused = true;
                    sysMsg('⏸ your message is held in the queue — the LLM could not start. Fix the issue (see text-gen console) and press Enter/send to retry.');
                    break;
                }
                await consumeQueueItem(sendQueue.shift());
                try {
                    await agentTurn();
                } catch (e) {
                    sysMsg('⚠ Something went wrong mid-turn: ' + e.message + ' — your chat history is intact, send another message to continue.');
                }
            }
        } finally {
            busy = false;
            stopRequested = false;
            setBusyUI(false);
            setActivity('');
        }
    }

    async function onAttach(which) {
        const img = await captureImage(which);
        if (!img) { sysMsg('No ' + which + ' image found on this tab.'); return; }
        pendingImages.push(img);
        attachPreview.textContent = pendingImages.length + ' image(s) attached to next message';
    }

    async function onStart() {
        startBtn.disabled = true;
        try {
            sysMsg('Unloading Forge model & starting text-gen… (model load can take a minute)');
            await startLLM();
            const ok = await waitForLLM(300000);
            sysMsg(ok ? 'LLM ready. VRAM is now with text-gen; Forge reloads its model automatically on the next generate.'
                      : 'LLM did not respond in time — check the text-gen console window.');
        } catch (e) {
            sysMsg('Start failed: ' + e.message);
        } finally {
            startBtn.disabled = false;
        }
    }

    async function onStop() {
        try {
            await apiJSON('/forge-ai/textgen/stop', { mode: 'kill' }, 90000);
            sysMsg('LLM fully shut down — VRAM and RAM freed.');
        } catch (e) {
            sysMsg('Stop failed: ' + e.message);
        }
    }

    async function refreshStatus() {
        try {
            const s = await apiJSON('/forge-ai/status');
            provider = s.provider || provider;
            if (providerBtn && providerBtn.value !== provider) providerBtn.value = provider;
            if (provider === 'claude_code' || provider === 'claude') {
                const label = provider === 'claude_code' ? 'Claude Code' : 'Claude API';
                statusDot.className = 'fai-dot ' + (s.claude_ready ? 'fai-on' : 'fai-off');
                statusDot.title = s.claude_ready ? (label + ' ready') : (label + ' selected — not connected');
                if (s.vram) vramEl.textContent = `${label} · Forge VRAM ${s.vram.free_gb}/${s.vram.total_gb} GB free`;
                return;
            }
            const ready = s.textgen_api_ready && s.model_loaded;
            const mid = !ready && (s.textgen_proc || s.textgen_api_ready);
            statusDot.className = 'fai-dot ' + (ready ? 'fai-on' : (mid ? 'fai-mid' : 'fai-off'));
            statusDot.title = s.sleeping ? 'LLM hibernated — VRAM free, wakes in ~1.5s'
                : (ready ? 'LLM ready'
                    : (s.textgen_api_ready ? 'server warm, model unloaded (fast start)'
                        : (s.textgen_proc ? 'llama-server starting…' : 'LLM off')));
            if (s.vram) vramEl.textContent = `VRAM ${s.vram.free_gb} / ${s.vram.total_gb} GB free`;
        } catch (e) { /* ignore */ }
    }

    function buildUI() {
        const fab = el('div', null, '🤖');
        fab.id = 'fai-fab';
        fab.title = 'AI Assistant';

        panel = el('div');
        panel.id = 'fai-panel';
        panel.style.display = 'none';

        const header = el('div', 'fai-header');
        statusDot = el('span', 'fai-dot fai-off');
        startBtn = el('button', 'fai-btn', '⏻ Start LLM');
        stopBtn = el('button', 'fai-btn', '⏹ Stop');
        stopBtn.title = 'Fully shut down text-gen (frees VRAM and RAM). The automatic VRAM juggle uses fast soft-unloads instead.';
        const eyeBtn = el('button', 'fai-btn fai-eye-on', '👁');
        eyeBtn.title = 'Auto-attach source & result images (on)';
        const bestBtn = el('button', 'fai-btn', '⭐');
        bestBtn.title = 'Mark the current result as the best — future judging compares against it';
        bestBtn.onclick = async () => {
            let img = null;
            try { img = await captureImage('result'); } catch (e) { /* none */ }
            if (!img) { sysMsg('No current result to mark as best.'); return; }
            const lastGen = settingsHistory.length ? settingsHistory[settingsHistory.length - 1] : null;
            const gen = lastGen ? lastGen.n : (genCounter || 1);
            bestResult = { url: img, gen: gen };
            tieStreak = 0;
            if (lastGen) { lastGen.verdict = 'better'; lastGen.note = 'user marked as best'; lastGen.img = img; persistRuns(); }
            sysMsg(`⭐ marked generation #${gen} as the best — judging now compares against it`);
            // tell the running assistant so it stops arguing for the old best
            if (busy || sendQueue.length) {
                sendQueue.push({ text: `[user override] I marked generation #${gen} as the BEST. Use it as the baseline; judge future results against it. Do not revert to any earlier "best".`, images: [], bubble: renderUser(`⭐ set #${gen} as best`, 0, true) });
                persistQueue();
            }
        };
        providerBtn = document.createElement('select');
        providerBtn.className = 'fai-provider';
        providerBtn.title = 'AI driver mode';
        [['local', '🧠 Local'], ['claude_code', '🤝 Claude Code'], ['claude', '☁ Claude API']].forEach(([v, t]) => {
            const o = document.createElement('option');
            o.value = v; o.textContent = t;
            providerBtn.appendChild(o);
        });
        const clearBtn = el('button', 'fai-btn', '🗑');
        clearBtn.title = 'Clear chat';
        const closeBtn = el('button', 'fai-btn', '✕');
        header.append(statusDot, providerBtn, startBtn, stopBtn, bestBtn, eyeBtn, clearBtn, closeBtn);
        providerBtn.onchange = async () => {
            const next = providerBtn.value;
            try {
                const r = await apiJSON('/forge-ai/provider', { provider: next });
                provider = r.provider;
                if (provider === 'claude_code') {
                    sysMsg(r.claude_ready
                        ? '🤝 Switched to Claude Code — your Claude Code session is driving. No VRAM juggle; Forge keeps its model.'
                        : '🤝 Claude Code selected, but no takeover session is active yet. In your Claude Code session, tell it to take over the Forge session.');
                } else if (provider === 'claude') {
                    sysMsg(r.claude_ready ? '☁ Switched to Claude API (cloud, billed to your key). No VRAM juggle.'
                        : '☁ Claude API selected but no key found — set ANTHROPIC_API_KEY or anthropic_key.txt.');
                } else {
                    sysMsg('🧠 Switched to Local (Qwen). VRAM juggle re-enabled.');
                }
                refreshStatus();
            } catch (e) { sysMsg('Provider switch failed: ' + e.message); }
        };
        eyeBtn.onclick = () => {
            autoSee = !autoSee;
            eyeBtn.className = 'fai-btn' + (autoSee ? ' fai-eye-on' : '');
            eyeBtn.title = 'Auto-attach source & result images (' + (autoSee ? 'on' : 'off') + ')';
        };
        eyeBtn.onclick = () => {
            autoSee = !autoSee;
            eyeBtn.className = 'fai-btn' + (autoSee ? ' fai-eye-on' : '');
            eyeBtn.title = 'Auto-attach source & result images (' + (autoSee ? 'on' : 'off') + ')';
        };

        vramEl = el('div', 'fai-vram', '');
        activityEl = el('div', 'fai-activity', '');
        msgsEl = el('div', 'fai-msgs');
        queueEl = el('div', 'fai-queue');
        attachPreview = el('div', 'fai-attach-preview', '');

        const inputRow = el('div', 'fai-input-row');
        inputEl = document.createElement('textarea');
        inputEl.className = 'fai-input';
        inputEl.placeholder = 'e.g. "the inpainted face looks mushy, help me fix it"';
        inputEl.rows = 2;
        // grow with the text (up to the CSS max-height) so long messages stay visible
        inputEl.addEventListener('input', () => {
            inputEl.style.height = 'auto';
            inputEl.style.height = Math.min(inputEl.scrollHeight + 2, 180) + 'px';
        });
        const attachRes = el('button', 'fai-btn', '📷');
        attachRes.title = 'Attach latest result image';
        const attachSrc = el('button', 'fai-btn', '🖼');
        attachSrc.title = 'Attach source/inpaint image';
        const refBtn = el('button', 'fai-btn', '📎');
        refBtn.title = 'Upload a reference image for the AI to match/use as a target';
        const refInput = document.createElement('input');
        refInput.type = 'file';
        refInput.accept = 'image/*';
        refInput.style.display = 'none';
        refBtn.onclick = () => refInput.click();
        refInput.onchange = () => {
            const f = refInput.files && refInput.files[0];
            if (!f) return;
            const reader = new FileReader();
            reader.onload = async () => {
                let url = String(reader.result);
                // reference is the TARGET to match precisely — feed it at high
                // resolution/quality so the model sees the details it must copy.
                try { url = await shrinkDataUrl(url, 1568, 0.92); } catch (e) { /* use original */ }
                referenceImage = url;
                sysMsg('📎 reference image uploaded (high-res) — the AI will use it as a target');
                // hand it to the assistant right away as a pinned reference
                messages.push({ role: 'user', content: [
                    { type: 'text', text: '[REFERENCE IMAGE uploaded by the user — this is a TARGET/reference to match (pose, lighting, style, anatomy, or whatever the user asks). Use it to guide your edits. It stays available for the rest of the session.]' },
                    { type: 'image_url', image_url: { url: url } },
                ] });
                persistChat();
                renderUser('📎 reference image', 1, false);
            };
            reader.readAsDataURL(f);
            refInput.value = '';
        };
        sendBtnEl = el('button', 'fai-btn fai-send', '➤');
        inputRow.append(inputEl, refInput, refBtn, attachSrc, attachRes, sendBtnEl);

        panel.append(header, vramEl, activityEl, msgsEl, queueEl, attachPreview, inputRow);
        renderActivity();
        document.body.append(fab, panel);

        fab.onclick = () => {
            const show = panel.style.display === 'none';
            panel.style.display = show ? 'flex' : 'none';
            if (show) {
                refreshStatus();
                if (!statusTimer) statusTimer = setInterval(refreshStatus, 5000);
            } else if (statusTimer) {
                clearInterval(statusTimer); statusTimer = null;
            }
        };
        closeBtn.onclick = () => fab.onclick();
        startBtn.onclick = onStart;
        stopBtn.onclick = onStop;
        clearBtn.onclick = () => {
            messages = [];
            msgsEl.innerHTML = '';
            lastSeen = {};
            genCounter = 0;
            settingsHistory.length = 0;
            bestResult = null;
            detectionLocked = true;
            sawMask = false;
            referenceImage = null;
            cnFailCount = 0;
            tieStreak = 0;
            sentImageUrls = new Set();
            try { sessionStorage.removeItem('fai_chat'); sessionStorage.removeItem('fai_runs'); sessionStorage.removeItem('fai_queue'); } catch (e) { }
            clearTimeout(botSaveTimer);   // cancel any pending autosave that would resurrect it
            apiJSON('/forge-ai/botstate/clear', {}).then(() => sysMsg('🗑 bot memory cleared — fresh start.')).catch(() => { });
        };
        sendBtnEl.onclick = () => {
            if (busy) {
                stopRequested = true;
                queuePaused = true;   // don't auto-resume queued messages after a manual stop
                sysMsg('⏹ stop requested — finishing the current step…');
            } else {
                onSend();
            }
        };
        attachRes.onclick = () => onAttach('result');
        attachSrc.onclick = () => onAttach('source');
        inputEl.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); }
        });

        apiJSON('/sdapi/v1/options').then(o => { forgePreset = o.forge_preset || null; }).catch(() => { });
        refreshMemory();
        refreshGuidance();

        sysMsg('Hi! Just tell me what you\'re trying to fix — I\'ll start the LLM myself if it\'s off. I can read and move every slider on this tab, and I automatically see your inpaint image and latest result (👁 toggles that). If you run a generation, I\'ll free the VRAM for it and come back after.');

        // restore the run log from before a page reload
        try {
            const runs = JSON.parse(sessionStorage.getItem('fai_runs') || 'null');
            if (runs && Array.isArray(runs.settingsHistory)) {
                genCounter = runs.genCounter || 0;
                settingsHistory.length = 0;
                settingsHistory.push(...runs.settingsHistory);
            }
        } catch (e) { /* start fresh */ }

        // restore chat: same-tab reload first (freshest), otherwise the bot's
        // own saved state from the server (survives browser/Forge restarts,
        // cleared only by 🗑)
        let restoredFromTab = false;
        try {
            const saved = JSON.parse(sessionStorage.getItem('fai_chat') || 'null');
            if (Array.isArray(saved) && saved.length) {
                messages = saved;
                for (const m of saved) {
                    const text = Array.isArray(m.content)
                        ? m.content.filter(p => p.type === 'text').map(p => p.text).join(' ')
                        : String(m.content || '');
                    if (!text) continue;
                    if (m.role === 'assistant') renderAssistant(text);
                    else renderUser(text.length > 400 ? text.slice(0, 400) + '…' : text, 0, false);
                }
                restoredFromTab = true;
                sysMsg('(chat restored after page reload)');
            }
        } catch (e) { /* corrupt save — start fresh */ }
        if (!restoredFromTab) botRestoreLatest();

        // restore messages that were still queued (never made it into the chat)
        // when the page reloaded — e.g. the first question of a session sent
        // while the LLM was cold-booting. Re-queue and process them.
        try {
            const pending = JSON.parse(sessionStorage.getItem('fai_queue') || 'null');
            if (Array.isArray(pending) && pending.length) {
                for (const text of pending) {
                    const item = { text, images: [] };
                    item.bubble = renderUser(text.length > 400 ? text.slice(0, 400) + '…' : text, 0, true);
                    sendQueue.push(item);
                }
                sysMsg(`(restored ${pending.length} unsent message${pending.length > 1 ? 's' : ''} from before the reload — sending…)`);
                setTimeout(() => processQueue(), 1500);
            }
        } catch (e) { /* corrupt save — start fresh */ }

        // top-bar "↺ Restore session" button + hint when a previous session
        // exists after a full Forge/browser restart (sessionStorage is empty)
        injectRestoreBtn();
        setInterval(injectRestoreBtn, 10000);

        // EVENT-DRIVEN capture: scanning the whole page is expensive (forced
        // reflows on 250+ controls), so it only runs after the user actually
        // edited something — never on an idle timer. This was a major source
        // of UI jank when it polled every 4s.
        let uiDirty = false;
        const markUiDirty = () => { uiDirty = true; };
        gradioApp().addEventListener('input', markUiDirty, true);
        gradioApp().addEventListener('change', markUiDirty, true);
        setInterval(() => {
            if (!uiDirty || document.hidden || busy || uiRestoring) return;
            uiDirty = false;
            captureUiSnapshot();
            const j = uiActiveTab && uiSnapshots[uiActiveTab] ? JSON.stringify(uiSnapshots[uiActiveTab]) : '';
            if (j && j !== lastUiSnapJson) {
                lastUiSnapJson = j;
                autosaveSession();
            }
        }, 5000);
        // final snapshot on page close — catches tweaks made in the last few
        // seconds (e.g. ControlNet changes right before closing Forge)
        window.addEventListener('beforeunload', () => {
            try {
                captureUiSnapshot(true);
                if (!Object.keys(uiSnapshots).length) return;   // never clobber a good save with nothing
                const blob = new Blob(
                    [JSON.stringify({ v: 2, ts: Date.now(), uiSnapshots: uiSnapshots, uiActiveTab: uiActiveTab })],
                    { type: 'application/json' });
                navigator.sendBeacon('/forge-ai/session/save', blob);
            } catch (e) { /* best-effort */ }
        });
        apiJSON('/forge-ai/session/info').then(i => {
            if (i && i.exists) {
                sysMsg('💾 Settings & prompts from your last session are saved' + (i.ts ? ' (' + new Date(i.ts).toLocaleString() + ')' : '')
                    + ' — click "↺ Restore session" (top left, next to the UI mode) to put them back. The chat always starts fresh.');
            }
        }).catch(() => { });

        // worker-paced so it keeps running when the tab is unfocused
        (async () => {
            for (;;) {
                await sleep(3000);
                try { await watchdog(); } catch (e) { /* keep ticking */ }
            }
        })();
    }

    onUiLoaded(buildUI);
})();
