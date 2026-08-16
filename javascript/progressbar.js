// code related to showing and updating progressbar shown as the image is being made

function rememberGallerySelection() {

}

function getGallerySelectedIndex() {

}

function request(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x) {
    return x < 10 ? '0' + x : x;
}

function formatTime(secs) {
    if (secs > 3600) {
        return pad2(Math.floor(secs / 60 / 60)) + ":" + pad2(Math.floor(secs / 60) % 60) + ":" + pad2(Math.floor(secs) % 60);
    } else if (secs > 60) {
        return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
    } else {
        return Math.floor(secs) + "s";
    }
}


var originalAppTitle = undefined;

onUiLoaded(function() {
    originalAppTitle = document.title;
});

function setTitle(progress) {
    var title = originalAppTitle;

    if (opts.show_progress_in_title && progress) {
        title = '[' + progress.trim() + '] ' + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}


function randomId() {
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

// The generate event's own JS only runs once gradio dispatches that listener,
// which happens after the other same-click listeners (e.g. the ControlNet unit
// state refreshes) — noticeably after the physical click. These native capture
// listeners give instant feedback: the moment a generate button is pressed, a
// "Queued…" bar appears and the button flips to Interrupt. requestProgress
// adopts/replaces the placeholder when it starts for real.
function showQueuedPlaceholder(parent, before, tab) {
    if (parent.querySelector(':scope > .progressDiv')) return;   // one already live
    var div = document.createElement('div');
    div.className = 'progressDiv pending-placeholder';
    div.style.display = opts.show_progressbar ? "block" : "none";
    var inner = document.createElement('div');
    inner.className = 'progress queued';
    inner.style.width = '100%';
    // NOT "Queued…": nothing has been sent yet, let alone queued. With more
    // than one person on a server, "queued" has to mean the one thing it
    // means on the server -- waiting behind somebody else's run.
    inner.textContent = 'Sending…';
    div.appendChild(inner);
    parent.insertBefore(div, before);
    // Self-recovery. requestProgress removes this placeholder the instant the
    // real submit reaches the backend (it runs synchronously in submit()). If
    // it's STILL a pending-placeholder after 12s, the submit never went through
    // — a stale gradio queue/websocket after idle or an error. Remove it AND
    // restore the Generate button so the user can just click again instead of
    // being stuck on "Queued…" with the button hidden (needing a page reload).
    setTimeout(function() {
        if (div.parentNode && div.classList.contains('pending-placeholder')) {
            div.parentNode.removeChild(div);
            if (tab) showSubmitButtons(tab, true);
        }
    }, 12000);
}

onAfterUiUpdate(function() {
    for (var btn of gradioApp().querySelectorAll('button[id$="_generate"]:not([data-instant-queued])')) {
        var tab = btn.id.slice(0, -"_generate".length);
        var container = gradioApp().getElementById(tab + '_gallery_container');
        if (!container) continue;
        btn.dataset.instantQueued = '1';
        btn.addEventListener('click', (function(tab, container) {
            return function() {
                showSubmitButtons(tab, false);
                // pass `tab`: the 12s self-recovery inside only restores the
                // Generate button `if (tab)`, so omitting it left the button
                // hidden forever when a submit never reached the backend
                showQueuedPlaceholder(container.parentNode, container, tab);
            };
        })(tab, container), true);
    }
});

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
// "Pause after first preview": while a run is paused the Generate button comes
// back as Resume, so the control that started the run is the control that
// continues it. Its click is caught HERE, on document capture, which runs
// before both the instant-placeholder listener on the button itself and
// gradio's own submit handler — otherwise pressing Resume would queue a second
// generation on top of the paused one.
function resumePairFor(tabname) {
    var app = gradioApp();
    return {
        resume: app.getElementById(tabname + '_skip'),        // right half
        cancel: app.getElementById(tabname + '_interrupt')     // left half
    };
}

function setResumeMode(tabname, on) {
    var p = resumePairFor(tabname);
    if (!p.resume || !p.cancel) return;                        // e.g. Replacer
    if (on) {
        if (p.resume.dataset.resumeMode === '1') return;
        p.resume.dataset.resumeMode = '1';
        p.resume.dataset.labelBeforeResume = p.resume.textContent;
        p.cancel.dataset.labelBeforeResume = p.cancel.textContent;
        p.resume.textContent = '▶ Resume';
        p.cancel.textContent = '✕ Cancel';
        p.resume.style.display = 'block';
        p.cancel.style.display = 'block';
    } else if (p.resume.dataset.resumeMode === '1') {
        delete p.resume.dataset.resumeMode;
        p.resume.textContent = p.resume.dataset.labelBeforeResume || 'Skip';
        p.cancel.textContent = p.cancel.dataset.labelBeforeResume || 'Interrupt';
        // visibility is left alone: the run carries on, and these two are
        // exactly the buttons a running job should be showing anyway
    }
}

// Resume rides on the SKIP button. The generate box shows one control at a
// time: while a run is alive, Generate is hidden and Interrupt/Skip overlay
// the box as two 50% halves. Rather than give Generate a geometry it does not
// have (it is the box's only in-flow child — making it an absolute half would
// collapse the box), the pair that is already sized, positioned and rounded
// becomes "▶ Resume | ✕ Cancel". Cancel needs no interception at all: it is
// Interrupt, and interrupting is exactly what breaks wait_while_paused.
document.addEventListener('click', function(ev) {
    var btn = ev.target && ev.target.closest ? ev.target.closest('button[id$="_skip"]') : null;
    if (!btn || btn.dataset.resumeMode !== '1') return;
    // capture phase on DOCUMENT, so this beats gradio's own handler on the
    // button — otherwise Resume would ALSO skip the image it just paused on
    ev.preventDefault();
    ev.stopImmediatePropagation();
    var tabname = btn.id.slice(0, -'_skip'.length);
    setResumeMode(tabname, false);
    fetch('./internal/resume', {method: 'POST'}).catch(function() {
        // the run lives on the server; if this never lands, put Resume back
        // rather than leaving a dead button (Cancel still works regardless)
        setResumeMode(tabname, true);
    });
}, true);

function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    var dateStart = new Date();
    var wasEverActive = false;
    var parentProgressbar = progressbarContainer.parentNode;
    var tabname = (progressbarContainer.id || '').replace(/_gallery_container$/, '');
    var wakeLock = null;

    // replace the instant placeholder from the click listener with the real bar
    for (var stale of parentProgressbar.querySelectorAll(':scope > .progressDiv.pending-placeholder')) {
        parentProgressbar.removeChild(stale);
    }

    // Clear any live-preview left behind by an earlier run (see
    // clearLivePreviews) so a new generation never starts with a stale preview
    // pinned in front of the gallery.
    if (gallery) {
        try {
            gallery.querySelectorAll(':scope > .livePreview').forEach(function(el) {
                if (el.parentNode) el.parentNode.removeChild(el);
            });
        } catch (e) { /* gallery mid-rebuild */ }
    }

    var requestWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || wakeLock) return;
        try {
            wakeLock = await navigator.wakeLock.request('screen');
        } catch (err) {
            console.error('Wake Lock is not supported.');
        }
    };

    var releaseWakeLock = async function() {
        if (!opts.prevent_screen_sleep_during_generation || !wakeLock) return;
        try {
            await wakeLock.release();
            wakeLock = null;
        } catch (err) {
            console.error('Wake Lock release failed', err);
        }
    };

    var divProgress = document.createElement('div');
    divProgress.className = 'progressDiv';
    divProgress.style.display = opts.show_progressbar ? "block" : "none";
    var divInner = document.createElement('div');
    divInner.className = 'progress';
    // show something the instant the button is clicked — the server only
    // reports the task once the generate event reaches it, which can take a
    // moment (queued state refreshes run first), and an empty 0-width bar
    // reads as "nothing happened". The first poll replaces this within
    // ~500ms with whatever the server actually says.
    divInner.classList.add('queued');
    divInner.style.width = '100%';
    divInner.textContent = 'Sending…';

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    var livePreview = null;
    // Poll-failure tolerance. request()'s error path fires on ANY non-200 or
    // parse error -- a single transient blip (server briefly busy, a dropped
    // request while the GPU is saturated) used to tear the whole progress UI
    // down mid-run. That also calls atEnd(), which clears the stored task id,
    // which in turn destroys the watchdog's ability to recover the result: one
    // hiccup and a still-running generation looked finished and became
    // unrecoverable. Tolerate a few consecutive failures instead.
    var progressFailures = 0;
    var previewFailures = 0;
    var MAX_POLL_FAILURES = 5;
    var stopped = false;

    var removeProgressBar = function() {
        releaseWakeLock();
        stopped = true;
        if (!divProgress) return;

        setTitle("");
        // Cancelling WHILE paused would otherwise leave the pair reading
        // "Resume / Cancel" for a job that no longer exists — and those labels
        // persist, since the buttons are reused by the next run.
        if (tabname) setResumeMode(tabname, false);
        parentProgressbar.removeChild(divProgress);
        divProgress = null;          // set BEFORE the sweep: any late img.onload
                                     // must see the run as over (see funLivePreview)
        clearLivePreviews();
        atEnd();
    };

    // Remove every live-preview node from this gallery, not just the one this
    // closure happens to hold. A preview whose <img> decoded after the run
    // finished used to insert a FRESH .livePreview in front of the results,
    // which nothing ever cleaned up -- the gallery then showed a stale preview
    // instead of the new image, and the next run's selection logic tripped over
    // the orphan ("gallery stuck").
    function clearLivePreviews() {
        livePreview = null;
        if (!gallery) return;
        try {
            gallery.querySelectorAll(':scope > .livePreview').forEach(function(el) {
                if (el.parentNode) el.parentNode.removeChild(el);
            });
        } catch (e) { /* gallery mid-rebuild */ }
    }

    var funProgress = function(id_task) {
        requestWakeLock();
        request("./internal/progress", {id_task: id_task, live_preview: false}, function(res) {
            if (res.completed) {
                removeProgressBar();
                return;
            }

            let progressText = "";

            if (res.progress > 0) {
                divInner.classList.remove('queued');
                divInner.style.width = (res.progress * 100.0) + '%';
                progressText = (res.progress * 100.0).toFixed(0) + '%';
            } else {
                // not started yet: keep the full-width queued look
                divInner.classList.add('queued');
                divInner.style.width = '100%';
            }

            if (res.eta) {
                progressText += " ETA: " + formatTime(res.eta);
            }

            setTitle(progressText);

            // The server's own status line, minus the two strings the stage
            // below already says better ("Waiting..." for a task it has never
            // heard of, and its queue position, which we reword).
            var info = '';
            if (res.textinfo && res.textinfo.indexOf("\n") == -1
                    && !/^Waiting\.\.\.$/.test(res.textinfo)
                    && !/^In queue:/i.test(res.textinfo)) {
                info = res.textinfo;
            }

            // Name the stage the server is actually in. The words used before
            // were whatever the API happened to put in textinfo -- "Waiting..."
            // for a task the server has never heard of, which reads as though
            // the SERVER is waiting when in fact nothing has reached it, and
            // "Queued…" for the model load, which reads as though nothing has
            // started when the run is already underway. Neither answered the
            // question worth answering while the bar sits still: has the server
            // got this, and what is it doing with it?
            //
            // add_task_to_queue() runs the moment the server enters the
            // handler, so `queued` means ACCEPTED; the wait after that is the
            // queue_lock -- another user's run, or another tab's. `active` with
            // no progress yet is the model load, which on a large checkpoint is
            // the longest silent stretch of a run. When the server is telling
            // us what it is doing (extras, merging), that wins over our guess.
            var stage = '';
            if (res.paused) {
                // the run is alive and holding on the sampler thread; the
                // preview under the bar is what there is to judge
                stage = 'Paused after first preview — Resume or Cancel';
            } else if (res.active) {
                if (!res.progress && !info) stage = 'Loading…';
            } else if (res.queued) {
                var pos = /In queue:\s*(\d+)\s*\/\s*(\d+)/i.exec(res.textinfo || '');
                stage = pos
                    ? 'Accepted — ' + pos[1] + ' of ' + pos[2] + ' in queue'
                    : 'Accepted — waiting for the GPU…';
            } else {
                stage = 'Waiting for the server…';
            }

            divInner.textContent = [stage, info, progressText]
                .filter(Boolean).join(' ').trim() || 'Sending…';

            if (tabname) setResumeMode(tabname, !!res.paused);

            var elapsedFromStart = (new Date() - dateStart) / 1000;

            if (res.active) wasEverActive = true;

            if (!res.active && wasEverActive) {
                removeProgressBar();
                return;
            }

            if (elapsedFromStart > inactivityTimeout && !res.queued && !res.active) {
                removeProgressBar();
                return;
            }

            if (onProgress) {
                onProgress(res);
            }

            progressFailures = 0;

            // Worker-paced: a hidden tab throttles main-thread timers to ~1/min,
            // which stalls progress AND the completion detection below.
            forgeTimer.setTimeout(() => {
                if (!stopped) funProgress(id_task, res.id_live_preview);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
            if (!stopped && ++progressFailures < MAX_POLL_FAILURES) {
                forgeTimer.setTimeout(() => {
                    if (!stopped) funProgress(id_task);
                }, 1000);
                return;
            }
            removeProgressBar();
        });
    };

    var funLivePreview = function(id_task, id_live_preview) {
        request("./internal/progress", {id_task: id_task, id_live_preview: id_live_preview}, function(res) {
            if (!divProgress) {
                return;
            }

            if (res.live_preview && gallery) {
                var img = new Image();
                img.onload = function() {
                    // Decode is ASYNC: the run can finish between the response
                    // above and this callback. Without re-checking, a late
                    // preview re-inserts itself into a gallery that is already
                    // showing the final result.
                    if (stopped || !divProgress) return;
                    if (!livePreview) {
                        livePreview = document.createElement('div');
                        livePreview.className = 'livePreview';
                        gallery.insertBefore(livePreview, gallery.firstElementChild);
                    }

                    livePreview.appendChild(img);
                    if (livePreview.childElementCount > 2) {
                        livePreview.removeChild(livePreview.firstElementChild);
                    }
                };
                img.src = res.live_preview;
            }

            previewFailures = 0;

            forgeTimer.setTimeout(() => {
                if (!stopped) funLivePreview(id_task, res.id_live_preview);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
            // A failed PREVIEW poll is cosmetic -- it must never tear down the
            // run's progress tracking. Retry a few times, then just stop
            // previewing; funProgress still owns completion detection.
            if (!stopped && ++previewFailures < MAX_POLL_FAILURES) {
                forgeTimer.setTimeout(() => {
                    if (!stopped) funLivePreview(id_task, id_live_preview);
                }, 1000);
            }
        });
    };

    funProgress(id_task, 0);

    if (gallery) {
        funLivePreview(id_task, 0);
    }

}
