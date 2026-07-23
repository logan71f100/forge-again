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
    inner.textContent = 'Queued…';
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
                showQueuedPlaceholder(container.parentNode, container);
            };
        })(tab, container), true);
    }
});

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    var dateStart = new Date();
    var wasEverActive = false;
    var parentProgressbar = progressbarContainer.parentNode;
    var wakeLock = null;

    // replace the instant placeholder from the click listener with the real bar
    for (var stale of parentProgressbar.querySelectorAll(':scope > .progressDiv.pending-placeholder')) {
        parentProgressbar.removeChild(stale);
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
    // show a queued state the instant the button is clicked — the server only
    // reports the task once the generate event reaches it, which can take a
    // moment (queued state refreshes run first), and an empty 0-width bar
    // reads as "nothing happened"
    divInner.classList.add('queued');
    divInner.style.width = '100%';
    divInner.textContent = 'Queued…';

    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    var livePreview = null;

    var removeProgressBar = function() {
        releaseWakeLock();
        if (!divProgress) return;

        setTitle("");
        parentProgressbar.removeChild(divProgress);
        if (gallery && livePreview) gallery.removeChild(livePreview);
        atEnd();

        divProgress = null;
    };

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

            if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
                progressText = res.textinfo + " " + progressText;
            }

            divInner.textContent = progressText || 'Queued…';

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

            setTimeout(() => {
                funProgress(id_task, res.id_live_preview);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
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

            setTimeout(() => {
                funLivePreview(id_task, res.id_live_preview);
            }, opts.live_preview_refresh_period || 500);
        }, function() {
            removeProgressBar();
        });
    };

    funProgress(id_task, 0);

    if (gallery) {
        funLivePreview(id_task, 0);
    }

}
