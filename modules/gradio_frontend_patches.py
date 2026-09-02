"""Launch-time patches to gradio's prebuilt frontend bundle.

gradio ships its UI as minified JS under gradio/templates/frontend/assets.
One hot path in it makes every click on a large app slow, and there is no
Python-side switch for it, so the bundle is patched on disk at launch:

  * `u(root, id)` -- the component-tree lookup. It is a recursive
    depth-first search over the WHOLE tree, and `update_loading_stati_state`
    calls it once per component that has ever carried a loading status, on
    EVERY event dispatch. With 1,800+ components (txt2img + img2img +
    ControlNet units + extensions) that is 100-260 ms of frozen main thread
    for a single checkbox click, growing with every tab the user opens.
    Measured on this install, headless Chromium, real clicks:
        checkbox with img2img mounted: 234 ms -> 0 ms of long-frame time
        tab switch:                    262 ms -> 0 ms
        Extras first open:            1393 ms -> 391 ms
    The patch memoises an id -> node Map per root, rebuilt on a miss and
    reset wherever the tree is restructured (rerender of a gr.render block,
    reload), so a stale node can never be returned.

Patches are exact-substring replacements guarded by a marker: applied once,
skipped if already present, skipped (with a console line) if the bundle no
longer contains the expected code -- a gradio upgrade then simply runs
unpatched instead of breaking. Re-applied automatically after a venv
reinstall because the original signature is back.

Browsers cache the asset by URL (the hashed filename does not change), so
a page loaded before the patch keeps the old code until a hard reload.
"""

import glob
import os

MARKER = "/*forge-patched:tree-lookup*/"

# (description, original, replacement, expected occurrences) -- every
# pattern must match exactly that many times or the file is left alone.
TREE_LOOKUP_PATCH = [
    (
        "memoised component-tree lookup",
        "function u(s,t){if(s.id===t)return s;if(s.children)for(const i of s.children){const e=u(i,t);if(e)return e}return null}",
        MARKER
        + "var __uCache=new WeakMap();"
        + "function __uIndex(s){const m=new Map();(function w(n){m.set(n.id,n);if(n.children)for(const c of n.children)w(c)})(s);__uCache.set(s,m);return m}"
        + "function u(s,t){let m=__uCache.get(s);if(!m)m=__uIndex(s);let r=m.get(t);if(r===undefined){m=__uIndex(s);r=m.get(t)}return r||null}",
        1,
    ),
    (
        # gr.render swaps a subtree under the SAME root object
        "cache reset on rerender",
        "o.children=n.children}",
        "o.children=n.children;__uCache=new WeakMap()}",
        1,
    ),
    (
        # reload() rebuilds root.children in place (two class variants)
        "cache reset on reload",
        "this.root.children=this.#i.children.map(",
        "__uCache=new WeakMap(),this.root.children=this.#i.children.map(",
        2,
    ),
]


def _patch_file(path, patches):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        return "already"
    for _desc, orig, _new, count in patches:
        if src.count(orig) != count:
            return None   # signature absent or ambiguous: leave the file alone
    for _desc, orig, new, _count in patches:
        src = src.replace(orig, new)
    tmp = path + ".forge-tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(src)
    os.replace(tmp, path)
    return "patched"


def apply():
    try:
        import gradio
        assets = os.path.join(os.path.dirname(gradio.__file__), "templates", "frontend", "assets")
        candidates = glob.glob(os.path.join(assets, "index-*.js"))
    except Exception as e:
        print(f"[frontend patch] skipped: {e}")
        return
    status = None
    for path in candidates:
        try:
            r = _patch_file(path, TREE_LOOKUP_PATCH)
        except OSError as e:
            print(f"[frontend patch] {os.path.basename(path)}: {e}")
            continue
        if r:
            status = (r, os.path.basename(path))
            break
    if status is None:
        print("[frontend patch] tree-lookup signature not found in this gradio build; running unpatched")
    elif status[0] == "patched":
        print(f"[frontend patch] {status[1]}: component-tree lookup memoised (hard-reload the page once to pick it up)")
