"""globe_viewer.py - the one viewer for every IAM globe (2026-10-21): a report's globe, a report's own full page, and the site's Globe Explorer.
A globe is a spec: {id, name, about, assets: {name: picture}, groups: [{name, layers: [layer]}]}; a layer is
  {kind: "image", src}                        a picture shown as it is
  {kind: "change", src, base, full, hint, fixed} the fixed change (2026-10-21): code 0 no site, 128 unchanged, 128 +/- k changed (size k/127 x
                                              full); the slider fades only the background (unchanged sites and the base picture) from full to none;
                                              the changed sites and their count (fixed) never move
  {kind: "diff", src, base, scale, above}     a coded difference (0 no site; 1..255 = -0.5..+0.5 x scale); a slider fades everything below the
                                              chosen size into the base picture's grey; above[t] = share of the measured globe beyond step t
  {kind: "steps", srcs, labels, notes}        a slider steps through pictures (the noise maps: 1, 2, 4 ... 100 draws averaged)
No data dependency: the site build imports it with the standard library only."""
import os, json, base64

def dedupe(spec):
    """Every picture stored once, in spec["assets"]; layers name it (a reference is both a layer and a difference's grey background)."""
    A, ix = {}, {}
    def one(v):
        if not isinstance(v, str) or not v.startswith("data:image/"): return v
        if v not in ix: ix[v] = f"a{len(ix)}"; A[ix[v]] = v
        return ix[v]
    for g in spec["groups"]:
        for L in g["layers"]:
            for k in ("src", "base"):
                if L.get(k): L[k] = one(L[k])
            if L.get("srcs"): L["srcs"] = [one(x) for x in L["srcs"]]
    spec["assets"] = A; return spec

VIEWER_JS = r"""
const IAMCache = {};
function IAMTex(src) { return IAMCache[src] || (IAMCache[src] = new Promise((ok, no) => { const im = new Image(); im.onload = () => {
  const o = document.createElement('canvas'); o.width = im.width; o.height = im.height; const g = o.getContext('2d'); g.drawImage(im, 0, 0);
  ok({ w: im.width, h: im.height, d: g.getImageData(0, 0, im.width, im.height).data }); }; im.onerror = () => no(src); im.src = src; })); }
function IAMGlobe(cv) { const cx = cv.getContext('2d'), s = { yaw: 0, pitch: 0.35, spin: true, drag: null, px: null, w: 1024, h: 512 };
  function draw() { if (!s.px) return; const N = cv.width, img = cx.createImageData(N, N), d = img.data, ca = Math.cos(s.yaw), sa = Math.sin(s.yaw),
      cb = Math.cos(s.pitch), sb = Math.sin(s.pitch), TW = s.w, TH = s.h;
    for (let py = 0; py < N; py++) for (let qx = 0; qx < N; qx++) { const sx = 2 * (qx + 0.5) / N - 1, sy = 1 - 2 * (py + 0.5) / N, r2 = sx * sx + sy * sy, k = 4 * (py * N + qx);
      if (r2 > 1) { d[k + 3] = 0; continue; } const x = Math.sqrt(1 - r2), y = sx, z = sy, x1 = x * cb - z * sb, z1 = x * sb + z * cb, x2 = x1 * ca - y * sa, y2 = x1 * sa + y * ca;
      const th = Math.acos(Math.max(-1, Math.min(1, z1))); let ph = Math.atan2(y2, x2); if (ph < 0) ph += 2 * Math.PI;
      const i = Math.min(TH - 1, Math.floor(th / Math.PI * TH)), j = Math.min(TW - 1, Math.floor(ph / (2 * Math.PI) * TW)), c = s.px(4 * (i * TW + j)), sh = 0.55 + 0.45 * x;
      d[k] = c[0] * sh; d[k + 1] = c[1] * sh; d[k + 2] = c[2] * sh; d[k + 3] = 255; }
    cx.putImageData(img, 0, 0); }
  cv.addEventListener('pointerdown', e => { s.drag = [e.clientX, e.clientY, s.yaw, s.pitch]; s.spin = false; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => { if (!s.drag) return; s.yaw = s.drag[2] - (e.clientX - s.drag[0]) * 0.01; s.pitch = Math.max(-1.5, Math.min(1.5, s.drag[3] + (e.clientY - s.drag[1]) * 0.01)); draw(); });
  cv.addEventListener('pointerup', () => s.drag = null);
  (function tick() { if (s.spin && s.px) { s.yaw += 0.006; draw(); } requestAnimationFrame(tick); })();
  return { draw, set(px, w, h) { s.px = px; s.w = w; s.h = h; draw(); }, toggle() { s.spin = !s.spin; } }; }
function IAMExplorer(root, specs) { const q = c => root.querySelector(c), G = IAMGlobe(q('.g-canvas'));
  const sg = q('.g-globe'), sp = q('.g-group'), sl = q('.g-layer'), rg = q('.g-range'), rv = q('.g-rv'), nt = q('.g-note'), ab = q('.g-about'), sw = q('.g-slide'), hn = q('.g-hint'), fx = q('.g-fixed');
  q('.g-spin').onclick = () => G.toggle();
  const fill = (sel, items) => { sel.innerHTML = ''; items.forEach((x, i) => sel.add(new Option(x.name, i))); sel.parentNode.style.display = items.length > 1 ? '' : 'none'; };
  fill(sg, specs); const want = new URLSearchParams(location.search).get('globe'); specs.forEach((x, i) => { if (x.id === want) sg.value = i; });
  const spec = () => specs[+sg.value], U = v => (spec().assets || {})[v] || v, grp = () => spec().groups[+sp.value], lay = () => grp().layers[+sl.value];
  function onGlobe() { fill(sp, spec().groups); ab.innerHTML = spec().about || ''; const c = new URLSearchParams(location.search).get('cell');
    spec().groups.forEach((g, i) => { if (c && g.id === c) sp.value = i; }); onGroup(); }
  function onGroup() { fill(sl, grp().layers); onLayer(); }
  function onLayer() { const L = lay(); sw.style.display = L.kind === 'image' ? 'none' : ''; hn.textContent = L.hint || ''; fx.textContent = L.fixed || '';
    if (L.kind === 'diff') { rg.max = 50; rg.value = L.start ?? 10; } else if (L.kind === 'change') { rg.max = 100; rg.value = 0; } else if (L.kind === 'steps') { rg.max = L.srcs.length - 1; rg.value = 0; } render(); }
  let tok = 0;
  async function render() { const L = lay(), t = +rg.value, my = ++tok;
    if (L.kind === 'change') { const T = await IAMTex(U(L.src)), B = L.base ? await IAMTex(U(L.base)) : null; if (my !== tok) return; const f = 1 - t / 100;
      rv.textContent = t === 0 ? 'full healthy background' : t === 100 ? 'only the change' : 'background ' + (100 - t) + ' %'; nt.textContent = L.note || '';
      G.set(k => { const c = T.d[k]; if (c === 0) return [30, 32, 38]; if (c === 128) { let g = 120; if (B) g = 105 + 0.55 * (0.3 * B.d[k] + 0.59 * B.d[k + 1] + 0.11 * B.d[k + 2]);
          const q = 22 + (g - 22) * f; return [q, q, q]; }
        const v = (c - 128) / 127, b = 0.55 + 0.45 * Math.abs(v); return v > 0 ? [255 * b, 50 * b, 45 * b] : [40 * b, 95 * b, 255 * b]; }, T.w, T.h); return; }
    if (L.kind !== 'diff') { const T = await IAMTex(U(L.kind === 'steps' ? L.srcs[t] : L.src)); if (my !== tok) return;
      if (L.kind === 'steps') rv.textContent = L.labels[t]; nt.textContent = L.kind === 'steps' ? ((L.notes || [])[t] || '') + ' ' + (L.note || '') : (L.note || '');
      G.set(k => [T.d[k], T.d[k + 1], T.d[k + 2]], T.w, T.h); return; }
    const T = await IAMTex(U(L.src)), B = L.base ? await IAMTex(U(L.base)) : null; if (my !== tok) return; const sc = L.scale || 1, th = t * sc / 100, a = (L.above || [])[t];
    rv.textContent = th.toFixed(sc >= 10 ? 1 : 2) + (L.unit ? ' ' + L.unit : '');
    nt.textContent = (a == null ? '' : (100 * a).toFixed(1) + ' % ' + (L.share_text || 'of the measured globe differs by more than this') + '. ') + (L.note || '');
    G.set(k => { const c = T.d[k]; if (c === 0) return [42, 45, 51]; const v = (c - 128) / 254 * sc, m = Math.abs(v);
      if (m <= th) { if (B) { const g = 0.3 * B.d[k] + 0.59 * B.d[k + 1] + 0.11 * B.d[k + 2], g2 = 105 + 0.55 * g; return [g2, g2, g2]; } return [95, 97, 102]; }
      const f = Math.min(1, 0.45 + 0.55 * m / (0.3 * sc)); return v > 0 ? [235 * f + 20, 40, 40] : [30, 80, 230 * f + 25]; }, T.w, T.h); }
  sg.onchange = onGlobe; sp.onchange = onGroup; sl.onchange = onLayer; rg.oninput = render; onGlobe(); }
"""

CONTROLS = """<div class="g-wrap" style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;justify-content:center">
<canvas class="g-canvas" width="__N__" height="__N__" style="max-width:100%;touch-action:none;cursor:grab"></canvas>
<div style="min-width:260px;max-width:420px"><p><label>Globe <select class="g-globe"></select></label></p><p><label>Show <select class="g-group"></select></label></p>
<p><label>Layer <select class="g-layer"></select></label></p><p class="g-slide"><input class="g-range" type="range" min="0" max="50" value="10" style="width:100%"> <span class="g-rv"></span><br><i class="g-hint"></i></p>
<p class="g-fixed" style="font-weight:600"></p>
<p class="g-note"></p><p><button class="g-spin" type="button">Stop or start turning</button></p><div class="g-about" style="font-size:0.92em"></div></div></div>"""

def fragment(specs, size=520):
    """The globe inside a report: one section, data and viewer inlined (the report stays one file that works offline)."""
    return ("<section id='sec-globe'><h2>The sample on the globe, against its healthy reference</h2>" + CONTROLS.replace("__N__", str(size))
            + "<script>" + VIEWER_JS + "IAMExplorer(document.getElementById('sec-globe'), " + json.dumps(specs) + ");</script></section>")

def page(specs, title, back=None, size=640, intro=""):
    """A full dark page for one or more globes (a report's own globe page, or the site's Globe Explorer); the specs are inlined."""
    css = ("body{background:#0b0d12;color:#e6e6e6;font:15px/1.5 system-ui,sans-serif;margin:0}main{max-width:1180px;margin:0 auto;padding:18px}"
           "a{color:#9cc3ff}select,button{background:#1b1f29;color:#e6e6e6;border:1px solid #39404f;border-radius:4px;padding:3px 6px}h1{font-weight:600}")
    nav = f"<p><a href='{back}'>&larr; back</a></p>" if back else ""
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{title}</title>"
            f"<style>{css}</style></head><body><main id='g-root'>{nav}<h1>{title}</h1>{intro}" + CONTROLS.replace("__N__", str(size))
            + "<script>" + VIEWER_JS + "IAMExplorer(document.getElementById('g-root'), " + json.dumps(specs) + ");</script></main></body></html>")

def write_files(spec, d, url):
    """Writes a spec's inlined pictures as files under d (served at url) and returns the spec pointing at them, for the site's explorer."""
    import copy; os.makedirs(d, exist_ok=True); s = copy.deepcopy(spec); n = [0]
    def one(v):
        if not isinstance(v, str) or not v.startswith("data:image/"): return v
        ext = "png" if v.startswith("data:image/png") else "jpg"; n[0] += 1; f = f"{spec['id']}_{n[0]:03d}.{ext}"
        open(os.path.join(d, f), "wb").write(base64.b64decode(v.split(",", 1)[1])); return url.rstrip("/") + "/" + f
    s["assets"] = {k: one(v) for k, v in (s.get("assets") or {}).items()}
    for g in s["groups"]:
        for L in g["layers"]:
            for k in ("src", "base"): L[k] = one(L.get(k)) if L.get(k) else L.get(k)
            if L.get("srcs"): L["srcs"] = [one(x) for x in L["srcs"]]
    return s
