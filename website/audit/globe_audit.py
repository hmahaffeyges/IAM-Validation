"""globe_audit.py - every globe, group and layer the public site shows, as a reader's browser draws it (2026-10-21; author: "You need to
really go through each one and do your dilligence"). Opens the live Globe Explorer and the live example report globes in Chromium
(playwright), selects each globe, each group and each layer, moves the slider to its left end, middle and right end, screenshots the globe,
and counts the screenshot's pixels by color: red, blue, gray, dark, other (the globe's disk only). Writes OUT/audit.csv (one row per frame,
with the layer's kind, hint, fixed line and key text) and OUT/shots/*.png. Nothing is scored; the table is read by hand and by the checks
that follow it. Usage: python3 website/audit/globe_audit.py OUT [BASE_URL]"""
import os, sys, csv, re, numpy as np
from playwright.sync_api import sync_playwright
from PIL import Image
OUT = sys.argv[1]; BASE = (sys.argv[2] if len(sys.argv) > 2 else "https://hmahaffeyges.github.io/IAM-Validation/").rstrip("/") + "/"
PAGES = [("explorer", "atlas/sphere.html"), ("report_healthy", "art/examples/globe_healthy_blood.html")]
os.makedirs(os.path.join(OUT, "shots"), exist_ok=True)
def classes(p):
    a = np.asarray(Image.open(p).convert("RGB")).astype(int); h, w, _ = a.shape; yy, xx = np.mgrid[:h, :w]
    disk = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= (0.47 * min(h, w)) ** 2; r, g, b = a[..., 0][disk], a[..., 1][disk], a[..., 2][disk]
    red = (r > 140) & (r > g + 60) & (r > b + 60); blue = (b > 140) & (b > r + 60) & (b > g + 30); dark = (r + g + b) < 75
    gray = ~red & ~blue & ~dark & (np.abs(r - g) < 14) & (np.abs(g - b) < 16); n = disk.sum()
    return {k: round(float(v.sum()) / n, 4) for k, v in (("red", red), ("blue", blue), ("gray", gray), ("dark", dark))} | {"other": round(float((~(red | blue | gray | dark)).sum()) / n, 4)}
rows = []; fh = open(os.path.join(OUT, "audit.csv"), "w", newline=""); W = None
def pick(cls, ri, i):   # 2026-10-21: set a chooser directly, so a hidden one (a globe with one group) is set too
    pg.evaluate(f"(() => {{ const e = document.querySelectorAll('{cls}')[{ri}]; e.selectedIndex = {i}; e.dispatchEvent(new Event('change')); }})()")
def opts(cls, ri):
    return pg.evaluate(f"Array.from(document.querySelectorAll('{cls}')[{ri}].options).map(o => o.textContent)")
with sync_playwright() as P:
    br = P.chromium.launch(); pg = br.new_page(viewport={"width": 1300, "height": 1000})
    # 2026-10-23: the information sphere page (its own viewer): loads with no script error, draws, and every button and chooser changes the picture
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(BASE + "atlas/information_sphere.html", wait_until="networkidle", timeout=180000); pg.wait_for_timeout(4000); pg.evaluate("document.getElementById('spin').click()")
    def lit():
        return pg.evaluate("(()=>{const c=document.getElementById('c'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]+d[i+1]+d[i+2]>90)n++;return n/(d.length/4)})()")
    base = lit(); pg.locator("#c").screenshot(path=os.path.join(OUT, "shots", "info_sphere_default.png")); res = [("default", round(base, 4))]
    for bid in ("ter", "lab", "d0", "d1", "d2", "dst", "far"):
        pg.evaluate(f"document.getElementById('{bid}').click()"); pg.wait_for_timeout(700); v = lit(); res.append((bid, round(v, 4)))
        pg.locator("#c").screenshot(path=os.path.join(OUT, "shots", f"info_sphere_{bid}.png"))
    pg.evaluate("(()=>{const s=document.getElementById('one');s.selectedIndex=1;s.dispatchEvent(new Event('change'))})()"); pg.wait_for_timeout(700); res.append(("one cell", round(lit(), 4)))
    open(os.path.join(OUT, "info_sphere.txt"), "w").write("\n".join(f"{k}: share of the canvas lit {v}" for k, v in res) + f"\nscript errors: {len(errs)} {errs[:3]}\n")
    print(open(os.path.join(OUT, "info_sphere.txt")).read(), flush=True)
    assert not errs and base > 0.05, f"information sphere page: errors {errs[:2]}, lit {base}"
    for tag, path in PAGES:
        if tag == "explorer" and "--skip-atlas" in sys.argv: pass
        pg.goto(BASE + path, wait_until="networkidle", timeout=180000); pg.wait_for_timeout(3000)
        if pg.locator(".g-spin").count(): pg.evaluate("document.querySelectorAll('.g-spin').forEach(b => b.click())")   # stop the spin for stable shots
        roots = pg.locator(".g-canvas").count()
        assert roots, f"{BASE + path}: no globe on the page"   # a missing page must stop the audit, not be skipped
        for ri in range(roots):
            globes = opts(".g-globe", ri) or [""]
            for gi, gname in enumerate(globes):
                if gname: pick(".g-globe", ri, gi); pg.wait_for_timeout(800)
                for si, sname in enumerate(opts(".g-group", ri)):
                    pick(".g-group", ri, si); pg.wait_for_timeout(500)
                    for li, lname in enumerate(opts(".g-layer", ri)):
                        pick(".g-layer", ri, li); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(3000)   # every picture loaded before a screenshot
                        rg = pg.locator(".g-range").nth(ri); vis = pg.locator(".g-slide").nth(ri).is_visible()
                        mx = int(rg.get_attribute("max") or 0) if vis else 0
                        for pos in ((0, mx // 2, mx) if vis else (None,)):
                            if pos is not None: rg.evaluate(f"(e) => {{ e.value = {pos}; e.dispatchEvent(new Event('input')); }}"); pg.wait_for_timeout(1800)
                            f = re.sub(r"[^A-Za-z0-9]+", "_", f"{tag}_{gi}_{si}_{li}_{pos}")[:120] + ".png"; pg.locator(".g-canvas").nth(ri).screenshot(path=os.path.join(OUT, "shots", f))
                            t = lambda c: (pg.locator(c).nth(ri).inner_text() if pg.locator(c).nth(ri).count() else "").replace("\n", " ")[:400]
                            rows.append(dict(page=tag, globe=gname, group=sname, layer=lname, slider=("none" if pos is None else f"{pos}/{mx}"), shot=f,
                                             readout=t(".g-rv"), hint=t(".g-hint"), fixed=t(".g-fixed"), key=t(".g-key"), note=t(".g-note"), **classes(os.path.join(OUT, "shots", f))))
                            if W is None: W = csv.DictWriter(fh, fieldnames=list(rows[0])); W.writeheader()
                            W.writerow(rows[-1]); fh.flush()
                            print(tag, gname[:20], sname[:20], lname[:40], rows[-1]["slider"], {k: rows[-1][k] for k in ("red", "blue", "gray", "dark")}, flush=True)
    br.close()
fh.close()
print("frames", len(rows))
