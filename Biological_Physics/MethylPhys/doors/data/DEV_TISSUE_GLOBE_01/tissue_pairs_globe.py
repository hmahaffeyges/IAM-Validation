"""DEV-TISSUE-GLOBE-01 (2026-10-21; development, a picture, nothing scored). The tissue results on the globe, made by the chain: each tumor
against the same person's own adjacent normal (chain/MethylPhys_Interface/report_globe_v4.py build_pair, the globe's block rule), arrays
calibrated by chain Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta), and each page screenshotted in Chromium as a reader's browser
draws it (left, middle, right of the slider; lesson C32).
Pairs: the first 3 sealed pairs of DEV-TISSUE-COLON-01 (GSE131013, GEO) and of DEV-TISSUE-BREAST-01 (TCGA-BRCA, NCI GDC); both tests are
already scored, so these arrays are spent and nothing here can change their results. A control: two different people's normal colon
(not a self pair), to show what differences between people look like.
Usage: python3 tissue_pairs_globe.py OUTDIR   (writes the pages, the screenshots and tissue_pairs_globe.json)"""
import os, sys, re, json, time, subprocess, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H_, "..", "..", "..")); CH = os.path.join(MP, "chain")
sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
import report_globe_v4 as G
from stage_1_idat_calibration import calibrate_idat_to_beta
out = sys.argv[1]; W = os.path.join(out, "idat"); os.makedirs(W, exist_ok=True)
def geo(g):
    u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"; got = {}
    for n in sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))):
        p = os.path.join(W, n); urllib.request.urlretrieve(u + n, p); got["Grn" if "_Grn" in n else "Red"] = p
    return got
def gdc(files):
    got = {}
    for ch, f in files.items():
        p = os.path.join(W, f["file_name"]); urllib.request.urlretrieve(f"https://api.gdc.cancer.gov/data/{f['file_id']}", p); got[ch] = p
    return got
def beta(f):
    o = calibrate_idat_to_beta(f["Grn"], f["Red"], verbose=False); b = pd.Series(o[0] if isinstance(o, tuple) else o, dtype="float64")
    for q in f.values(): os.remove(q)
    return b
CO = json.load(open(os.path.join(MP, "doors", "data", "DEV_TISSUE_COLON_01", "sealed_colon_tissue.json")))["pairs"][:3]
BR = json.load(open(os.path.join(MP, "doors", "data", "DEV_TISSUE_BREAST_01", "sealed_breast_tissue.json")))["pairs"][:3]
T = pd.read_csv(os.path.join(MP, "doors", "data", "DEV_TISSUE_BREAST_01", "tcga_brca_450k_files.csv"))
jobs = [(f"colon_{p['pid']}", "colon tumor against the same person's adjacent normal", lambda p=p: beta(geo(p["tumour_gsm"])), lambda p=p: beta(geo(p["normal_gsm"])), "adjacent normal colon") for p in CO]
for p in BR:
    nf = {("Grn" if "_Grn" in r.file_name else "Red"): dict(file_id=r.file_id, file_name=r.file_name) for r in T[T["sample"] == p["normal_sample"]].itertuples()}
    jobs.append((f"breast_{p['case']}", "breast tumor against the same person's adjacent normal", lambda p=p: beta(gdc(p["tumour_files"])), lambda nf=nf: beta(gdc(nf)), "adjacent normal breast"))
jobs.append(("control_two_people", "control: two different people's normal colon (not a self pair)", lambda: beta(geo(CO[0]["normal_gsm"])), lambda: beta(geo(CO[1]["normal_gsm"])), "normal colon of another person"))
res = {}
for tag, title, fs, fr, rn in jobs:
    try:
        r = G.build_pair(fs(), fr(), rn, os.path.join(out, tag + ".html"), title); res[tag] = {k: v for k, v in r.items() if k != "spec"}; print(tag, res[tag]["blocks_changed"], "of", res[tag]["blocks"], flush=True)
    except Exception as e: res[tag] = dict(error=f"{type(e).__name__}: {e}"[:300]); print(tag, res[tag], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "playwright"], check=True); subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=True)
from playwright.sync_api import sync_playwright
from PIL import Image
def lit(png):
    a = np.asarray(Image.open(png).convert("RGB")).astype(int); n, m = a.shape[:2]; yy, xx = np.mgrid[0:n, 0:m]; disk = (yy - n / 2) ** 2 + (xx - m / 2) ** 2 < (min(n, m) / 2 - 2) ** 2
    r, g, b = a[..., 0], a[..., 1], a[..., 2]; return float((((r > 150) & (g < 110) & (b < 110)) | ((b > 150) & (r < 110)))[disk].mean())
with sync_playwright() as P:
    br = P.chromium.launch(); pg = br.new_page(viewport={"width": 1100, "height": 760})
    for tag in [t for t in res if "html" in res[t]]:
        pg.goto("file://" + res[tag]["html"]); pg.wait_for_timeout(2500); pg.click(".g-spin"); pg.wait_for_timeout(500); res[tag]["lit"] = {}
        for t in (0, 50, 100):
            pg.eval_on_selector(".g-range", f"e => {{ e.value = {t}; e.dispatchEvent(new Event('input')); }}"); pg.wait_for_timeout(1500)
            f = os.path.join(out, f"{tag}_slider{t}.png"); pg.locator(".g-canvas").screenshot(path=f); res[tag]["lit"][str(t)] = round(lit(f), 5)
    br.close()
json.dump(res, open(os.path.join(out, "tissue_pairs_globe.json"), "w"), indent=1); print(json.dumps({k: (v.get("blocks_changed"), v.get("lit")) for k, v in res.items()}))
