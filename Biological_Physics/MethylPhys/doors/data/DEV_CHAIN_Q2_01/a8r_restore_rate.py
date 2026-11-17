#!/usr/bin/env python3
"""A8R (register A8; plan: register Part P '### A8R plan', written before reading). Runs ON A GITHUB MACHINE. The restore crew's rate g in human
ESCs from the open Repli-BS time course (GSE82045, Charlton et al. 2018): nascent-strand methylation after a 1 h BrdU pulse and 0-16 h chase, against
the bulk level of the same line. Not a chain reading: the series holds per-site processed beds (no molecules), so this is a calibration of one
simulator lever (the cell simulator's held_marked.d, assumed), labelled as such. Streams each file; keeps only chr1 for the held-site join.
Prediction written first: g in 0.3-1.5 per hour at held sites; nascent/bulk at 16 h >= 0.95 at held sites."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy", "numpy", "requests"], check=True)
import gzip, io, json, os, math, re, requests, numpy as np
from scipy.optimize import curve_fit
G = "https://ftp.ncbi.nlm.nih.gov/geo/samples"
def url(gsm, name): return f"{G}/{gsm[:-3]}nnn/{gsm}/suppl/{name}"
BULK = ("GSM2182522", "GSM2182522_HUES64_WT_AllCells_1hBrdu_16hArrest_bulk_processed.bed.gz")
TC = [(0, "GSM2642845", "GSM2642845_HUES64_WT_AllCells_1hBrdU_0Chase_nascent_rep3.bed.gz"),
      (1, "GSM2642846", "GSM2642846_HUES64_WT_AllCells_1hBrdU_1hChase_nascent_rep3.bed.gz"),
      (2, "GSM2642847", "GSM2642847_HUES64_WT_AllCells_1hBrdU_2hChase_nascent_rep3.bed.gz"),
      (4, "GSM2642848", "GSM2642848_HUES64_WT_AllCells_1hBrdU_4hChase_nascent_rep3.bed.gz"),
      (8, "GSM2642849", "GSM2642849_HUES64_WT_AllCells_1hBrdU_8hChase_nascent_rep3.bed.gz"),
      (16, "GSM2642850", "GSM2642850_HUES64_WT_AllCells_1hBrdU_16hChase_nascent_rep3.bed.gz")]
def lines(gsm, name):
    for k in range(5):
        try:
            r = requests.get(url(gsm, name), stream=True, timeout=300); r.raise_for_status(); r.raw.decode_content = False
            with gzip.GzipFile(fileobj=r.raw) as gz:
                for b in io.TextIOWrapper(gz, encoding="utf-8", errors="replace"): yield b
            return
        except requests.RequestException as e:
            import time; time.sleep(60 * (k + 1)); print(f"::notice::retry {gsm}: {type(e).__name__}")
    raise SystemExit(f"could not read {gsm}")
def layout(sample):
    """decide which columns give methylated and total reads from the first numeric data lines; refuse if unclear"""
    rows = [l.rstrip("\n").split("\t") for l in sample if l.strip() and not l.startswith(("track", "#", "browser"))][:2000]
    w = len(rows[0]); num = []
    for j in range(3, w):   # Repli-BS processed beds: column 4 is the quoted string 'm/t' (methylated / total reads)
        q = [r[j].strip("'\"") for r in rows]
        if all(re.fullmatch(r"\d+/\d+", x) for x in q):
            print("::notice::columns " + json.dumps(dict(width=w, first=rows[0], layout=f"m/t string in column {j}"))); return ("mt_str", j, j)
    for j in range(3, w):
        try: v = [float(r[j]) for r in rows]
        except ValueError: continue
        num.append((j, v))
    desc = [(j, min(v), max(v), all(x == int(x) for x in v)) for j, v in num]
    print("::notice::columns " + json.dumps(dict(width=w, first=rows[0], numeric=desc)))
    ints = [d for d in desc if d[3]]; frac = [d for d in desc if not d[3]]
    if len(num) >= 2 and len(ints) >= 2:
        a, b = ints[-2], ints[-1]
        va, vb = dict(num)[a[0]], dict(num)[b[0]]
        if all(x <= y for x, y in zip(va, vb)): return ("meth_total", a[0], b[0])
        return ("meth_unmeth", a[0], b[0])
    if frac and ints:
        f, c = frac[0], ints[-1]
        return ("pct_cov" if f[2] > 1.0 else "frac_cov", f[0], c[0])
    raise SystemExit("bed layout not recognised: " + json.dumps(desc))
def mt(p, lay):
    kind, i, j = lay
    if kind == "mt_str":
        m, t = p[i].strip("'\"").split("/"); return float(m), float(t)
    a, b = float(p[i]), float(p[j])
    if kind == "meth_total": return a, b
    if kind == "meth_unmeth": return a, a + b
    if kind == "pct_cov": return a / 100 * b, b
    return a * b, b
def read(gsm, name, keep_chr1=False):
    it = lines(gsm, name); head = []
    for l in it:
        head.append(l)
        if len(head) >= 2000: break
    lay = layout(head); M = T = 0.0; c1 = {}
    def take(l):
        nonlocal M, T
        p = l.rstrip("\n").split("\t")
        if len(p) <= max(lay[1], lay[2]) or p[0].startswith(("track", "#")): return
        try: m, t = mt(p, lay)
        except ValueError: return
        if t <= 0: return
        M += m; T += t
        if keep_chr1 is not None and p[0] in ("chr1", "1"): c1[int(p[1])] = (m, t)
    for l in head: take(l)
    for l in it: take(l)
    return lay, M / T, c1, T
# planted check of the fit before any data are fitted
def model(t, m0, minf, g): return minf - (minf - m0) * np.exp(-g * t)
tt = np.array([0.5, 1.5, 2.5, 4.5, 8.5, 16.5]); yy = model(tt, 0.55, 0.80, 0.7)
pp, _ = curve_fit(model, tt, yy, p0=(0.5, 0.8, 0.5)); assert abs(pp[2] - 0.7) < 1e-4, pp
print(f"::notice::planted check: fit returns g {pp[2]:.4f} for planted 0.7")
lay, mb, B1, Tb = read(*BULK, keep_chr1=True)
held = {k: v[0] / v[1] for k, v in B1.items() if v[1] >= 10 and v[0] / v[1] >= 0.80}
print(f"::notice::bulk: layout {lay[0]}, genome level {mb:.4f}, chr1 sites {len(B1)}, held chr1 sites (>=0.80, depth >=10) {len(held)}")
out = dict(bulk=dict(level=mb, held_sites=len(held)), points=[])
for ch, gsm, name in TC:
    lay, m, C1, T = read(gsm, name, keep_chr1=True)
    hm = sum(C1[k][0] for k in held if k in C1); ht = sum(C1[k][1] for k in held if k in C1)
    hb = sum(B1[k][0] for k in held if k in C1) / max(1.0, sum(B1[k][1] for k in held if k in C1))
    pt = dict(chase_h=ch, t_h=ch + 0.5, level=m, ratio_all=m / mb, held_level=hm / ht if ht else None, held_bulk=hb, ratio_held=(hm / ht) / hb if ht else None, held_reads=ht)
    out["points"].append(pt); print("::notice::A8R POINT " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in pt.items()}))
for key in ("ratio_held", "ratio_all"):
    t = np.array([p["t_h"] for p in out["points"]]); y = np.array([p[key] for p in out["points"]], dtype=float)
    pp, cov = curve_fit(model, t, y, p0=(y[0], y[-1], 0.5), maxfev=20000); se = np.sqrt(np.diag(cov))
    out[f"fit_{key}"] = dict(m0=pp[0], m_inf=pp[1], g_per_h=pp[2], g_se=se[2])
    print(f"::notice::A8R FIT {key}: g {pp[2]:.3f} +/- {se[2]:.3f} per hour; m0 {pp[0]:.4f}; m_inf {pp[1]:.4f}; at 16 h {y[-1]:.4f}")
g = out["fit_ratio_held"]["g_per_h"]; y16 = out["points"][-1]["ratio_held"]
print(f"::notice::A8R VS PREDICTION: g {g:.3f}/h vs window 0.3-1.5 -> {'MET' if 0.3 <= g <= 1.5 else 'NOT MET'}; held ratio at 16 h {y16:.4f} vs >= 0.95 -> {'MET' if y16 >= 0.95 else 'NOT MET'}")
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/a8r_restore_rate.json", "w"), indent=1)
