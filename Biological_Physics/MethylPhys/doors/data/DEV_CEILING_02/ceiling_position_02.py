"""DEV-CEILING-02: where does a tumor sit between healthy and the derived ceiling? (written 2026-10-13, BEFORE it was run)

The gauge is fixed at both ends by physics: the floor by thermal kicks against one ATP, the CEILING by a coin flip at every identity
site - one bit per site, reached when sites the cell holds methylated average beta = 1/2 (CANON/foundation.json, entropy_ceiling).
Only the middle - the healthy reference - is measured. So a tumor's position can be written WITHOUT UNITS, as the fraction of the
headroom between its own healthy tissue and the ceiling that it has used:

    u = (H_tumor - H_normal) / (1 - H_normal)          H = mean per-site binary entropy in bits over the sealed sites
    u = 0  reads as its own healthy tissue;  u = 1  sits at the ceiling;  u < 0  more ordered than its own normal.

Both arms are measured, because both approach the SAME ceiling from opposite sides: the methylated channel falls toward 1/2 (loss),
the unmethylated channel rises toward 1/2 (gain). u_m and u_u are reported separately and never averaged.

What is read, and nothing else: the exact sites and the exact tumor/normal pairs each cancer was SEALED and SCORED on (sealed_*.json),
calibrated by the chain's own Stage 1. Nothing is re-chosen.
  breast    DEV-TISSUE-BREAST-01   53 pairs      colorectal  DEV-TISSUE-COLON-01   88 pairs
  liver     DEV-TISSUE-LIVER-01    50 pairs      lung        DEV-TISSUE-LUNG-01    28 pairs (that test read NO FINGERPRINT)

DATA SPENDING: every one of these sets is already read, sealed and scored. This is DEVELOPMENT use - a first descriptive look that
cannot change any sealed verdict and cannot later serve as a held-out test of a ceiling claim. Any claim that comes out of it must be
sealed and tested on cohorts nobody has opened.

THE READING RULE, fixed here before any number exists (not a seal, because nothing here is held out):
  - per cancer: the median u_m and u_u, with a bootstrap 95 % interval over pairs (2,000 resamples, seed fixed).
  - "consistent across cancers" is said ONLY if the four u_m intervals all overlap AND the spread of the four medians (SD / mean) is
    under 0.25. Otherwise the record says "not consistent at this depth", and what differs.
  - every pair whose methylated sites average below 1/2 is counted as PAST THE CEILING and reported, never dropped.

KNOWN BIAS, stated before running: every tumor sample is a mixture of tumor and normal cells, and the normal cells pull the reading
back toward healthy. Bulk tissue therefore gives a LOWER BOUND on the tumor cells' own position. Correcting it needs each sample's
purity (next step), and the sequenced cancer cell lines - pure tumor cells read molecule by molecule - are the other bracket.

A planted check runs first, on made-up numbers, and the script refuses to read any tumor if it fails: a normal read against itself
must give u = 0, a planted sample at beta = 1/2 must give u = 1, and a sample halfway in entropy must give u = 1/2.

# NO CHAIN: Stage 1 already produced every beta read here; this computes the gauge position from them with the same binary entropy
# the chain uses (dev_comp_methods.H), imported, not re-written.

Writes ceiling_position_02_rows.csv (one row per pair) and ceiling_position_02_summary.txt.
Usage: python3 ceiling_position_02.py WORKDIR      (WORKDIR holds the calibrated beta folders named below)
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W = sys.argv[1]
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")))
from dev_comp_methods import H
DATA = os.path.abspath(os.path.join(HERE, ".."))
SETS = [("breast", "DEV_TISSUE_BREAST_01/sealed_breast_tissue.json", "tcga_brca/betas", "tcga_brca_t/betas", "normal_sample", "tumour_sample"),
        ("colorectal", "DEV_TISSUE_COLON_01/sealed_colon_tissue.json", "colon/betas", "colon_t/betas", "normal_gsm", "tumour_gsm"),
        ("liver", "DEV_TISSUE_LIVER_01/sealed_liver_tissue.json", "lihc/betas", "lihc_t/betas", "normal_sample", "tumour_sample"),
        ("lung", "DEV_TISSUE_LUNG_01/sealed_lung_01.json", "lung/betas", "lung_t/betas", "normal_gsm", "tumour_gsm")]

def Hb(x): return float(np.mean(H(np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6))))
def u_of(bt, bn):
    hn, ht = Hb(bn), Hb(bt); return (ht - hn) / (1 - hn), hn, ht

# planted check, on made-up numbers, before any tumor is read
rg = np.random.default_rng(1)
bn = np.clip(rg.normal(0.92, 0.03, 3000), 0, 1)
hn = Hb(bn); target = hn + 0.5 * (1 - hn)
from scipy.optimize import brentq
half = np.full(3000, brentq(lambda b: Hb(np.full(10, b)) - target, 0.5, 0.999))
chk = [("normal against itself", u_of(bn, bn)[0], 0.0), ("planted at beta = 1/2", u_of(np.full(3000, 0.5), bn)[0], 1.0)]
# a uniform sample halfway in entropy: compare against a uniform normal so the target is exact
bn_u = np.full(3000, 0.92); hn_u = Hb(bn_u)
mid = brentq(lambda b: Hb(np.full(10, b)) - (hn_u + 0.5 * (1 - hn_u)), 0.5, 0.92)
chk.append(("halfway in entropy", u_of(np.full(3000, mid), bn_u)[0], 0.5))
L = ["DEV-CEILING-02: the tumor's position between its own healthy tissue (u = 0) and the derived ceiling (u = 1)", "",
     "PLANTED CHECK (made-up numbers, before any tumor is read)"]
okp = True
for name, got, want in chk:
    ok = abs(got - want) < 1e-6; okp &= ok; L.append(f"  {name:<24} u = {got:+.6f}  (must be {want})  {'ok' if ok else 'FAIL'}")
if not okp:
    L.append("PLANTED CHECK FAILED: no tumor read."); open(os.path.join(HERE, "ceiling_position_02_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); sys.exit(1)

def beta(path):
    b = pd.read_parquet(path); return b["beta"] if "beta" in b.columns else b.iloc[:, 0]
rows = []
for cancer, seal, nd, td, nk, tk in SETS:
    S = json.load(open(os.path.join(DATA, seal)))
    for p in S["pairs"]:
        fn, ft = os.path.join(W, nd, p[nk] + ".parquet"), os.path.join(W, td, p[tk] + ".parquet")
        if not (os.path.isfile(fn) and os.path.isfile(ft)):
            rows.append(dict(cancer=cancer, pair=p.get("case") or p.get("pid"), note="calibrated file missing")); continue
        n, t = beta(fn), beta(ft); r = dict(cancer=cancer, pair=p.get("case") or p.get("pid"), note="")
        for ch, key in (("m", "sites_methylated"), ("u", "sites_unmethylated")):
            sites = S[key]; a, b = t.reindex(sites), n.reindex(sites); ok = a.notna() & b.notna()
            uu, hn_, ht_ = u_of(a[ok].values, b[ok].values)
            r.update({f"u_{ch}": uu, f"Hn_{ch}": hn_, f"Ht_{ch}": ht_, f"beta_n_{ch}": float(b[ok].mean()),
                      f"beta_t_{ch}": float(a[ok].mean()), f"sites_{ch}": int(ok.sum())})
        r["past_ceiling"] = bool(r["beta_t_m"] < 0.5)
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "ceiling_position_02_rows.csv"), index=False)
R = T[T.note == ""]
L += ["", f"pairs read: {len(R)} of {len(T)} ({(T.note != '').sum()} with a calibrated file missing, listed in the rows file)", "",
      f"{'cancer':<11}{'pairs':>6}  {'u_m median [95 % CI]':<28}{'u_u median [95 % CI]':<28}{'past ceiling':>13}  beta_m normal -> tumor"]
med = {}
def boot(x):
    b = [np.median(rg2.choice(x, len(x))) for _ in range(2000)]; return np.percentile(b, [2.5, 97.5])
rg2 = np.random.default_rng(20261013)
for cancer, *_ in SETS:
    g = R[R.cancer == cancer]
    if not len(g): L.append(f"{cancer:<11}{0:>6}  not read"); continue
    cm, cu = boot(g.u_m.values), boot(g.u_u.values); med[cancer] = (np.median(g.u_m), cm)
    L.append(f"{cancer:<11}{len(g):>6}  {np.median(g.u_m):+.4f} [{cm[0]:+.4f}, {cm[1]:+.4f}]   {np.median(g.u_u):+.4f} [{cu[0]:+.4f}, {cu[1]:+.4f}]"
             f"   {int(g.past_ceiling.sum()):>4} of {len(g):<4}  {g.beta_n_m.median():.3f} -> {g.beta_t_m.median():.3f}")
ms = np.array([v[0] for v in med.values()]); lo = max(v[1][0] for v in med.values()); hi = min(v[1][1] for v in med.values())
overlap = lo <= hi; cv = float(np.std(ms, ddof=1) / abs(np.mean(ms))) if len(ms) > 1 else float("nan")
L += ["", f"the reading rule: all {len(ms)} u_m intervals overlap: {'yes' if overlap else 'no'} | spread of the medians (SD / mean): {cv:.3f} (rule: under 0.25)",
      ("CONSISTENT ACROSS CANCERS at this depth (development use of spent sets; a claim needs a seal on unopened cohorts)" if overlap and cv < 0.25
       else "NOT CONSISTENT across cancers at this depth - see which cancer differs above"),
      "", "Bulk tissue is a LOWER BOUND on the tumor cells' own position (normal cells in the sample pull it toward healthy)."]
open(os.path.join(HERE, "ceiling_position_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
