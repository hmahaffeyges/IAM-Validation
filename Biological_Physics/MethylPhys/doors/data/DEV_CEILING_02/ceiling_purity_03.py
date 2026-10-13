"""DEV-CEILING-02, step 2: is the cancers' disagreement a PURITY ordering? (written 2026-10-13, BEFORE it was run)

Step 1 found the tumors moved toward the derived ceiling in 193 of 219 pairs but the four cancers did not land in one place. Bulk
tissue is a mixture, and the normal cells in it pull the reading toward healthy, so a less pure cancer reads lower. Two tests of that,
on the two cancers with published purity (TCGA breast and liver; PanCanAtlas ABSOLUTE calls, tcga_absolute_purity.txt, downloaded
from the GDC 2026-10-13). Colorectal and lung come from GEO series with no published purity: NOT ASSESSED here, and said so.

TEST A, no correction needed: within each cancer, does the bulk position rise with purity? If u is a property of the tumor cells being
diluted, it must. Reported: Spearman rho of u_m against purity, per cancer, with its p value. No bar - this is a direction check.

TEST B, the correction. Per site, the tumor cells' own beta, assuming the cells mixed into the tumor sample look like the patient's
own normal tissue:
        beta_tumor_cells = (beta_sample - (1 - p) beta_normal) / p,   clipped to [0, 1]
then u on the corrected betas. Fixed before running:
  - pairs with purity below 0.30 are EXCLUDED and counted (the correction divides by p and amplifies noise 1/p times);
  - pairs with no ABSOLUTE call ("called" status) are EXCLUDED and counted;
  - the same consistency rule as step 1, applied to the two cancers: their corrected u_m intervals must overlap AND their medians
    must differ by less than a quarter of their mean.
KNOWN LIMITS, stated before running: the cells mixed into a tumor are stroma and immune cells, not exactly the adjacent normal tissue;
the clip at 0 and 1 biases sites near the ends; ABSOLUTE purity is itself an estimate. A corrected number is therefore an estimate of
the tumor cells' position, not a measurement of it, and is reported as one.

# NO CHAIN: arithmetic over the calibrated betas Stage 1 produced, with the chain's own binary entropy (dev_comp_methods.H).

Writes ceiling_purity_03_rows.csv and ceiling_purity_03_summary.txt. Usage: python3 ceiling_purity_03.py WORKDIR
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); W = sys.argv[1]
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")))
from dev_comp_methods import H
DATA = os.path.abspath(os.path.join(HERE, "..")); PMIN = 0.30
P = pd.read_csv(os.path.join(HERE, "tcga_absolute_purity.txt"), sep="\t")
P = P[P["call status"] == "called"].set_index("array")["purity"]
SETS = [("breast", "DEV_TISSUE_BREAST_01/sealed_breast_tissue.json", "tcga_brca/betas", "tcga_brca_t/betas"),
        ("liver", "DEV_TISSUE_LIVER_01/sealed_liver_tissue.json", "lihc/betas", "lihc_t/betas")]
def Hb(x): return float(np.mean(H(np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6))))
def beta(path):
    b = pd.read_parquet(path); return b["beta"] if "beta" in b.columns else b.iloc[:, 0]
U1 = pd.read_csv(os.path.join(HERE, "ceiling_position_02_rows.csv"))
rows = []
for cancer, seal, nd, td in SETS:
    S = json.load(open(os.path.join(DATA, seal))); sites = S["sites_methylated"]
    for pr in S["pairs"]:
        key = pr["tumour_sample"][:15]; p = P.get(key, np.nan)
        r = dict(cancer=cancer, pair=pr["case"], purity=p,
                 u_bulk=float(U1[(U1.cancer == cancer) & (U1.pair == pr["case"])].u_m.iloc[0]))
        if not np.isfinite(p): r["excluded"] = "no ABSOLUTE call"; rows.append(r); continue
        if p < PMIN: r["excluded"] = f"purity {p:.2f} below {PMIN}"; rows.append(r); continue
        n = beta(os.path.join(W, nd, pr["normal_sample"] + ".parquet")).reindex(sites)
        t = beta(os.path.join(W, td, pr["tumour_sample"] + ".parquet")).reindex(sites); ok = n.notna() & t.notna()
        bn, bt = n[ok].values, t[ok].values
        bc = np.clip((bt - (1 - p) * bn) / p, 0, 1)
        hn = Hb(bn); r.update(u_corrected=(Hb(bc) - hn) / (1 - hn), beta_cells=float(bc.mean()), past_ceiling=bool(bc.mean() < 0.5),
                              clipped_fraction=float(np.mean((bc == 0) | (bc == 1))), excluded="")
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "ceiling_purity_03_rows.csv"), index=False)
rg = np.random.default_rng(20261013)
def boot(x): return np.percentile([np.median(rg.choice(x, len(x))) for _ in range(2000)], [2.5, 97.5])
L = ["DEV-CEILING-02 step 2: purity", "", "TEST A - does the bulk position rise with purity, within each cancer?"]
for c, g in T.groupby("cancer", sort=False):
    g2 = g[np.isfinite(g.purity)]; rho, pv = spearmanr(g2.purity, g2.u_bulk)
    L.append(f"  {c:<7} {len(g2)} pairs with a purity call (median purity {g2.purity.median():.2f}) | Spearman rho {rho:+.3f}, p = {pv:.2g}")
L += ["", "TEST B - the tumor cells' own position, corrected for purity"]
med = {}
for c, g in T.groupby("cancer", sort=False):
    k = g[g.excluded == ""]; ci = boot(k.u_corrected.values); med[c] = (np.median(k.u_corrected), ci)
    L.append(f"  {c:<7} used {len(k)} of {len(g)} (excluded: {', '.join(f'{v} x{n}' for v, n in g[g.excluded != ''].excluded.str.split(' below').str[0].value_counts().items()) or 'none'})"
             f" | bulk median {g.u_bulk.median():+.4f} -> corrected median {med[c][0]:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
             f" | past the ceiling after correction: {int(k.past_ceiling.sum())} | sites clipped {k.clipped_fraction.median():.1%}")
(a, ca), (b, cb) = med.values(); ov = max(ca[0], cb[0]) <= min(ca[1], cb[1]); rel = abs(a - b) / abs((a + b) / 2)
L += ["", f"the rule: intervals overlap {'yes' if ov else 'no'} | medians differ by {rel:.1%} of their mean (rule: under 25 %)",
      ("CONSISTENT after correction, for these two cancers (development use; an estimate, not a measurement; a claim needs a seal)" if ov and rel < 0.25
       else "NOT CONSISTENT after correction: purity does not account for the difference between these two cancers"),
      "Colorectal and lung: NOT ASSESSED (no published purity for their series)."]
open(os.path.join(HERE, "ceiling_purity_03_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
