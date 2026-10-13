"""DEV-CEILING-02, step 3: every normal and every tumor placed ON THE CELL GAUGE (written 2026-10-13, BEFORE it was run).

Why step 3. Steps 1-2 measured each tumor from its own adjacent normal (u = 0). That is not the gauge: the gauge's middle is the
PURIFIED HEALTHY REFERENCE of the cell type, not adjacent tissue, and adjacent normal tissue may itself sit off the middle. A zero point
that moves patient by patient could produce the cross-cancer disagreement by itself. Here nothing is measured from a patient's own
tissue: both samples are read against the cell of origin's healthy reference in the atlas.

The gauge (Runtime Matrices/Gauge/cell_gauge_marks_v1.json; CANON/foundation.json gauge_direction - remember the black hole):
    A = mean H(beta_sample) / mean H(mu_reference)      over the cancer's SEALED methylated sites
    middle A = 1 (the healthy reference);  full surface A_full = 1 / mean H(mu_reference)  (every site at a coin flip, 1 bit);
    the floor (thermal kicks against one ATP) is ~1e-7, far off the left edge.
    RIGHT of 1 is over-writing toward the full surface (where cancer is expected); LEFT of 1 is more order than healthy.
Reported for every sample: A; distance from the middle, A - 1; distance to the full surface, A_full - A; the fraction of the way from
the middle to the full surface, f = (A - 1)/(A_full - 1); and, per site, the share of sites already AT the surface (H >= 0.9 bit,
beta within about 0.32-0.68) - the per-site count that an average over sites cannot hide (the author's pooling question).

References, fixed before running: breast -> breast luminal epithelium (breast basal epithelium reported alongside); colorectal ->
colon epithelium; liver -> hepatocyte; lung -> lung alveolar epithelium (lung bronchus epithelium reported alongside).

THE READING RULE, fixed here (same form as step 1, so the two can be compared):
  - per cancer: median f for tumors and for normals, with bootstrap 95 % intervals (2,000 resamples, fixed seed);
  - "a shared position across cancers" only if the four TUMOR f intervals overlap AND the medians' spread (SD / mean) is under 0.25;
  - the same rule on the per-site share of sites at the surface, tumors;
  - where the adjacent normal sits is reported for every cancer: if its interval excludes f = 0, part of each tumor's distance was
    already present in the patient's normal tissue before the tumor.
KNOWN LIMITS, stated first: tissue is a mixture of cell types, the reference is one cell type, so a normal tissue reads off 1 partly by
composition (the sealed sites were chosen to be composition-blind, which limits but does not remove this); bulk tumors are diluted by
normal cells (step 2). Development use of spent sets throughout; nothing here can change a sealed verdict.

# NO CHAIN: Stage 1 produced every beta; the chain's own binary entropy (dev_comp_methods.H) is imported, not rewritten.

Writes ceiling_gauge_04_rows.csv and ceiling_gauge_04_summary.txt. Usage: python3 ceiling_gauge_04.py WORKDIR ATLAS_POSTERIOR_DIR
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W, POST = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")))
from dev_comp_methods import H
DATA = os.path.abspath(os.path.join(HERE, ".."))
SETS = [("breast", "DEV_TISSUE_BREAST_01/sealed_breast_tissue.json", "tcga_brca/betas", "tcga_brca_t/betas", "normal_sample", "tumour_sample",
         "breast_luminal_epithelium", "breast_basal_epithelium"),
        ("colorectal", "DEV_TISSUE_COLON_01/sealed_colon_tissue.json", "colon/betas", "colon_t/betas", "normal_gsm", "tumour_gsm", "colon_epithelium", None),
        ("liver", "DEV_TISSUE_LIVER_01/sealed_liver_tissue.json", "lihc/betas", "lihc_t/betas", "normal_sample", "tumour_sample", "hepatocyte", None),
        ("lung", "DEV_TISSUE_LUNG_01/sealed_lung_01.json", "lung/betas", "lung_t/betas", "normal_gsm", "tumour_gsm",
         "lung_alveolar_epithelium", "lung_bronchus_epithelium")]
SURF = 0.9
def h(x): return H(np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6))
def ref(name): return pd.read_parquet(os.path.join(POST, f"atlas_{name}_posterior.parquet")).set_index("cpg")["mean"]
def beta(path):
    b = pd.read_parquet(path); return b["beta"] if "beta" in b.columns else b.iloc[:, 0]
def place(b, mu):
    ok = b.notna() & mu.notna(); hr = float(h(mu[ok]).mean()); hs = h(b[ok])
    A = float(hs.mean() / hr); Af = 1.0 / hr
    return dict(A=A, A_full=Af, from_middle=A - 1, to_surface=Af - A, f=(A - 1) / (Af - 1), at_surface=float(np.mean(hs >= SURF)), sites=int(ok.sum()))
rows = []
for cancer, seal, nd, td, nk, tk, r1, r2 in SETS:
    S = json.load(open(os.path.join(DATA, seal))); sites = S["sites_methylated"]
    mu1 = ref(r1).reindex(sites); mu2 = ref(r2).reindex(sites) if r2 else None
    ref_surface = float(np.mean(h(mu1.dropna()) >= SURF))
    for p in S["pairs"]:
        pid = p.get("case") or p.get("pid")
        for kind, folder, key in (("normal", nd, nk), ("tumor", td, tk)):
            b = beta(os.path.join(W, folder, p[key] + ".parquet")).reindex(sites)
            r = dict(cancer=cancer, pair=pid, kind=kind, reference=r1, ref_at_surface=ref_surface, **place(b, mu1))
            if mu2 is not None: r.update({f"alt_{k}": v for k, v in place(b, mu2).items() if k in ("A", "f")}, alt_reference=r2)
            rows.append(r)
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "ceiling_gauge_04_rows.csv"), index=False)
rg = np.random.default_rng(20261013)
def boot(x): return np.percentile([np.median(rg.choice(x, len(x))) for _ in range(2000)], [2.5, 97.5])
L = ["DEV-CEILING-02 step 3: every sample on the cell gauge (middle = the healthy reference of the cell of origin, A = 1)", "",
     f"{'cancer':<11}{'kind':<7}{'n':>4}  {'A median':>9}  {'A_full':>7}  {'f median [95 % CI]':<26}{'sites at surface':>17}"]
tum = {}; surf = {}
for cancer, *_ in SETS:
    for kind in ("normal", "tumor"):
        g = T[(T.cancer == cancer) & (T.kind == kind)]; ci = boot(g.f.values)
        L.append(f"{cancer:<11}{kind:<7}{len(g):>4}  {g.A.median():>9.4f}  {g.A_full.median():>7.3f}  {g.f.median():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                 f"   {g.at_surface.median():>7.2%} (ref {g.ref_at_surface.iloc[0]:.2%})")
        if kind == "tumor": tum[cancer] = (g.f.median(), ci); surf[cancer] = (g.at_surface.median(), boot(g.at_surface.values))
        if kind == "normal":
            L.append(f"{'':<18}adjacent normal {'EXCLUDES' if ci[0] > 0 or ci[1] < 0 else 'includes'} f = 0 "
                     f"({'it already sits ' + ('right' if ci[0] > 0 else 'left') + ' of the middle' if ci[0] > 0 or ci[1] < 0 else 'at the middle'})")
    alt = T[(T.cancer == cancer) & T.alt_reference.notna()] if "alt_reference" in T else pd.DataFrame()
    if len(alt):
        for kind in ("normal", "tumor"):
            a = alt[alt.kind == kind]; L.append(f"{'':<11}{kind:<7}     against {a.alt_reference.iloc[0]}: A {a.alt_A.median():.4f}, f {a.alt_f.median():+.4f}")
def rule(D, name):
    ms = np.array([v[0] for v in D.values()]); ov = max(v[1][0] for v in D.values()) <= min(v[1][1] for v in D.values())
    cv = float(np.std(ms, ddof=1) / abs(np.mean(ms)))
    return f"{name}: intervals overlap {'yes' if ov else 'no'} | spread {cv:.3f} (rule: under 0.25) -> " + \
           ("SHARED across the four cancers at this depth" if ov and cv < 0.25 else "NOT shared across the four cancers at this depth")
L += ["", rule(tum, "tumor position f"), rule(surf, "tumor share of sites at the surface"), "",
      "Development use of spent sets; bulk tumors are diluted by normal cells, so tumor positions are lower bounds."]
open(os.path.join(HERE, "ceiling_gauge_04_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
