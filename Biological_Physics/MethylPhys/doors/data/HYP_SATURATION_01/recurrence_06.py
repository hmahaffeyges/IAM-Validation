"""HYP-SATURATION-01, falsifier 6: are the tumors' gains and losses NON-RANDOM? (written 2026-10-13, BEFORE it was run)

The claim (author): "it is intentional in that its not random erasure of entire blocks of genes or random tagging of entire blocks."
Restated: the sites that GAIN and the sites that LOSE methylation recur across patients far more than chance, and gains concentrate
at gene promoters. If they scatter no more than a shuffle gives, the moves are drift and the patterned-program picture is wrong.

# NO CHAIN: a per-site paired difference (tumor beta minus the patient's own normal beta) and counting. No gauge reading is involved;
# the calibrated betas are the chain's Stage 1 output for the four sealed tissue sets.

Data (all four already read and spent; development use, declared in the note): the 219 tumor/normal pairs of the sealed breast,
colorectal, liver and lung tissue tests, the same pairs and folders as DEV-CEILING-02 step 3 (ceiling_gauge_04.py).

Fixed before running:
  sites   autosomal probes present in both members of a pair; annotation from the Illumina 450K manifest (EPIC-only probes are
          dropped, declared); promoter = UCSC_RefGene_Group containing TSS200 or TSS1500
  call    gain if tumor - normal >= +0.20, loss if <= -0.20 (one fixed threshold, both directions)
  recur   a site is recurrent if called in >= 50 % of that cancer's patients
  null    20 permutations per cancer: each tumor's calls moved to random sites, keeping that tumor's own number of gains and losses
BARS
  B1 non-random gains   in EVERY cancer, recurrent-gain sites > the maximum over the 20 permutations
  B2 non-random losses  in EVERY cancer, recurrent-loss sites > the maximum over the 20 permutations
  B3 gains at promoters in EVERY cancer, the promoter share of recurrent gains >= 1.5 x the promoter share of all tested sites
The falsifier BREAKS if B1 or B2 fails in any cancer. B3 failing breaks only "gains at promoters".
PLANTED CHECK, run first on the real breast site list, real refuses if it fails:
  P1 random calls only (2 % gains and 2 % losses per synthetic patient, independent)  -> recurrent counts must not exceed the null max
  P2 the same plus 500 planted sites gained in 60 % of patients                     -> at least 400 recurrent gain sites detected
Honest note on strength: the literature already reports recurrent promoter gains, so this is expected to pass; it is a check that
our data agree, not a hard test of the picture. Usage: python3 recurrence_06.py WORKDIR
"""
import gzip, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W = sys.argv[1]; DATA = os.path.abspath(os.path.join(HERE, ".."))
THR, RECUR, NPERM = 0.20, 0.50, 20
SETS = [("breast", "DEV_TISSUE_BREAST_01/sealed_breast_tissue.json", "tcga_brca/betas", "tcga_brca_t/betas", "normal_sample", "tumour_sample"),
        ("colorectal", "DEV_TISSUE_COLON_01/sealed_colon_tissue.json", "colon/betas", "colon_t/betas", "normal_gsm", "tumour_gsm"),
        ("liver", "DEV_TISSUE_LIVER_01/sealed_liver_tissue.json", "lihc/betas", "lihc_t/betas", "normal_sample", "tumour_sample"),
        ("lung", "DEV_TISSUE_LUNG_01/sealed_lung_01.json", "lung/betas", "lung_t/betas", "normal_gsm", "tumour_gsm")]
MAN = os.path.join(W, "mphome", ".methylprep_manifest_files", "HumanMethylation450k_15017482_v3.csv.gz")
_hdr = next(i for i, l in enumerate(gzip.open(MAN, "rt")) if l.startswith("IlmnID"))   # this copy may or may not carry the 7-line preamble
M = pd.read_csv(MAN, skiprows=_hdr, usecols=["IlmnID", "CHR", "UCSC_RefGene_Group"], low_memory=False, dtype=str).dropna(subset=["CHR"])
M = M[M.IlmnID.str.startswith("cg") & ~M.CHR.isin(["X", "Y"])].set_index("IlmnID")
PROM = M.UCSC_RefGene_Group.fillna("").str.contains("TSS200|TSS1500")
rg = np.random.default_rng(20261013)
def beta(p):
    b = pd.read_parquet(p); return (b["beta"] if "beta" in b.columns else b.iloc[:, 0]).astype("float32")
def recurrent(calls):              # calls: patients x sites, bool
    return int((calls.mean(0) >= RECUR).sum())
def null_max(calls):
    n, m = calls.shape; k = calls.sum(1); out = []
    for _ in range(NPERM):
        z = np.zeros((n, m), bool)
        for i in range(n): z[i, rg.choice(m, k[i], replace=False)] = True
        out.append(recurrent(z))
    return max(out), float(np.mean(out))
def score(G, Ls, prom):
    rgain, rloss = recurrent(G), recurrent(Ls); ng, ng_mean = null_max(G); nl, nl_mean = null_max(Ls)
    rec = G.mean(0) >= RECUR; base = float(prom.mean()); share = float(prom[rec].mean()) if rec.any() else float("nan")
    return dict(recurrent_gain=rgain, null_gain_max=ng, null_gain_mean=ng_mean, recurrent_loss=rloss, null_loss_max=nl, null_loss_mean=nl_mean,
                promoter_share_recurrent_gain=share, promoter_share_all=base, promoter_ratio=share / base if base else float("nan"),
                B1=rgain > ng, B2=rloss > nl, B3=bool(share >= 1.5 * base) if rec.any() else False)
# ---------------- planted check, on the breast site list
S0 = json.load(open(os.path.join(DATA, SETS[0][1]))); p0 = S0["pairs"][0]
sites0 = beta(os.path.join(W, SETS[0][2], p0[SETS[0][4]] + ".parquet")).index.intersection(M.index)
m0, n0 = len(sites0), 40; prom0 = PROM.reindex(sites0).fillna(False).values
G1 = rg.random((n0, m0)) < 0.02; L1 = rg.random((n0, m0)) < 0.02
P1 = score(G1, L1, prom0)
G2 = G1.copy(); plant = rg.choice(m0, 500, replace=False); G2[:, plant] |= rg.random((n0, 500)) < 0.60
P2 = score(G2, L1, prom0)
ok1 = P1["recurrent_gain"] <= P1["null_gain_max"] and P1["recurrent_loss"] <= P1["null_loss_max"]; ok2 = P2["recurrent_gain"] >= 400
L = [f"PLANTED P1 random only: recurrent gains {P1['recurrent_gain']} (null max {P1['null_gain_max']}), losses {P1['recurrent_loss']} "
     f"(null max {P1['null_loss_max']}) -> {'PASS' if ok1 else 'FAIL'}",
     f"PLANTED P2 500 planted at 60 %: recurrent gains {P2['recurrent_gain']} (need >= 400) -> {'PASS' if ok2 else 'FAIL'}"]
if not (ok1 and ok2):
    open(os.path.join(HERE, "recurrence_06_summary.txt"), "w").write("\n".join(L + ["PLANTED CHECK FAILED - real data not read"]) + "\n")
    print("\n".join(L)); sys.exit(1)
# ---------------- the real pairs
rows = []
for cancer, seal, nd, td, nk, tk in SETS:
    S = json.load(open(os.path.join(DATA, seal))); pairs = S["pairs"]
    D = {}
    for p in pairs:
        n = beta(os.path.join(W, nd, p[nk] + ".parquet")); t = beta(os.path.join(W, td, p[tk] + ".parquet"))
        D[p.get("case") or p.get("pid") or p[tk]] = (t - n)
    X = pd.DataFrame(D).T; X = X.loc[:, X.columns.intersection(M.index)]; X = X.loc[:, X.notna().all(0)]
    prom = PROM.reindex(X.columns).fillna(False).values; V = X.values
    r = dict(cancer=cancer, patients=len(X), sites=X.shape[1], **score(V >= THR, V <= -THR, prom))
    r["mean_gains_per_tumor"] = float((V >= THR).sum(1).mean()); r["mean_losses_per_tumor"] = float((V <= -THR).sum(1).mean())
    rows.append(r); print(cancer, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "cancer"}, flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "recurrence_06_output.csv"), index=False)
for r in T.itertuples():
    L.append(f"{r.cancer:<11} n {r.patients:>3} | sites {r.sites:>6} | recurrent gains {r.recurrent_gain:>6} vs null max {r.null_gain_max:>3} "
             f"| recurrent losses {r.recurrent_loss:>6} vs null max {r.null_loss_max:>3} | promoter share of recurrent gains "
             f"{r.promoter_share_recurrent_gain:.1%} vs {r.promoter_share_all:.1%} all ({r.promoter_ratio:.2f}x) | B1 {r.B1} B2 {r.B2} B3 {r.B3}")
v = "NOT BROKEN" if T.B1.all() and T.B2.all() else "BROKEN"
L.append(f"FALSIFIER 6: {v} (B1 all {bool(T.B1.all())}, B2 all {bool(T.B2.all())}); gains at promoters (B3) all {bool(T.B3.all())}")
open(os.path.join(HERE, "recurrence_06_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
