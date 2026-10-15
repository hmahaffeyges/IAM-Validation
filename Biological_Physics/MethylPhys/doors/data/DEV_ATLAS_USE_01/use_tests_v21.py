"""DEV-ATLAS-USE-01 (ATL-060): atlas v2.1 scored on the jobs Loyfer and Moss use a reference atlas for. Written 2026-10-15 BEFORE any atlas
v2.1 value or held-out array is opened. Showcase bars (author: "not meant for physics referees"): each asks whether v2.1 does the job at least
as well as Moss's published reference (moss_reference_atlas.csv, nloyfer/meth_atlas, 25 cell types) on the same held-out data. A miss is
reported and never blocks the release.
UT1 MARKERS. For each cell type C in both references with held-out purified arrays from an independent laboratory, take the top K = 100
  marker sites of C from each reference: score(site) = value(C) - max over other shared cells (hypermethylated markers) or min over others -
  value(C) (hypomethylated); the K highest scores, sites on the held-out platform only. A marker is CONFIRMED if, in the held-out arrays,
  the mean of C differs from the mean of EVERY other shared cell by >= 0.20 in the marker's direction (0.20: the Delta-beta most array
  marker studies call a clear difference). BAR: v2.1's confirmed share >= Moss's minus 0.05 (0.05 = 5 of 100 markers, about one binomial
  standard error at K = 100), per cell; reported per cell and pooled.
UT2 MIXTURES (deconvolution, the blood test). 200 in-silico mixtures, each from distinct held-out arrays of one platform, proportions drawn
  from a flat Dirichlet, seed 2026; deconvolution by non-negative least squares normalized to 1 (Moss's method), on each reference's own
  K-marker union. Error = mean absolute error of the proportions per mixture.
  A. Same ground: only cells in both references. BAR: v2.1's median error <= 1.25 x Moss's on the same mixtures (a ratio holds at any error
     scale; equal references read 1.0). Synthetic run 1 used an absolute bar (Moss + 0.01), which passed a planted v2.1 four times noisier
     than Moss because mixing averages the noise down; it was replaced before any real value was read.
  B. Wide area: the same plus cells only v2.1 has, 10-30 percent of each mixture. Prediction: Moss, lacking those cells, assigns their share to
     its nearest cells; v2.1 recovers them. BAR: v2.1 recovers the outside share within 0.05 (U3's per-sample bar) in >= 90 percent of
     mixtures, and its median error over all components <= 0.03.
UT3 EXTRAS, shown, not scored: one blood cell's site intervals, its 100 draws and the array-to-sequencing bridge, as a worked example.
Usage: python use_tests_v21.py synthetic | real <v21_means.csv> <heldout_betas.parquet> <heldout_samples.csv>"""
# NO CHAIN: scores the atlas against a published reference on held-out arrays; no chain stage is involved
import os, sys, json, numpy as np, pandas as pd
from scipy.optimize import nnls
H = os.path.dirname(os.path.abspath(__file__)); K = 100; GAP = 0.20
def markers(ref, cells, k=K):
    R = ref[cells]; out = {}
    for c in cells:
        o = R.drop(columns=c); hyper = R[c] - o.max(axis=1); hypo = o.min(axis=1) - R[c]
        s = pd.concat([hyper.rename("s").to_frame().assign(d=1), hypo.rename("s").to_frame().assign(d=-1)]).sort_values("s", ascending=False)
        s = s[~s.index.duplicated()].head(k); out[c] = s.d
    return out
def confirmed(mk, means):
    out = {}
    for c, d in mk.items():
        x = means.reindex(d.index).dropna()
        if x.empty or c not in x: out[c] = np.nan; continue
        o = x.drop(columns=c); dd = d.reindex(x.index)
        ok = np.where(dd > 0, (x[c].values[:, None] - o.values >= GAP).all(1), (o.values - x[c].values[:, None] >= GAP).all(1))
        out[c] = float(ok.mean())
    return out
def deconv(ref, mix):
    w, _ = nnls(np.asarray(ref, float), np.asarray(mix, float)); return w / w.sum() if w.sum() > 0 else w
def mixtures(samples, cells, n, rng, outside=None):
    M = []
    for _ in range(n):
        p = rng.dirichlet(np.ones(len(cells)))
        if outside:
            f = rng.uniform(0.10, 0.30); q = rng.dirichlet(np.ones(len(outside))) * f; p = p * (1 - f); allc = cells + outside; pr = np.concatenate([p, q])
        else: allc, pr = cells, p
        x = sum(w * samples[c][rng.integers(len(samples[c]))] for c, w in zip(allc, pr)); M.append((dict(zip(allc, pr)), x))
    return M
def score_mix(M, ref, cells_ref):
    e = []; outs = []
    for truth, x in M:
        w = dict(zip(cells_ref, deconv(ref[cells_ref], x)))
        keys = set(truth) | set(w); e.append(np.mean([abs(truth.get(k, 0) - w.get(k, 0)) for k in keys]))
        outs.append((sum(v for k, v in truth.items() if k not in SHARED), sum(v for k, v in w.items() if k not in SHARED)))
    return np.array(e), outs
if sys.argv[1] == "synthetic":
    rng = np.random.default_rng(15); n_sites = 4000
    SHARED = [f"shared{i}" for i in range(7)]; ONLY = [f"only{i}" for i in range(3)]; ALL = SHARED + ONLY
    truth = pd.DataFrame(rng.beta(0.6, 0.6, (n_sites, len(ALL))), columns=ALL)
    v21 = (truth + rng.normal(0, 0.03, truth.shape)).clip(0, 1); moss = (truth[SHARED] + rng.normal(0, 0.03, (n_sites, 7))).clip(0, 1)
    samples = {c: [(truth[c] + rng.normal(0, 0.05, n_sites)).clip(0, 1).values for _ in range(6)] for c in ALL}
    heldmeans = pd.DataFrame({c: np.mean(samples[c], 0) for c in SHARED})
    L = []
    for lab, noise in (("equal references", 0.03), ("v2.1 noisier (should lose UT1 and UT2A)", 0.12)):
        v = (truth + rng.normal(0, noise, truth.shape)).clip(0, 1)
        cv = confirmed(markers(v, SHARED), heldmeans); cm = confirmed(markers(moss, SHARED), heldmeans)
        ut1 = all(cv[c] >= cm[c] - 0.05 for c in SHARED)
        mkv = markers(v, SHARED); mkm = markers(moss, SHARED)
        siv = sorted(set().union(*[set(m.index) for m in mkv.values()])); sim = sorted(set().union(*[set(m.index) for m in mkm.values()]))
        MA = mixtures({c: [s for s in samples[c]] for c in ALL}, SHARED, 200, rng)
        ev, _ = score_mix([(t, x[siv]) for t, x in MA], v.loc[siv], SHARED); em, _ = score_mix([(t, x[sim]) for t, x in MA], moss.loc[sim], SHARED)
        ut2a = np.median(ev) <= 1.25 * np.median(em)
        L.append(f"{lab}: UT1 confirmed v2.1 {np.mean(list(cv.values())):.2f} Moss {np.mean(list(cm.values())):.2f} -> {'MET' if ut1 else 'NOT MET'}; "
                 f"UT2A median error v2.1 {np.median(ev):.3f} Moss {np.median(em):.3f} -> {'MET' if ut2a else 'NOT MET'}")
    mkv = markers(v21, ALL); siv = sorted(set().union(*[set(m.index) for m in mkv.values()]))
    MB = mixtures(samples, SHARED, 200, rng, outside=ONLY)
    ev, ov = score_mix([(t, x[siv]) for t, x in MB], v21.loc[siv], ALL); em, om = score_mix([(t, x[sim]) for t, x in MB], moss.loc[sim], SHARED)
    rec = np.mean([abs(a - b) <= 0.05 for a, b in ov]); ut2b = rec >= 0.90 and np.median(ev) <= 0.03
    L.append(f"wide area: v2.1 recovers the outside share within 0.05 in {rec:.2f} of mixtures, median error {np.median(ev):.3f}; Moss median error "
             f"{np.median(em):.3f} with the outside share forced onto its own cells -> UT2B {'MET' if ut2b else 'NOT MET'}")
    bad = v21.copy(); bad[ONLY] = rng.beta(0.6, 0.6, (n_sites, len(ONLY)))       # v2.1 with wrong profiles for the outside cells
    mkb = markers(bad, ALL); sib = sorted(set().union(*[set(m.index) for m in mkb.values()]))
    eb, ob = score_mix([(t, x[sib]) for t, x in MB], bad.loc[sib], ALL); recb = np.mean([abs(a - b) <= 0.05 for a, b in ob]); ut2b_bad = recb >= 0.90 and np.median(eb) <= 0.03
    L.append(f"wide area, v2.1 with wrong outside profiles (should fail): recovers within 0.05 in {recb:.2f}, median error {np.median(eb):.3f} -> UT2B {'MET' if ut2b_bad else 'NOT MET'}")
    ok = L[0].count("MET") == 2 and "NOT MET" not in L[0] and L[1].count("NOT MET") == 2 and ut2b and not ut2b_bad
    L.append(f"SYNTHETIC: {'MET' if ok else 'NOT MET'} (equal references pass, a noisier v2.1 fails UT1 and UT2A, the wide-area arm recovers the outside cells and fails with wrong outside profiles)")
    open(os.path.join(H, "use_tests_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
elif sys.argv[1] == "real":
    raise SystemExit("REFUSED: the real run waits for the seal (DEV_ATLAS_USE_01.md) and the v2.1 release")
