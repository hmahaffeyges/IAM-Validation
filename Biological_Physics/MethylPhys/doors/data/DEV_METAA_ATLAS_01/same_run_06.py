"""DEV-METAA-ATLAS-01, exploration 6 (written and committed 2026-10-21 before it runs; not sealed). The instrument's job: a healthy
monocyte, read against the atlas's monocytes and tared against healthy monocytes run on the same slide, reads Normal; a planted 1 % loss
of its pattern stands out. This is the chain's own same-run tare (Stage T: references of the same cell, run the same way).
GSE168779: purified monocytes of 130 healthy children (14 to 24 months), one array each, 67 not vaccinated and 63 given BCG at birth (an
independent laboratory, not among the atlas's sources, never read; CANON/data_spent.csv). The atlas's monocytes are adults; that gap, with the
laboratory's, is what the tare must cancel. Every array through chain Stage 1 (calibrate_idat_to_beta) and chain/stage_metaa_atlas.read,
unchanged; the plant as in exploration 3. Scored on the 67 unvaccinated arrays; four tares, none using the array itself:
  slide, controls   A / median A of the other unvaccinated arrays on the same slide (at least 3, else not tared)
  slide, all        A / median A of every other array on the same slide (both groups; at least 3)
  runs, controls    A / median A of the unvaccinated arrays on the other slides (as step 3)
  runs, all         A / median A of every array on the other slides
Reported per tare: n tared, the share in Normal (0.95-1.05), the spread of A_rel, and what the planted 1 % loss moves A_rel by in
spreads; and the vaccinated arrays' median A_rel against the controls (slide, controls). Each array's values at the atlas monocytes'
identity sites are kept (same_run_06_betas.npz). Usage: python3 same_run_06.py OUTDIR"""
import os, sys, re, time, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H_, "..", "..", "..", "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA
from stage_1_idat_calibration import calibrate_idat_to_beta
out = sys.argv[1]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "work"); os.makedirs(W, exist_ok=True)
S = pd.read_csv(os.path.join(H_, "gse168779_monocyte_samples.csv"), dtype={"slide": str}); CELL = "monocytes"; SITES = SA.identity_sites(CELL)
mu = SA._col("mu", CELL).reindex(SITES); keep = {}; rows = []
def files(g):
    u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"
    for t in range(6):
        try:
            names = sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))); got = {}
            for n in names:
                p = os.path.join(W, n); urllib.request.urlretrieve(u + n, p); got["Grn" if "_Grn" in n else "Red"] = p
            if len(got) == 2: return got
        except Exception: pass
        time.sleep(20 * (t + 1))
    raise RuntimeError(f"{g}: IDATs not fetched")
for k, r in S.iterrows():
    try:
        f = files(r.gsm); o = calibrate_idat_to_beta(f["Grn"], f["Red"], verbose=False); b = pd.Series(o[0] if isinstance(o, tuple) else o, dtype="float64")
        keep[r.gsm] = b.reindex(SITES).to_numpy(dtype="float32"); rec = SA.read(b, CELL)
        bp = b.copy(); x = b.reindex(SITES); ok = x.notna() & mu.notna(); bp.loc[x[ok].index] = (x[ok] + 0.01 * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6)
        rows.append(dict(r, A=rec.get("A"), A_planted=SA.read(bp, CELL).get("A"), n_sites=rec.get("n_sites"), reason=rec.get("reason")))
    except Exception as e:
        rows.append(dict(r, reason=f"{type(e).__name__}: {e}"[:200]))
    for q in os.listdir(W): os.remove(os.path.join(W, q))
    print(k + 1, "of", len(S), r.gsm, r.group, rows[-1].get("A"), flush=True)
    if k == 2 and not any(x.get("A") for x in rows): sys.exit("the first 3 arrays all failed: " + "; ".join(str(x.get("reason")) for x in rows))
np.savez_compressed(os.path.join(out, "same_run_06_betas.npz"), sites=np.array(SITES), gsm=np.array(list(keep)), beta=np.array(list(keep.values())))
R = pd.DataFrame(rows); R["control"] = R.group.eq("Non vaccinated control"); ok = R.A.notna()
REFS = {"slide_controls": lambda r: ok & R.control & (R.slide == r.slide) & (R.gsm != r.gsm), "slide_all": lambda r: ok & (R.slide == r.slide) & (R.gsm != r.gsm),
        "runs_controls": lambda r: ok & R.control & (R.slide != r.slide), "runs_all": lambda r: ok & (R.slide != r.slide)}
for i, r in R[ok].iterrows():
    for t, sel in REFS.items():
        t_ = SA.tare(r.A, list(R[sel(r)].A)); R.loc[i, f"A_rel_{t}"] = t_["A_rel"]
        if t_["A_rel"] is not None: R.loc[i, f"A_rel_{t}_planted"] = SA.tare(r.A_planted, list(R[sel(r)].A))["A_rel"]
R.to_csv(os.path.join(out, "same_run_06_rows.csv"), index=False)
C = R[R.control]; L = ["DEV-METAA-ATLAS-01 exploration 6: GSE168779 healthy monocytes, the chain's same-run tare against the atlas (not sealed)",
     f"arrays read {int(ok.sum())} of {len(R)} (unvaccinated {int((ok & R.control).sum())} of {int(R.control.sum())})"]
for t in REFS:
    a = C[f"A_rel_{t}"].dropna()
    if len(a) < 3: L.append(f"tare {t}: fewer than 3 tared"); continue
    sp = float(np.std(a / a.median(), ddof=1)); ch = float((C[f"A_rel_{t}_planted"] - C[f"A_rel_{t}"]).mean())
    L.append(f"tare {t:<15}: n {len(a)} | Normal {a.between(0.95, 1.05).mean():.2f} | median {a.median():.4f} | spread {sp:.4f} | planted 1 % moves {ch:+.4f} = {ch / sp:.1f} spreads")
v = R[~R.control].A_rel_slide_controls.dropna(); L.append(f"vaccinated, tared against the controls on their slide: n {len(v)} | median A_rel {v.median():.4f} | Normal {v.between(0.95, 1.05).mean():.2f}" if len(v) else "vaccinated: none tared")
open(os.path.join(out, "same_run_06_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
