"""DEV-METAA-ATLAS-01, exploration 5 (written and committed 2026-10-21 before it runs; not sealed: "we explore without sealing until we are
certain"). The instrument's job: a healthy sorted cell, read against its cell in IAM-Atlas v2.1 and tared against healthy cells run the
same way, reads Normal; a planted 1 % loss of its pattern stands out.
Step 3 (GSE184269) could tare each array only against other slides (one donor per slide), and its spread moved with the slide. Here,
GSE166844 (purified blood cells of 30 volunteers from a general-population cohort, an independent laboratory, not among the atlas's 21
sources, never read before; CANON/data_spent.csv): 141 sorted arrays, B, CD4 T, CD8 T, monocytes and granulocytes (read against the atlas's
neutrophils), and each slide holds two donors' cells. Every array through chain Stage 1 (calibrate_idat_to_beta) and
chain/stage_metaa_atlas.read, unchanged; the plant as in exploration 3 (every identity site moved 1 % of the way to 0.5). Two tares,
neither using the array's own donor:
  across runs   A_rel = A / median A of the same cell type on the other slides (as step 3)
  same slide    the same, divided by the slide's offset, taken only from the other donor on that slide: the median, over that donor's
                cell types, of its A over the median A of its cell type on the other slides. A loss shared by all of one donor's
                cells is not hidden, because that donor never enters its own tare.
Reported per cell type and tare: n, the share in Normal (0.95-1.05), the spread of A_rel, and what the planted 1 % loss moves A_rel by in
spreads. Also saved: every array's values at the five cells' identity sites (same_slide_05_betas.npz), so a later exploration needs no
re-download. Usage: python3 same_slide_05.py OUTDIR"""
import os, sys, re, time, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H_, "..", "..", "..", "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA
from stage_1_idat_calibration import calibrate_idat_to_beta
out = sys.argv[1]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "work"); os.makedirs(W, exist_ok=True)
S = pd.read_csv(os.path.join(H_, "gse166844_sorted_samples.csv")); CELLS = sorted(S.atlas_cell.unique())
SITES = sorted(set(s for c in CELLS for s in SA.identity_sites(c))); keep = {}; rows = []
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
        keep[r.gsm] = b.reindex(SITES).to_numpy(dtype="float32")
        rec = SA.read(b, r.atlas_cell); S_ = SA.identity_sites(r.atlas_cell); mu = SA._col("mu", r.atlas_cell).reindex(S_)
        bp = b.copy(); x = b.reindex(S_); ok = x.notna() & mu.notna(); bp.loc[x[ok].index] = (x[ok] + 0.01 * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6)
        rows.append(dict(r, A=rec.get("A"), A_planted=SA.read(bp, r.atlas_cell).get("A"), n_sites=rec.get("n_sites"), reason=rec.get("reason")))
    except Exception as e:
        rows.append(dict(r, reason=f"{type(e).__name__}: {e}"[:200]))
    for q in os.listdir(W): os.remove(os.path.join(W, q))
    print(k + 1, "of", len(S), r.gsm, r.atlas_cell, rows[-1].get("A"), flush=True)
    if k == 2 and not any(x.get("A") for x in rows): sys.exit("the first 3 arrays all failed: " + "; ".join(str(x.get("reason")) for x in rows))
np.savez_compressed(os.path.join(out, "same_slide_05_betas.npz"), sites=np.array(SITES), gsm=np.array(list(keep)), beta=np.array(list(keep.values())))
R = pd.DataFrame(rows)
def med_other_slides(cell, slide): return R[(R.atlas_cell == cell) & (R.slide != slide) & R.A.notna()].A.median()
for i, r in R.iterrows():
    if pd.isna(r.A): continue
    m = med_other_slides(r.atlas_cell, r.slide); R.loc[i, "A_rel_runs"] = r.A / m; R.loc[i, "A_rel_runs_planted"] = r.A_planted / m
    o = R[(R.slide == r.slide) & (R.donor != r.donor) & R.A.notna()]
    off = np.median([q.A / med_other_slides(q.atlas_cell, q.slide) for _, q in o.iterrows()]) if len(o) else np.nan
    R.loc[i, "slide_offset_other_donor"] = off; R.loc[i, "n_offset_arrays"] = len(o)
    R.loc[i, "A_rel_slide"] = r.A / (m * off); R.loc[i, "A_rel_slide_planted"] = r.A_planted / (m * off)
R.to_csv(os.path.join(out, "same_slide_05_rows.csv"), index=False)
L = [f"DEV-METAA-ATLAS-01 exploration 5: GSE166844 sorted cells, tared across runs and against the other donor on the same slide (not sealed)",
     f"arrays read {int(R.A.notna().sum())} of {len(R)}"]
for c, g in R.groupby("atlas_cell"):
    for t in ("runs", "slide"):
        a = g[f"A_rel_{t}"].dropna(); p = g[f"A_rel_{t}_planted"].dropna()
        if len(a) < 3: L.append(f"{c} | {t}: fewer than 3 readings"); continue
        sp = float(np.std(a / a.median(), ddof=1)); ch = float((g[f"A_rel_{t}_planted"] - g[f"A_rel_{t}"]).mean())
        L.append(f"{c} | tare {t:<5}: n {len(a)} | Normal {a.between(0.95, 1.05).mean():.2f} | spread {sp:.4f} | planted 1 % moves {ch:+.4f} = {ch / sp:.1f} spreads")
open(os.path.join(out, "same_slide_05_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
