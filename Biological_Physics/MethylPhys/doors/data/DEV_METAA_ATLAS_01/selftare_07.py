"""DEV-METAA-ATLAS-01, exploration 7 (written 2026-10-22 before it runs; exploratory, not sealed; both series already read, so spent).
Why: the chain's commissioned Met-A reads an array only after Stage T step 1, the self-tare (conductor_v3.stage_t_selftare_ii, adopted
2026-10-04: each design's values mapped onto the reference arrays' scale by the array's OWN fixed-site anchors; no other person). Exploration 6
and the sealed test METAA-ATLAS-SEAL-01 read Met-A against the atlas WITHOUT that step. The sealed test failed one bar on the spread across
the laboratory's slides (0.041). Question: does the chain's own step 1, done first, shrink that spread?
Every array: IDATs from GEO, chain Stage 1, then Met-A against IAM-Atlas v2.1 monocytes twice: without step 1 (as before) and after step 1
(dev_stages.selftare_map, unchanged). The plant (exploration 3's rule) on each array's own values before step 1. Tares never use the array
itself: laboratory (every other healthy array of the series) and same slide (where a slide holds at least 3 others).
Data: GSE168779's 67 unvaccinated healthy children (exploration 6) and GSE267660's 31 healthy donors (the sealed test).
Usage: python3 selftare_07.py OUTDIR   (writes selftare_07_rows.csv and selftare_07_summary.txt)"""
import os, sys, re, time, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H_, "..", "..", "..", "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, dev_stages as DV
from stage_1_idat_calibration import calibrate_idat_to_beta
out = sys.argv[1]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "work"); os.makedirs(W, exist_ok=True)
CELL = "monocytes"; SITES = SA.identity_sites(CELL); mu = SA._col("mu", CELL).reindex(SITES)
A6 = pd.read_csv(os.path.join(H_, "gse168779_monocyte_samples.csv"), dtype={"slide": str}); A6 = A6[A6.group == "Non vaccinated control"].assign(series="GSE168779")
SL = pd.read_csv(os.path.join(H_, "..", "METAA_ATLAS_SEAL_01", "seal_samples.csv"), dtype={"slide": str}).assign(series="GSE267660")
S = pd.concat([A6[["gsm", "slide", "series"]], SL[["gsm", "slide", "series"]]], ignore_index=True)
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
rows = []
for k, r in S.iterrows():
    try:
        f = files(r.gsm); o = calibrate_idat_to_beta(f["Grn"], f["Red"], verbose=False); b = pd.Series(o[0] if isinstance(o, tuple) else o, dtype="float64")
        bp = b.copy(); x = b.reindex(SITES); ok = x.notna() & mu.notna(); bp.loc[x[ok].index] = (x[ok] + 0.01 * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6)
        bs, info = DV.selftare_map(b); bsp, _ = DV.selftare_map(bp)
        rows.append(dict(r, A=SA.read(b, CELL).get("A"), A_planted=SA.read(bp, CELL).get("A"),
                         A_st=SA.read(bs, CELL).get("A") if bs is not None else None, A_st_planted=SA.read(bsp, CELL).get("A") if bsp is not None else None,
                         slope_I=(info.get("maps", {}).get("I") or {}).get("slope"), slope_II=(info.get("maps", {}).get("II") or {}).get("slope")))
    except Exception as e: rows.append(dict(r, reason=f"{type(e).__name__}: {e}"[:200]))
    for q in os.listdir(W): os.remove(os.path.join(W, q))
    print(k + 1, "of", len(S), r.gsm, rows[-1].get("A"), rows[-1].get("A_st"), flush=True)
    if k == 2 and not any(x.get("A_st") for x in rows): sys.exit("the first 3 arrays all failed: " + "; ".join(str(x.get("reason")) for x in rows))
R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "selftare_07_rows.csv"), index=False)
L = ["DEV-METAA-ATLAS-01 exploration 7 (exploratory, not sealed): Met-A against IAM-Atlas v2.1 monocytes, without and with the chain's Stage T step 1"]
for ser, g in R.groupby("series"):
    for tag, a_col, p_col in (("without step 1", "A", "A_planted"), ("with step 1", "A_st", "A_st_planted")):
        ok = g[a_col].notna(); G = g[ok]
        for scope in ("laboratory", "same slide"):
            rel, relp = [], []
            for i, r in G.iterrows():
                ref = G[(G.gsm != r.gsm) & ((G.slide == r.slide) if scope == "same slide" else True)][a_col]
                if len(ref) >= 3: rel.append(r[a_col] / ref.median()); relp.append(r[p_col] / ref.median())
            rel, relp = np.array(rel), np.array(relp)
            if len(rel) < 3: L.append(f"{ser} | {tag:<14} | {scope:<10}: fewer than 3 tared"); continue
            sp = float(np.std(rel, ddof=1)); mv = float(np.mean(relp - rel))
            L.append(f"{ser} | {tag:<14} | {scope:<10}: n {len(rel)} | Normal {np.mean((rel >= 0.95) & (rel <= 1.05)):.3f} | spread {sp:.4f} | planted 1 % moves {mv:+.4f} = {mv / sp:.2f} spreads")
open(os.path.join(out, "selftare_07_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
