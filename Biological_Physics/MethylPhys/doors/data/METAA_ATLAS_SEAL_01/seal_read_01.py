"""METAA-ATLAS-SEAL-01: Met-A against IAM-Atlas v2.1 on purified monocytes, read once (written 2026-10-22; doors/METAA_ATLAS_SEAL_01.md).
The instrument's job: a healthy purified monocyte, read against the atlas's monocytes and tared against other healthy monocytes of the same
laboratory, reads Normal; a 1 % loss of its pattern stands out.
Every array: its two IDATs from GEO, chain Stage 1 (calibrate_idat_to_beta) and chain/stage_metaa_atlas.read (cell "monocytes"), unchanged.
The plant, on each array's own values: every identity site moved 1 % of the way to 0.5 (exploration 3's rule), read the same way.
The tare (never the array itself): A_rel = A / median A of every OTHER healthy array in the list (stage_metaa_atlas.tare, at least 3).
Reported, not scored: the same tare taken only from other healthy arrays on the same slide, where a slide holds at least 3 others.
Bars (sealed in sealed_01.json; nothing here is tuned to the result):
  B1  at least 90 % of the listed arrays receive a tared reading
  B2  at least 90 % of the tared healthy readings are Normal (0.95 to 1.05)
  B3  the planted 1 % loss moves the tared reading, on average, by at least 2 healthy spreads (spread: SD of the healthy A_rel)
Usage: python3 seal_read_01.py SAMPLES_CSV OUTDIR TAG   (TAG: rehearsal or sealed; writes TAG_rows.csv, TAG_summary.txt, TAG.json)"""
import os, sys, re, time, json, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H_, "..", "..", "..", "chain"))
import stage_metaa_atlas as SA
from stage_1_idat_calibration import calibrate_idat_to_beta
SAMPLES, out, TAG = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "work"); os.makedirs(W, exist_ok=True)
S = pd.read_csv(SAMPLES, dtype={"slide": str}); CELL = "monocytes"; SITES = SA.identity_sites(CELL); mu = SA._col("mu", CELL).reindex(SITES)
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
        rec = SA.read(b, CELL); bp = b.copy(); x = b.reindex(SITES); ok = x.notna() & mu.notna()
        bp.loc[x[ok].index] = (x[ok] + 0.01 * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6)
        rows.append(dict(r, A=rec.get("A"), A_planted=SA.read(bp, CELL).get("A"), n_sites=rec.get("n_sites"), reason=rec.get("reason")))
    except Exception as e:
        rows.append(dict(r, A=None, reason=f"{type(e).__name__}: {e}"[:200]))
    for q in os.listdir(W): os.remove(os.path.join(W, q))
    print(k + 1, "of", len(S), r.gsm, rows[-1].get("A"), flush=True)
    if k == 2 and not any(x.get("A") for x in rows): sys.exit("the first 3 arrays all failed: " + "; ".join(str(x.get("reason")) for x in rows))
R = pd.DataFrame(rows); ok = R.A.notna()
for i, r in R[ok].iterrows():
    lab = list(R[ok & (R.gsm != r.gsm)].A); t = SA.tare(r.A, lab)
    R.loc[i, "A_rel"] = t["A_rel"]; R.loc[i, "state"] = t.get("state")
    if t["A_rel"] is not None: R.loc[i, "A_rel_planted"] = SA.tare(r.A_planted, lab)["A_rel"]
    sl = list(R[ok & (R.gsm != r.gsm) & (R.slide == r.slide)].A)
    if len(sl) >= 3: R.loc[i, "A_rel_slide"] = SA.tare(r.A, sl)["A_rel"]; R.loc[i, "A_rel_slide_planted"] = SA.tare(r.A_planted, sl)["A_rel"]
R.to_csv(os.path.join(out, f"{TAG}_rows.csv"), index=False)
a = R.A_rel.dropna(); n = len(R); sp = float(np.std(a, ddof=1)) if len(a) > 2 else float("nan"); mv = float((R.A_rel_planted - R.A_rel).mean())
b1 = len(a) >= 0.9 * n; b2 = len(a) > 0 and float(a.between(0.95, 1.05).mean()) >= 0.9; b3 = np.isfinite(sp) and sp > 0 and mv / sp >= 2
s = R.A_rel_slide.dropna(); ss = float(np.std(s, ddof=1)) if len(s) > 2 else None; ms = float((R.A_rel_slide_planted - R.A_rel_slide).mean()) if len(s) else None
J = dict(tag=TAG, samples=os.path.basename(SAMPLES), n_listed=n, n_tared=len(a), share_normal=round(float(a.between(0.95, 1.05).mean()), 4) if len(a) else None,
         median_A_rel=round(float(a.median()), 5) if len(a) else None, spread=round(sp, 5), planted_move=round(mv, 5), planted_spreads=round(mv / sp, 2) if sp else None,
         B1=bool(b1), B2=bool(b2), B3=bool(b3), all_met=bool(b1 and b2 and b3),
         slide_tare_reported=dict(n=len(s), share_normal=round(float(s.between(0.95, 1.05).mean()), 4) if len(s) else None, spread=round(ss, 5) if ss else None, planted_spreads=round(ms / ss, 2) if ss else None))
json.dump(J, open(os.path.join(out, f"{TAG}.json"), "w"), indent=1)
L = [f"METAA-ATLAS-SEAL-01 {TAG}: {os.path.basename(SAMPLES)}, Met-A against IAM-Atlas v2.1 monocytes, laboratory tare",
     f"B1 tared {len(a)} of {n} (bar: at least 90 %) -> {'MET' if b1 else 'NOT MET'}",
     f"B2 Normal {J['share_normal']} of tared, median {J['median_A_rel']} (bar: at least 0.90) -> {'MET' if b2 else 'NOT MET'}",
     f"B3 planted 1 % moves {J['planted_move']:+} = {J['planted_spreads']} healthy spreads (spread {J['spread']}; bar: at least 2) -> {'MET' if b3 else 'NOT MET'}",
     f"reported, not scored: same-slide tare {J['slide_tare_reported']}",
     f"ALL BARS {'MET' if J['all_met'] else 'NOT MET'}"]
open(os.path.join(out, f"{TAG}_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
