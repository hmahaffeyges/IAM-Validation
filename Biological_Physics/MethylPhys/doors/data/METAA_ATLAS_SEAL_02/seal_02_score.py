"""METAA-ATLAS-SEAL-02 scorer (written 2026-10-23 before any listed array is opened; frozen by SHA-256 in sealed_02.json). It only reads the table of
bundles the one command wrote (rows.csv from chain/MethylPhys_Interface/run_series.py, series.yml, scope slide, plant metaa_atlas); it reads and
tares nothing. The instrument's job: a healthy purified cell, read against its atlas cell and tared against healthy cells of the same type on its own
slide, reads Normal; a planted 1 % loss stands out.
B1: at least 90 % of the listed arrays receive a same-slide tared reading.
B2: at least 90 % of the tared healthy readings are Normal (0.95 to 1.05).
B3: the planted 1 % loss moves the tared reading by at least 2 healthy spreads on average (spread: SD of the tared healthy readings).
Each series is also reported on its own; only the pooled bars are scored.  Usage: python3 seal_02_score.py ROWS_CSV SAMPLES_CSV OUTDIR"""
import os, sys, numpy as np, pandas as pd
R = pd.read_csv(sys.argv[1]); S = pd.read_csv(sys.argv[2], dtype={"slide": str}); out = sys.argv[3]
if "series" not in S.columns: S["series"] = os.path.basename(sys.argv[2])   # a list without a series column (the rehearsal) is one series
g = R[R.pass_ == "pass2_slide"]; u = g[~g.planted].set_index("gsm"); p = g[g.planted].set_index("gsm")
def bars(gs, label):
    a = u.A_atlas_rel.reindex(gs).dropna(); n = len(gs)
    if len(a) < 3: return [f"{label}: {len(a)} of {n} tared"], None
    sp = float(np.std(a, ddof=1)); mv = (p.A_atlas_rel.reindex(a.index) - a).dropna(); nf = float(a.between(0.95, 1.05).mean())
    b = dict(B1=len(a) / n >= 0.90, B2=nf >= 0.90, B3=(len(mv) > 0 and mv.mean() / sp >= 2.0))
    return [f"{label}: tared {len(a)} of {n} ({len(a) / n:.3f}) | Normal {nf:.3f} | median {a.median():.5f} | spread {sp:.5f} | planted 1 % moves "
            f"{mv.mean():+.5f} (n {len(mv)}) = {mv.mean() / sp:.2f} spreads"], b
L = [f"METAA-ATLAS-SEAL-02, scored once: {len(S)} healthy monocyte arrays listed ({', '.join(f'{k} {v}' for k, v in S.series.value_counts().items())})"]
l, b = bars(list(S.gsm), "pooled (scored)"); L += l
for s in sorted(S.series.unique()): L += bars(list(S[S.series == s].gsm), f"   {s} (reported)")[0]
if b: L.append(" | ".join(f"{k} {'MET' if v else 'NOT MET'}" for k, v in b.items())); L.append("ALL BARS MET" if all(b.values()) else "NOT ALL BARS MET")
else: L.append("NOT ALL BARS MET (too few tared readings)")
ref = R[R.pass_ == "pass1"]; L.append(f"pass 1: {len(ref)} bundles | Stage 0 {ref.intake.value_counts(dropna=False).to_dict()} | refusals {u.reason.dropna().head(3).tolist()}")
open(os.path.join(out, "seal_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
