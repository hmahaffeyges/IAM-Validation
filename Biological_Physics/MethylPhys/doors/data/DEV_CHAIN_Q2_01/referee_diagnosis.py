"""PHY-084 after the result (exploration on spent counts; the sealed verdict, NOT MET, is unchanged; 2026-11-13). Why did B cells read low?
Three parts, each from committed counts (referee_counts/, the referee lab GSE173787; percall_out/, the atlas lab, Loyfer):
 (1) between types: the same four types in the atlas lab, at zero loose share along IAM-A's measured slope (0.034625), against the one
     prediction (0.0438 for B);
 (2) between laboratories: the referee lab against the atlas lab, type by type, on the same projection;
 (3) the scorer's own slope: the sealed scorer fits each type's slope in its own libraries (Theil-Sen) and reads the intercept, which for B
     lies below the projection along IAM-A's slope.
And the meter: the unmarked side's stray marks as an in-sample calibrator (incomplete conversion moves both sides), molecule length mix, the long-to-short loss ratio and the end-to-inner ratio per laboratory (the forward model used the atlas
lab's molecules). Writes referee_diagnosis_summary.txt."""
import os, json, glob, csv, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); B = 0.034625; TYPES = ("b cells", "cd4 t cells", "cd8 t cells", "monocytes")
S = {r["gsm"]: r["cell"].lower() for r in csv.DictReader(open(os.path.join(H, "referee_samples.csv")))}
def meter(c):
    L = {int(k): v for k, v in c["held_by_length"].items()}; D = {int(k): v for k, v in c["held_by_distance"].items()}; opp = sum(v[2] for v in L.values())
    rs = sum(v[1] for k, v in L.items() if k <= 8) / sum(v[2] for k, v in L.items() if k <= 8); rl = sum(v[1] for k, v in L.items() if k >= 14) / max(1, sum(v[2] for k, v in L.items() if k >= 14))
    return dict(short_opp=sum(v[2] for k, v in L.items() if k <= 8) / opp, mean_len=sum(k * v[0] for k, v in L.items()) / sum(v[0] for v in L.values()),
                long_over_short=rl / rs, end_over_inner=(D[1][0] / D[1][1]) / (sum(D[k][0] for k in D if k >= 4) / sum(D[k][1] for k in D if k >= 4)))
R = []
for f in glob.glob(os.path.join(H, "referee_counts", "*_count.json")):
    c = json.load(open(f)); R.append(dict(lab="referee", cell=S[c["gsm"]], held=c["held_err_length"], loose=c["loose_share"], gain=c["gain_err"], **meter(c)))
for f in glob.glob(os.path.join(H, "percall_out", "*.json")):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and str(x.get("cell", "")).lower() in TYPES and x.get("held_by_length"):
            R.append(dict(lab="atlas", cell=str(x["cell"]).lower(), held=x["held_err_length"], loose=x["loose_share"], gain=x["gain_err"], **meter(x)))
X = pd.DataFrame(R); X["at0"] = X.held - B * X.loose
O = pd.read_csv(os.path.join(H, "referee_out.csv")).set_index("cell")
L = ["PHY-084 after the result: exploration on spent counts; the sealed verdict (NOT MET) is unchanged.", "",
     "type        | prediction | atlas lab at 0 (n) | referee lab at 0 (n) | referee/atlas | sealed scorer intercept"]
for t in TYPES:
    a, r = X[(X.lab == "atlas") & (X.cell == t)], X[(X.lab == "referee") & (X.cell == t)]
    L.append(f"{t:11} | {O.loc[t, 'pred_mid']:.4f}     | {a.at0.median():.4f} ({len(a)})         | {r.at0.median():.4f} ({len(r)})          | {r.at0.median() / a.at0.median():.3f}         | {O.loc[t, 'intercept']:.4f}")
L += ["", "meter, median per laboratory (the four types): " + "; ".join(
      f"{lab}: mean length {g.mean_len.median():.2f} CpGs, short-molecule share {g.short_opp.median():.3f}, long/short loss {g.long_over_short.median():.3f}, end/inner {g.end_over_inner.median():.3f}"
      for lab, g in X.groupby("lab"))]
ga, gr = X[X.lab == "atlas"].gain.median(), X[X.lab == "referee"].gain.median()
ha, hr = X[X.lab == "atlas"].at0.median(), X[X.lab == "referee"].at0.median()
L += [f"calibrator, the unmarked side's stray marks (gain_err): atlas {ga:.4f}, referee {gr:.4f} ({gr / ga:.3f}); the marked side at 0: atlas {ha:.4f}, referee {hr:.4f} ({hr / ha:.3f})",
      f"incomplete conversion c raises the unmarked side by c and lowers the marked side by the factor (1 - c): the observed rise {gr - ga:.4f} accounts for {100 * (gr - ga) / (1 - hr / ha):.1f} % of the marked side's fall, so conversion is not the cause"]
open(os.path.join(H, "referee_diagnosis_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
