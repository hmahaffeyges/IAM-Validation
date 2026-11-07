"""DEV-METAA-HEALTHY-01 synthetic test 5 (written and committed 2026-11-07 before it runs): test 4 again after the --dev-line-censored flag
was found inside the sequencing branches only (test 4 ran the plain line); seed 20261117 + DRAW; writes synthetic_05_draw<DRAW>.json. Test 4: the line measured with readings at 0 or 1
counted as censored (run_sample.py --dev-line-censored), ONE NOISE DRAW per run: Usage: python3 synthetic_04.py OUTDIR DRAW (DRAW 0 to 9;
seed 20261107 + DRAW). Each draw reads the three real-range arrays of test 3 (center, low tail, high tail) through the one command; draw 0
also reads their 5 % loss and test 3's weak array (recorded, not scored). Writes OUTDIR/synthetic_04_draw<DRAW>.json; synthetic_04_score.py
combines the ten. Test 3's text follows.
Test 3 (written and committed 2026-11-07 before it runs). Test 2's grid (mild, typical, weak) was set without
measuring real arrays; the line and noise the chain measured on 331 real arrays (R14 lung EPIC, the 450K blood rehearsal; array_lines.csv in the
record's working notes) lie inside slope 0.993 to 1.030 and noise 0.003 to 0.024, so test 3 spans that range and both tails; test 2's weak array
is kept, recorded, not scored. Same construction as test 2 (its text follows).
Test 2 (written and committed 2026-10-24 before it runs; replaced test 1, which called the Met-A stage
directly and took its shares from a file copied by hand, so under the chain-only rule it does not count; its two runs stay recorded).
Everything through run_sample.py: (1) the healthy whole blood GSE235717 GSM7508429 (spent) read from its raw IDATs, which gives a real
healthy A / A_pred and the IAM-Deconvolver's shares; (2) synthetic arrays: IAM-Atlas v2.1 mixed in those shares at every atlas site with an
expectation, through a known line and noise per probe design (mild, typical, weak), with and without a 5 % loss of the held pattern at every
site except the atlas ruler's anchors; each written as a betas file and read by run_sample.py, so the self-tare and the Deconvolver act on it
as on a real array. Bar 1 as recorded: healthy A / A_pred within 0.005 of 1; the loss raises A / A_pred by at least half as much as it raises A.
Usage: python3 synthetic_02.py OUTDIR"""
import os, sys, json, re, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; DRAW = int(sys.argv[2]); os.makedirs(out, exist_ok=True); G = "GSM7508429"
base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{G[:7]}nnn/{G}/suppl/"; files = {}
for k in range(5):
    try:
        for n in sorted(set(re.findall(rf"{G}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(base, timeout=120).read().decode()))):
            f = os.path.join(out, n); urllib.request.urlretrieve(base + n, f); files["grn" if "_Grn" in n else "red"] = f
        break
    except Exception as e: print("download retry", type(e).__name__, flush=True); time.sleep(60 * (k + 1))
assert set(files) == {"grn", "red"}, files
def one(tag, extra):
    b = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--grn", files["grn"], "--red", files["red"], "--specimen", "whole blood", "--id", G, "--out",
                        os.path.join(out, tag + ".html"), "--bundle", b] + extra, capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tag}: run_sample.py exit {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}")
    B = json.load(open(b)); return B, {x["cell"]: x for x in (B.get("met_a_each_cell") or {}).get("cells", [])}, {x["cell"]: x["share"] for x in B["cell_shares"]["cells"]}
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
def betas_run(tag, x):
    f = os.path.join(out, tag + "_betas.csv"); pd.DataFrame({"cpg_id": x.index, "beta": x.round(4).to_numpy()}).to_csv(f, index=False)
    b = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", b, "--dev-line-censored"], capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tag}: run_sample.py exit {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}")
    os.remove(f); return json.load(open(b))
def hp(B):
    w = B.get("met_a_whole") or {}; h = w.get("healthy_prediction") or {}
    return w.get("A"), h.get("A_pred"), h.get("A_over_pred"), h.get("blur"), h.get("reason")
B0, _, S0 = one("real_GSM7508429", []); A0, P0, R0, bl0, why0 = hp(B0)
L = ["DEV-METAA-HEALTHY-01 synthetic test 3, every array through run_sample.py",
     f"real healthy blood GSM7508429 (raw IDATs): A {A0} | A_pred {P0} | A/A_pred {R0} | measured line {bl0}" + (f" | reason {why0}" if why0 else "")]
print(L[-1], flush=True)
F = pd.Series({k: v for k, v in S0.items() if v > 0}); des, sets, _ = SA._ruler(); idx = des.index
MU = pd.concat([SA._col("mu", k).reindex(idx).rename(k) for k in F.index], axis=1); have = MU.notna()
cover = have.mul(F, axis=1).sum(1) / F.sum(); e = (MU.fillna(0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1)).where(cover >= SA.MIX_COVER).dropna()
del MU, have; d = des.reindex(e.index).to_numpy(); anc = e.index.isin(sum(sets.values(), [])); g = np.random.default_rng(20261117 + DRAW)
GRID = [("center", {"I": (-0.0033, 1.0026, 0.0066), "II": (-0.0121, 1.0175, 0.0098)}),            # the real arrays' medians
        ("low tail", {"I": (0.0022, 0.9928, 0.0148), "II": (-0.0036, 0.9991, 0.0240)}),        # the flattest, noisiest real arrays (beyond the 1st percentile)
        ("high tail", {"I": (-0.0059, 1.0085, 0.0148), "II": (-0.0183, 1.0301, 0.0240)}),      # the steepest, noisiest
        ("weak (beyond the real range; recorded, not scored)", {"I": (0.05, 0.88, 0.05), "II": (0.09, 0.78, 0.06)})]
def array(t, P):
    x = np.empty(len(t))
    for k, (a, b, s) in P.items():
        m = d == k; x[m] = a + b * t[m] + s * g.standard_normal(m.sum())
    return pd.Series(np.clip(x, 0, 1), index=e.index)
L.append(f"synthetic arrays: {len(e)} sites, the atlas in the shares the chain found for GSM7508429; seed 20261024")
RES = {}
for name, P in GRID:
    if name.startswith("weak") and DRAW != 0: continue
    t1 = e.copy(); t1[~anc] = e[~anc] + 0.05 * (0.5 - e[~anc])
    Ah, Ph, Rh, blh, wh = hp(betas_run(f"{name.split()[0]}_{DRAW}_healthy", array(e.to_numpy(), P)))
    RES[name] = dict(A=Ah, A_pred=Ph, ratio=Rh, line=blh, reason=wh)
    if DRAW == 0:
        Al, Pl, Rl, _, _ = hp(betas_run(f"{name.split()[0]}_{DRAW}_loss", array(t1.to_numpy(), P))); RES[name].update(A_loss=Al, ratio_loss=Rl)
    print(name, RES[name], flush=True)
import json as _j
_j.dump(dict(draw=DRAW, seed=20261117 + DRAW, real=dict(A=A0, A_pred=P0, ratio=R0), arrays=RES), open(os.path.join(out, f"synthetic_05_draw{DRAW}.json"), "w"), indent=1)
for f in files.values(): os.remove(f)
