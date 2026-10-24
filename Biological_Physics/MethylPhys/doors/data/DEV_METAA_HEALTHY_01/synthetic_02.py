"""DEV-METAA-HEALTHY-01 synthetic test 2 (written and committed 2026-10-24 before it runs; replaces test 1, which called the Met-A stage
directly and took its shares from a file copied by hand, so under the chain-only rule it does not count; its two runs stay recorded).
Everything through run_sample.py: (1) the healthy whole blood GSE235717 GSM7508429 (spent) read from its raw IDATs, which gives a real
healthy A / A_pred and the IAM-Deconvolver's shares; (2) synthetic arrays: IAM-Atlas v2.1 mixed in those shares at every atlas site with an
expectation, through a known line and noise per probe design (mild, typical, weak), with and without a 5 % loss of the held pattern at every
site except the atlas ruler's anchors; each written as a betas file and read by run_sample.py, so the self-tare and the Deconvolver act on it
as on a real array. Bar 1 as recorded: healthy A / A_pred within 0.005 of 1; the loss raises A / A_pred by at least half as much as it raises A.
Usage: python3 synthetic_02.py OUTDIR"""
import os, sys, json, re, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); G = "GSM7508429"
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
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", b], capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tag}: run_sample.py exit {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}")
    os.remove(f); return json.load(open(b))
def hp(B):
    w = B.get("met_a_whole") or {}; h = w.get("healthy_prediction") or {}
    return w.get("A"), h.get("A_pred"), h.get("A_over_pred"), h.get("blur"), h.get("reason")
B0, _, S0 = one("real_GSM7508429", []); A0, P0, R0, bl0, why0 = hp(B0)
L = ["DEV-METAA-HEALTHY-01 synthetic test 2, every array through run_sample.py",
     f"real healthy blood GSM7508429 (raw IDATs): A {A0} | A_pred {P0} | A/A_pred {R0} | measured line {bl0}" + (f" | reason {why0}" if why0 else "")]
print(L[-1], flush=True)
F = pd.Series({k: v for k, v in S0.items() if v > 0}); des, sets, _ = SA._ruler(); idx = des.index
MU = pd.concat([SA._col("mu", k).reindex(idx).rename(k) for k in F.index], axis=1); have = MU.notna()
cover = have.mul(F, axis=1).sum(1) / F.sum(); e = (MU.fillna(0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1)).where(cover >= SA.MIX_COVER).dropna()
del MU, have; d = des.reindex(e.index).to_numpy(); anc = e.index.isin(sum(sets.values(), [])); g = np.random.default_rng(20261024)
GRID = [("mild", {"I": (0.02, 0.95, 0.02), "II": (0.04, 0.90, 0.03)}), ("typical", {"I": (0.03, 0.93, 0.03), "II": (0.06, 0.85, 0.04)}),
        ("weak", {"I": (0.05, 0.88, 0.05), "II": (0.09, 0.78, 0.06)})]
def array(t, P):
    x = np.empty(len(t))
    for k, (a, b, s) in P.items():
        m = d == k; x[m] = a + b * t[m] + s * g.standard_normal(m.sum())
    return pd.Series(np.clip(x, 0, 1), index=e.index)
L.append(f"synthetic arrays: {len(e)} sites, the atlas in the shares the chain found for GSM7508429; seed 20261024")
for name, P in GRID:
    t1 = e.copy(); t1[~anc] = e[~anc] + 0.05 * (0.5 - e[~anc])
    Ah, Ph, Rh, blh, wh = hp(betas_run(f"{name}_healthy", array(e.to_numpy(), P)))
    Al, Pl, Rl, _, _ = hp(betas_run(f"{name}_loss", array(t1.to_numpy(), P)))
    L.append(f"{name:8} healthy: A {Ah} | A_pred {Ph} | A/A_pred {Rh} | measured line {blh}" + (f" | reason {wh}" if wh else ""))
    if None not in (Ah, Al, Rh, Rl): L.append(f"{'':8} 5 % loss off the anchors: A {Al} (+{Al - Ah:.4f}) | A/A_pred {Rl} (+{Rl - Rh:.4f})")
    print(L[-2]); print(L[-1], flush=True)
for f in files.values(): os.remove(f)
open(os.path.join(out, "synthetic_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
