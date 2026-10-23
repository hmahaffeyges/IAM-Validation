"""METAA-ATLAS-SEAL-02, synthetic test (written and committed 2026-10-23 before it runs and before the seal; constructed arrays, nothing real).
The instrument's job: a healthy purified monocyte, read against the atlas's monocytes and tared against healthy monocytes on its own slide, reads
Normal; a 1 % loss of its pattern stands out. This test runs the sealed design end to end through the one command, on arrays whose answer is known.
Constructed arrays: the atlas's monocyte best value at every monocyte identity site, softened or hardened by the slide's offset d
(beta + d x (0.5 - beta)), plus each site's healthy person-to-person spread (donor_sd), clipped to (0, 1); fixed seed. Six slides of 8, 8, 4, 4, 4
and 8 arrays (the sealed list's layout), offsets -0.03, 0, +0.03, +0.06, -0.03, +0.03. Every reading by chain/MethylPhys_Interface/run_sample.py:
pass 1 without references; pass 2 with --metaa-atlas-ref-A from the other arrays on the same slide; the planted copy (stage_metaa_atlas.plant, as
run_series.py plants) read with the same references. rows.csv by run_series.py table; scored by the sealed scorer seal_02_score.py, unchanged.
Bars: the sealed scorer's three (B1 tared share, B2 Normal share, B3 plant in spreads) must all be met on constructed arrays.
Reported: the untared reading by slide (the offset the same-slide tare must cancel).   Usage: python3 synthetic_02.py OUTDIR"""
import os, sys, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); SR = os.path.join(CH, "MethylPhys_Interface", "run_series.py")
out = os.path.abspath(sys.argv[1]); d1, d2, Wd = (os.path.join(out, x) for x in ("pass1", "pass2_slide", "work")); [os.makedirs(x, exist_ok=True) for x in (d1, d2, Wd)]
SL = [(8, -0.03), (8, 0.0), (4, 0.03), (4, 0.06), (4, -0.03), (8, 0.03)]; SPEC = "isolated monocytes"
S = SA.identity_sites("monocytes"); mu = SA._col("mu", "monocytes").reindex(S).to_numpy(); sd = SA._col("donor_sd", "monocytes").reindex(S).fillna(0).to_numpy()
rng = np.random.default_rng(20261023); rows = []
for k, (n, d) in enumerate(SL):
    for j in range(n):
        g = f"SYN{k}_{j:02d}"; x = np.clip(mu + d * (0.5 - mu) + rng.normal(0, 1, len(S)) * sd, 1e-4, 1 - 1e-4)
        f = os.path.join(Wd, g + ".csv"); pd.Series(x, index=S, name="beta").to_frame().to_csv(f); rows.append(dict(gsm=g, slide=f"slide{k}", series="constructed", offset=d, f=f))
L0 = pd.DataFrame(rows); L0[["gsm", "slide", "series"]].to_csv(os.path.join(out, "synthetic_02_samples.csv"), index=False)
def run(args, log):
    with open(log, "w") as fh: return subprocess.run([sys.executable, RS] + args, stdout=fh, stderr=subprocess.STDOUT).returncode
A1 = {}
for r in rows:
    rc = run(["--betas", r["f"], "--specimen", SPEC, "--id", r["gsm"], "--out", os.path.join(Wd, r["gsm"] + ".html"), "--bundle", os.path.join(d1, r["gsm"] + ".json")], os.path.join(d1, r["gsm"] + ".log"))
    b = json.load(open(os.path.join(d1, r["gsm"] + ".json"))) if rc == 0 else {}; A1[r["gsm"]] = ((b.get("development") or {}).get("metaa_atlas") or {}).get("A"); print("pass 1", r["gsm"], rc, A1[r["gsm"]], flush=True)
for r in rows:
    at = [str(A1[q["gsm"]]) for q in rows if q["slide"] == r["slide"] and q["gsm"] != r["gsm"] and A1[q["gsm"]] is not None]
    ref = ["--metaa-atlas-ref-A", ",".join(at)] if at else []
    run(["--betas", r["f"], "--specimen", SPEC, "--id", r["gsm"]] + ref + ["--out", os.path.join(Wd, r["gsm"] + "_2.html"), "--bundle", os.path.join(d2, r["gsm"] + ".json")], os.path.join(d2, r["gsm"] + ".log"))
    pv = os.path.join(Wd, r["gsm"] + "_planted.csv"); SA.plant(pd.read_csv(r["f"], index_col=0).iloc[:, 0], "monocytes").rename("beta").to_frame().to_csv(pv)
    run(["--betas", pv, "--specimen", SPEC, "--id", r["gsm"] + "_planted", "--banner", "PLANTED: a constructed 1 % loss of the monocytes pattern, not a person."] + ref
        + ["--out", os.path.join(Wd, r["gsm"] + "_planted.html"), "--bundle", os.path.join(d2, r["gsm"] + "_planted.json")], os.path.join(d2, r["gsm"] + "_planted.log")); print("pass 2", r["gsm"], len(at), "refs", flush=True)
subprocess.run([sys.executable, SR, "table", out], check=True)
sc = subprocess.run([sys.executable, os.path.join(H, "seal_02_score.py"), os.path.join(out, "rows.csv"), os.path.join(out, "synthetic_02_samples.csv"), out], capture_output=True, text=True)
L = ["METAA-ATLAS-SEAL-02 synthetic test: 36 constructed monocyte arrays on six slides with offsets, every reading by run_sample.py, scored by the sealed scorer",
     "untared A by slide (offset): " + "; ".join(f"slide{k} ({d:+.2f}) {np.nanmedian([A1[r['gsm']] for r in rows if r['slide'] == f'slide{k}' and A1[r['gsm']] is not None]):.4f}" for k, (n, d) in enumerate(SL))]
L += sc.stdout.strip().split("\n") + ([sc.stderr.strip()[-400:]] if sc.returncode else [])
open(os.path.join(out, "synthetic_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
