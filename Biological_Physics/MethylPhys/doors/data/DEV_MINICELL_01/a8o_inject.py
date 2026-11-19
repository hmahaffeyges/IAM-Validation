import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt'], check=True, timeout=900)
#!/usr/bin/env python3
"""A8O orders test 2 as an INJECTION test (register A8O item 21c; prediction written before this ran). Runs ON A GITHUB MACHINE. One spent sorted
library (GSM5652282, CD8 T cells, IAM Sequencing Atlas; already used by the planted-mixture test T1) thinned to 15 M molecules; extra loss planted
at a MATCHED rate (+0.02 per methylated call) in three patterns, plus the unplanted baseline; each file read by THE ONE COMMAND with every default
(run_sample.py --pat, specimen 'sorted cells', no cell named). Patterns: UNIFORM (each methylated call lost with p = 0.02: the mechanical control,
identical to orders reaching each cell on its own schedule at scattered sites); STRETCH (5 % of molecules, cells under an order there, lose each
methylated call with p = 0.40); SYNC (5 % of 10-CpG regions, every molecule there loses each methylated call with p = 0.40). Read from the chain's
bundle: held error, A, A_derived, coupling, loss dispersion, loss-run ratio, molecule mid share, loose share, cell named."""
import os, sys, json, subprocess, tempfile, time, math
import numpy as np, pandas as pd
IF = "Biological_Physics/MethylPhys/chain/MethylPhys_Interface"; CH = "Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH); import stage_q0_intake as Q0
G = "GSM5652282"; NTOT = int(os.environ.get("INJ_NTOT", "15000000")); P_U, P_H, F = 0.02, 0.40, 0.05
os.makedirs("xout", exist_ok=True); W = tempfile.mkdtemp(prefix="inj"); rng = np.random.default_rng(11)
def chunks(f):
    for d in pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chr", "idx", "pat", "n"], dtype={"chr": str, "idx": np.int64, "pat": str, "n": np.int64}, chunksize=2_000_000):
        yield d
src = Q0.geo_pat(G, W)["path"]; tot = int(sum(d["n"].sum() for d in chunks(src))); p = min(1.0, NTOT / tot)
base = os.path.join(W, "base.tsv"); k = 0
with open(base, "w") as fo:
    for d in chunks(src):
        d["n"] = rng.binomial(d["n"].to_numpy(), p); d = d[d["n"] > 0]
        d = d.loc[d.index.repeat(d["n"])].assign(n=1); k += len(d); d.to_csv(fo, sep="\t", header=False, index=False)
os.remove(src); print(f"::notice::INJ source {G}: {tot:,} molecules, kept {k:,}", flush=True)
def lose_calls(pat, q, r):
    s = list(pat)
    for i, ch in enumerate(s):
        if ch == "C" and r.random() < q: s[i] = "T"
    return "".join(s)
def lose_atleast_one(pat, q, r):
    idx = [i for i, ch in enumerate(pat) if ch == "C"]; c = len(idx)
    if c == 0: return pat
    pk = np.array([math.comb(c, j) * q ** j * (1 - q) ** (c - j) for j in range(1, c + 1)]); kk = 1 + int(r.choice(c, p=pk / pk.sum()))
    s = list(pat)
    for i in r.choice(idx, kk, replace=False): s[i] = "T"
    return "".join(s)
def plant(mode, out):
    r = np.random.default_rng({"uniform": 21, "stretch": 22, "sync": 23}[mode]); hit_regions = None
    with open(out, "w") as fo:
        for d in pd.read_csv(base, sep="\t", header=None, names=["chr", "idx", "pat", "n"], dtype={"chr": str, "idx": np.int64, "pat": str, "n": np.int64}, chunksize=2_000_000):
            pats = d["pat"].to_numpy(dtype=object).copy()
            if mode == "uniform":
                c = d["pat"].str.count("C").to_numpy(); sel = np.where(r.random(len(d)) < 1 - (1 - P_U) ** c)[0]
                for i in sel: pats[i] = lose_atleast_one(pats[i], P_U, r)
            elif mode == "stretch":
                for i in np.where(r.random(len(d)) < F)[0]: pats[i] = lose_calls(pats[i], P_H, r)
            elif mode == "sync":
                reg = (d["idx"].to_numpy() // 10)
                hashed = (reg * 2654435761 % (2 ** 32)) / 2 ** 32
                for i in np.where(hashed < F)[0]: pats[i] = lose_calls(pats[i], P_H, r)
            d["pat"] = pats; d.to_csv(fo, sep="\t", header=False, index=False)
def find(dct, key):
    if isinstance(dct, dict):
        if key in dct and isinstance(dct[key], (int, float, str)) and not isinstance(dct[key], bool): return dct[key]
        for v in dct.values():
            x = find(v, key)
            if x is not None: return x
    if isinstance(dct, list):
        for v in dct:
            x = find(v, key)
            if x is not None: return x
RES = {}
for mode in ("baseline", "uniform", "stretch", "sync"):
    t0 = time.time(); tsv = base if mode == "baseline" else os.path.join(W, f"{mode}.tsv")
    if mode != "baseline": plant(mode, tsv)
    pg = os.path.join(W, f"{mode}.pat.gz"); subprocess.run(f"sort -k2,2n -k3,3 -S 1G -T {W} {tsv} | gzip -1 > {pg}", shell=True, check=True)
    if mode != "baseline": os.remove(tsv)
    b = os.path.join(W, f"{mode}.json")
    with open(os.path.join("xout", f"inj_{mode}_log.txt"), "w") as fh:
        rc = subprocess.run(["python3", "run_sample.py", "--pat", pg, "--specimen", "sorted cells", "--id", f"INJ_{mode}", "--out", os.path.join(W, f"{mode}.html"), "--bundle", b],
                            cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
    B = json.load(open(b)) if os.path.exists(b) else {}; ia = B.get("iam_a") or {}
    R = {kk: find(ia, kk) for kk in ("A", "A_derived", "held_err", "coupling", "loss_dispersion", "loss_run_ratio", "mol_mid_share", "mol_low_share", "loose_share", "cell", "refusal_code")}
    R.update(rc=rc, minutes=round((time.time() - t0) / 60, 1)); RES[mode] = R
    if os.path.exists(b): subprocess.run(["cp", b, os.path.join("xout", f"inj_{mode}_bundle.json")])
    os.remove(pg)
    print("::notice::INJ " + mode + " " + json.dumps({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in R.items()}), flush=True)
    if rc != 0 and mode == "baseline": print("::error::" + open(os.path.join("xout", "inj_baseline_log.txt")).read()[-1500:].replace("\n", " | ")); break
json.dump(RES, open("xout/a8o_inject.json", "w"), indent=1); print("::notice::INJ done", flush=True)
