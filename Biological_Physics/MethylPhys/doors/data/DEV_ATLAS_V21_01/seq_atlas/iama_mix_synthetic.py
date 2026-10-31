"""IAM-A in the sample's mix, against the IAM Sequencing Atlas's held-error reference (stage_seq_atlas.iama_in_mix): synthetic, written and
committed before it runs. The instrument's job: a healthy mixed sample reads 1, and the reading moves with the held error.

Known answer: molecule files (.pat) built from the reference's own rates. In every tile where each planted cell has at least 50
opportunities, 40 molecules of 10 CpGs, each from a planted cell drawn by its share; each interior CpG unmethylated with chance p, p solved
so that a lone error (unmethylated between two methylated) occurs at the cell's counted rate in that tile (p (1 - p)^2 = rate). Read by the
one command with the full chain and the planted shares declared (run_sample.py --pat FILE --dev-seq-values 1 --dev-seq-ref auto
--full-chain --specimen unknown --dev-shares ...). Samples: (1) b cells alone; (2) a plasma-like mix (monocytes 0.4, cd4 t cells 0.2,
nk cells 0.1, hepatocyte 0.1, lung alveolar endothelium 0.2); (3) the same mix read against donor half B only (run 2, amended before it ran: run 1 used half A, which holds no hepatocyte donor, disclosed); (4) the mix with every rate
raised 20 %. Seed 20261103. Bars (operational, set before the run): samples 1 to 3 within 0.02 of 1; sample 4 within 0.03 of 1.2.
Usage: python3 iama_mix_synthetic.py WORKDIR OUT.txt"""
import sys, os, gzip, json, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q
H = pd.read_parquet(Q.HELD_REF); rng = np.random.default_rng(20261104); L0, M = 10, 40   # run 2 (run 1 seed 20261103)
def solve(r):
    p = r.copy()
    for _ in range(30): p = r / (1 - p) ** 2
    return p
def build(mx, path, sfx="", scale=1.0):
    ok = np.all([H[f"{c}|opp{sfx}"].to_numpy() >= Q.HELD_MIN_OPP for c in mx], 0); T = H[ok]; cells = list(mx); f = np.array([mx[c] for c in cells])
    R = np.stack([(T[f"{c}|iso{sfx}"] / T[f"{c}|opp{sfx}"]).to_numpy() for c in cells], 1) * scale; P = solve(R)
    with gzip.open(path, "wt") as fo:
        for i, (ch, t) in enumerate(zip(T.chrom.to_numpy(), T.tile.to_numpy())):
            who = rng.choice(len(cells), M, p=f / f.sum()); starts = np.sort(t * 2000 + 1 + rng.integers(0, 2000 - L0, M))
            for s0, k in zip(starts, who):
                m = rng.random(L0 - 2) < P[i, k]; pat = "C" + "".join("T" if x else "C" for x in m) + "C"
                fo.write(f"{ch}\t{s0}\t{pat}\t1\n")
    return int(ok.sum())
MIX = {"monocytes": 0.4, "cd4 t cells": 0.2, "nk cells": 0.1, "hepatocyte": 0.1, "lung alveolar endothelium": 0.2}
runs = [("b cells alone", {"b cells": 1.0}, "", 1.0, None, 1.0), ("plasma-like mix", MIX, "", 1.0, None, 1.0),
        ("plasma-like mix, donor half B", MIX, "_dB", 1.0, "B", 1.0), ("plasma-like mix, every rate x1.2", MIX, "", 1.2, None, 1.2)]
L = ["IAM-A in the sample's mix, synthetic: 4 constructed molecule files through the one command", "sample | tiles built | A | expected | tiles read | bar"]; ok = []
for j, (name, mx, sfx, sc, half, want) in enumerate(runs):
    f = os.path.join(W, f"m{j}.pat.gz"); nt = build(mx, f, sfx, sc); b = os.path.join(W, f"m{j}_bundle.json")
    cmd = [sys.executable, "run_sample.py", "--pat", f, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--full-chain", "--specimen", "unknown",
           "--dev-shares", ",".join(f"{c}:{v}" for c, v in mx.items()), "--id", f"m{j}", "--out", os.path.join(W, f"m{j}.html"), "--bundle", b] + (["--dev-seq-ref-half", half] if half else [])
    subprocess.run(cmd, cwd=os.path.join(C, "MethylPhys_Interface"), capture_output=True, text=True)
    r = (json.load(open(b)).get("iam_a_mix") or {}) if os.path.exists(b) else {"reason": "no bundle"}
    A = r.get("A"); g = A is not None and abs(A - want) <= (0.02 if want == 1.0 else 0.03); ok.append(g)
    L.append(f"{name} | {nt} | {A} | {want} | {r.get('tiles_read')} | {'met' if g else 'not met'}" + ("" if A is not None else f" ({r.get('reason')})")); os.remove(f)
L.append(f"synthetic: {'met' if all(ok) else 'not met'}"); open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
