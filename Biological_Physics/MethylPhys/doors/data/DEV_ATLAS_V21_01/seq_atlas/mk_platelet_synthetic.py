"""DEV-ATLAS-V21-01, plan 60.24, synthetic for the platelet test of the megakaryocyte column (written and committed before it runs, and before the
platelet file is opened; development). The test will read a deep level track pooled from 3 people (no read counts) against every cell of the
IAM Sequencing Atlas plus the megakaryocyte column, by the chain's own comparison (run_sample.py --build seq-compare, ranked by the counting-
weighted score). The question this synthetic answers: does a level-only sample, converted by the chain (--build seq-table, format
level-bedgraph, LEVEL_N reads at each CpG), come out nearest its own cell when that cell's column is shallow and the other columns deep?
Construction (seed 20261106), on the first 2,000,000 CpGs of the hg19 numbering, written as constructed atlas pieces of 250,000 CpGs in the
atlas's own column form (cpg; per cell level, reads, meth, spread, halfA, halfB; uint16, 65535 blank):
  - 12 deep cells, each with its own true level per CpG (bimodal: most CpGs near 0.03 or 0.92, some between; a cell differs from the shared
    profile at 8 % of CpGs), each read from 6 donors at about 20 reads a donor, donor spread 0.03;
  - one shallow cell "target" (the megakaryocyte column's case): 2 donors at about 2 reads each, at 40 % of CpGs only;
  - the sample: target's true level, pooled from 3 people (each person's level within 0.03 of the truth), written as a bedGraph of levels
    (rounded to 0.01, as a published track), and only at CpGs a 30x pooled sample would hold.
A second sample is the same from one of the deep cells ("deep b"), to show a deep cell is also found. Bars (both read through the one command):
1. the target sample's nearest cell by the weighted score is target; 2. the deep-b sample's nearest cell by the weighted score is deep b.
Recorded: each sample's weighted scores and its rank by plain distance. Usage: python3 mk_platelet_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, gzip, subprocess, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
IF = os.path.join(MP, "chain", "MethylPhys_Interface"); sys.path.insert(0, os.path.join(MP, "chain")); import stage_seq_atlas as Q
CPG_BED = os.environ.get("CPG_BED"); assert CPG_BED and os.path.exists(CPG_BED), "set CPG_BED to the hg19 CpG dictionary (cpg/CpG.bed.gz)"
rng = np.random.default_rng(20261106); NC = int(os.environ.get("NC", 2_000_000)); BL = 65535
base = np.where(rng.random(NC) < 0.55, rng.beta(30, 2, NC), np.where(rng.random(NC) < 0.8, rng.beta(1.5, 40, NC), rng.uniform(0.15, 0.85, NC)))
cells = [f"deep {chr(97 + i)}" for i in range(12)] + ["target"]; truth = {}
for c in cells:
    t = base.copy(); d = rng.random(NC) < 0.08; t[d] = np.where(t[d] > 0.5, rng.uniform(0.02, 0.4, d.sum()), rng.uniform(0.6, 0.98, d.sum())); truth[c] = np.clip(t, 0, 1)
def column(c, donors, depth, cover):
    lv = np.clip(truth[c][None, :] + rng.normal(0, 0.03, (donors, NC)), 0, 1); n = rng.poisson(depth, (donors, NC)); n[:, rng.random(NC) > cover] = 0
    m = rng.binomial(n, lv); R = n.sum(0); M = m.sum(0); has = R > 0
    half = donors // 2; hA = np.where(n[:half].sum(0) > 0, m[:half].sum(0) / np.maximum(n[:half].sum(0), 1), np.nan); hB = np.where(n[half:].sum(0) > 0, m[half:].sum(0) / np.maximum(n[half:].sum(0), 1), np.nan)
    per = np.where(n > 0, m / np.maximum(n, 1), np.nan); sp = np.nanstd(per, axis=0) if donors > 1 else np.zeros(NC)
    u = lambda x: np.where(np.isfinite(x), np.rint(np.clip(x, 0, 1) * 1e4), BL).astype(np.uint16)
    return {"level": u(np.where(has, M / np.maximum(R, 1), np.nan)), "reads": np.where(has, np.minimum(R, 65534), BL).astype(np.uint16), "meth": np.where(has, np.minimum(M, 65534), BL).astype(np.uint16),
            "spread": u(np.where(has, np.nan_to_num(sp), np.nan)), "halfA": u(hA), "halfB": u(hB)}
import warnings; warnings.filterwarnings("ignore")
COL = {c: column(c, 6, 20, 0.97) for c in cells[:-1]}; COL["target"] = column("target", 2, 2, 0.40)
PD = os.path.join(W, "pieces"); os.makedirs(PD, exist_ok=True)
for s0 in range(0, NC, 250_000):
    e = s0 + 250_000; T = {"cpg": np.arange(s0 + 1, e + 1, dtype=np.int32)}
    for c in cells:
        for k, v in COL[c].items(): T[f"{c}|{k}"] = v[s0:e]
    pq.write_table(pa.table(T), os.path.join(PD, f"piece_{s0:08d}_{e:08d}.parquet"))
D = Q.hg19_dictionary(CPG_BED); rows = []
for chrom, (pos, ix) in D.items():
    k = ix <= NC
    if k.any(): rows.append(pd.DataFrame(dict(chr=chrom, pos=pos[k], ix=ix[k])))
P = pd.concat(rows).sort_values("ix"); res = {}; L = [f"constructed atlas: {len(cells)} cells on {NC:,} CpGs; target column 2 donors at about 2 reads, 40 % of CpGs"]
for name in ("target", "deep b"):
    lv = np.mean([np.clip(truth[name] + rng.normal(0, 0.03, NC), 0, 1) for _ in range(3)], axis=0); held = rng.poisson(30, NC) >= 5
    Pk = P[held[P.ix.to_numpy() - 1]]; bg = os.path.join(W, f"{name.replace(' ', '_')}.bedGraph.gz")
    with gzip.open(bg, "wt") as f:
        f.write("track type=bedGraph\n"); pd.DataFrame(dict(c=Pk.chr, s=Pk.pos - 1, e=Pk.pos + 1, v=np.round(lv[Pk.ix.to_numpy() - 1], 2))).to_csv(f, sep="\t", header=False, index=False)
    lb = bg.replace(".bedGraph.gz", ".lbeta"); J = lb.replace(".lbeta", "_compare.json")
    r1 = subprocess.run([sys.executable, "run_sample.py", "--build", "seq-table", "--build-input", bg, "--build-out", lb, "--build-samples", f"level-bedgraph:hg19:{CPG_BED}"], cwd=IF, capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, "run_sample.py", "--build", "seq-compare", "--build-input", lb, "--build-samples", PD, "--bundle", J], cwd=IF, capture_output=True, text=True)
    R = json.load(open(J)) if os.path.exists(J) else {}; wr = R.get("weighted_ranked") or []; rk = R.get("ranked") or []
    res[name] = wr[0] if wr else None
    L.append(f"{name} sample: converted exit {r1.returncode} {r1.stdout.strip()[-160:]}; compared exit {r2.returncode}; sites {R.get('sites')}; nearest by weighted score {wr[:3]} "
             f"({', '.join(str(R['cells'][c].get('weighted')) for c in wr[:3])}); its rank by plain distance {rk.index(name) + 1 if name in rk else None} of {len(rk)}"
             + ("" if r2.returncode == 0 else " | " + (r2.stdout + r2.stderr)[-300:].replace("\n", " ")))
    print(L[-1], flush=True)
b1 = res.get("target") == "target"; b2 = res.get("deep b") == "deep b"
L += [f"bar 1 (the target sample nearest its shallow column): {'met' if b1 else 'not met'}", f"bar 2 (the deep-b sample nearest deep b): {'met' if b2 else 'not met'}", "synthetic: " + ("met" if b1 and b2 else "not met")]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-3:]))
