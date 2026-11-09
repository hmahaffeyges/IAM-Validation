"""Synthetic for the IAM-Deconvolver weighted by each site's read count on sequencing (run_sample.py --dev-seq-noise-weight; written and
committed 2026-11-09 before it runs; development). The instrument's job: a sequenced sample's cells are read closer to what is in it when each
site counts by the reads it rests on (the model's binomial noise, as each sky pixel counts by its own noise) than when every site counts alike.
Six constructed samples from the IAM Sequencing Atlas's own levels at IAM-Atlas v2.1's sites (stage_metaa_atlas._col with SEQ_ATLAS): three
blood-like mixes and three plasma-like (blood with erythroid, liver and endothelial DNA), each site read as k of n reads, n Poisson around a
depth that varies site to site (lognormal, sd 1), k Binomial(n, level), seeded, written as a .beta count file (hg19 CpG numbering). Depths:
plasma-like 6 with a third of the sites read (as the healthy GSE186007 plasmas), blood-like 6 and 20. Each read twice by the one command
(--dev-seq-beta), with and without --dev-seq-noise-weight. PART p (0 to 5): sample p. Usage: seq_noise_weight_synthetic.py OUTDIR PART"""
import os, sys, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_atlas as SQ, stage_seq_values as SV
out = sys.argv[1]; PART = int(sys.argv[2]); os.makedirs(out, exist_ok=True)
SA.SEQ_ATLAS = SQ.ARRAY_COLS
BLOOD = {"granulocytes": 0.55, "monocytes": 0.12, "cd4 t cells": 0.13, "cd8 t cells": 0.08, "b cells": 0.06, "nk cells": 0.06}
ADD = {"erythrocyte progenitors": 0.08, "liver hepatocytes": 0.06, "vascular endothelial cells": 0.10}
g = np.random.default_rng(20261109 + PART); kind = "plasma" if PART >= 3 else "blood"; depth = 6 if (kind == "plasma" or PART == 0) else (20 if PART == 1 else 6)
mix = dict(BLOOD); j = g.dirichlet(np.full(len(mix), 40.0)); mix = {c: float(v) for c, v in zip(mix, j * sum(mix.values()))}
if kind == "plasma":
    mix = {c: v * (1 - sum(ADD.values())) for c, v in mix.items()}; mix.update(ADD)
k_ = SV.key(); probes = list(k_["probe"]); idx = np.asarray(k_["idx"], dtype=np.int64)
lev = np.zeros(len(probes)); wsum = np.zeros(len(probes))
for c, w in mix.items():
    v = SA._col("mu", c).reindex(probes).to_numpy(dtype=float); ok = np.isfinite(v); lev[ok] += w * v[ok]; wsum[ok] += w
lev = np.where(wsum > 0, lev / np.maximum(wsum, 1e-9), np.nan)
lam = depth * g.lognormal(0, 1.0, len(lev)); lam *= depth / np.median(lam)
n = g.poisson(lam); n[~np.isfinite(lev)] = 0
if kind == "plasma": n[g.random(len(n)) > 1 / 3] = 0
kk = g.binomial(n, np.clip(np.nan_to_num(lev), 0, 1)); n = np.minimum(n, 255); kk = np.minimum(kk, n)
arr = np.zeros((SV.HG19_CPGS, 2), dtype=np.uint8); arr[idx - 1, 0] = kk; arr[idx - 1, 1] = n
bp = os.path.join(out, f"nw_{PART}.beta"); arr.tofile(bp)
res = dict(part=PART, kind=kind, depth=depth, truth=mix, sites_read=int((n > 0).sum()), readings={})
for wt in (False, True):
    bj = os.path.join(out, f"nw_{PART}_{int(wt)}.json")
    cmd = [sys.executable, os.path.join(CH, "MethylPhys_Interface", "run_sample.py"), "--dev-seq-beta", bp, "--dev-seq-values", "1", "--dev-allow-partial-genome",
           "--specimen", "plasma" if kind == "plasma" else "whole blood", "--id", f"nw{PART}{int(wt)}", "--out", os.path.join(out, f"nw_{PART}_{int(wt)}.html"), "--bundle", bj, "--no-info-sphere"] + (["--dev-seq-noise-weight"] if wt else [])
    p = subprocess.run(cmd, cwd=os.path.join(CH, "MethylPhys_Interface"), capture_output=True, text=True)
    B = json.load(open(bj)) if os.path.exists(bj) else {}
    res["readings"][str(wt)] = dict(rc=p.returncode, err=p.stderr[-400:] if p.returncode else "", shares={c["cell"]: c["share"] for c in (B.get("cell_shares") or {}).get("cells") or []},
                                    weighting=(B.get("cell_shares") or {}).get("noise_weighting"))
os.remove(bp); json.dump(res, open(os.path.join(out, f"seq_noise_weight_synthetic_p{PART}.json"), "w"), indent=1)
print(json.dumps({k: (v["rc"], v["weighting"]) for k, v in res["readings"].items()}))
