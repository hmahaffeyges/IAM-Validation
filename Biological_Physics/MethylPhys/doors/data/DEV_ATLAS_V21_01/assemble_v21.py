"""Atlas v2.1 assembly (release F03, F04, F05), written 2026-10-16 after the fit and before any held-out array is read. Reads ONLY the fit's own
block files (outputs_v21.json); computes nothing new about cells.
 F04 atlas_v21_summary.parquet : every block_NNNNN.parquet stacked (cell, cpg, n_obs, mu, mu_sd, mu_lo, mu_hi, donor_sd, rhat, ess), sorted
 F03 output_manifest_v21.json  : every block file (parquet, draws, prior, warnings) with its size and SHA-256
 F05 convergence_v21.json      : share of measured cell-site pairs with R-hat >= 1.01 and ESS < 100, per block; the divergent blocks and their
                                 divergences; the flatness check (spread of mu across cells per site: v2's broken run had std about 0.005)
Usage: python assemble_v21.py <blocks folder> <out folder>"""
import os, sys, json, glob, hashlib, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
files = sorted(os.listdir(src)); assert len(files) == 2800, len(files)
json.dump({f: dict(bytes=os.path.getsize(os.path.join(src, f)), sha256=sha(os.path.join(src, f))) for f in files}, open(os.path.join(out, "output_manifest_v21.json"), "w"), indent=0)
P = sorted(glob.glob(os.path.join(src, "block_[0-9][0-9][0-9][0-9][0-9].parquet"))); assert len(P) == 700, len(P)
A = pd.concat([pd.read_parquet(p).assign(block=int(os.path.basename(p)[6:11])) for p in P], ignore_index=True).sort_values(["cell", "cpg"])
A.drop(columns="block").to_parquet(os.path.join(out, "atlas_v21_summary.parquet"), index=False)
m = A[A.n_obs > 0]
per = m.groupby("block").agg(pairs=("mu", "size"), rhat_ge_101=("rhat", lambda x: float((x >= 1.01).mean())), ess_lt_100=("ess", lambda x: float((x < 100).mean())))
W = {int(os.path.basename(p)[6:11]): json.load(open(p)) for p in glob.glob(os.path.join(src, "block_*_warnings.json"))}
div = {b: w["divergences"] for b, w in W.items() if w.get("divergences", 0) > 0}
wide = m.pivot_table(index="cpg", columns="cell", values="mu"); spread = wide.std(axis=1)
C = dict(cell_site_pairs_measured=int(len(m)), cells=int(A.cell.nunique()), sites=int(A.cpg.nunique()),
         share_rhat_ge_101=float((m.rhat >= 1.01).mean()), share_ess_lt_100=float((m.ess < 100).mean()),
         worst_block_share_rhat_ge_101=float(per.rhat_ge_101.max()), divergent_blocks=div, divergences_total=int(sum(div.values())),
         flatness=dict(median_spread_across_cells=float(spread.median()), p10=float(spread.quantile(0.1)), p90=float(spread.quantile(0.9)), v2_broken_run="about 0.005"))
json.dump(C, open(os.path.join(out, "convergence_v21.json"), "w"), indent=1); print(json.dumps(C))
