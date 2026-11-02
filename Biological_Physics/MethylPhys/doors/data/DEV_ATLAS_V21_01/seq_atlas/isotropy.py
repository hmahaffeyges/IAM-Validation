"""Isotropy of the IAM Sequencing Atlas (checklist D14): each autosome's mean of the two noise maps, per cell, against its own standard
error (the map's spread there over the square root of its sites); a healthy split has no chromosome standing apart. From the noise
columns built by the one command (run_sample.py --build seq-noise-cols); the sex chromosomes and poorly mapping probes left out
(stage_seq_atlas.sky_mask_sites). Usage: python3 isotropy.py NOISE_COLS_DIR OUT.csv OUT.txt"""
import os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, CH); import stage_metaa_atlas as SA, stage_seq_atlas as Q
D = sys.argv[1]; ids = pq.read_table(os.path.join(SA.ATLAS, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe").chrom_hg19.reindex(ids).to_numpy()
auto = [f"chr{i}" for i in range(1, 23)]; keep = ~Q.sky_mask_sites(ids) & np.isin(K, auto); M, _ = Q.noise_maps(D); rows = []
for c, (m1, m2) in M.items():
    for name, v in (("map1 donor halves", m1), ("map2 read halves", m2)):
        for ch in auto:
            x = v[keep & (K == ch)]; x = x[np.isfinite(x)]
            if len(x) < 100: continue
            rows.append(dict(cell=c, map=name, chrom=ch, sites=len(x), mean=float(x.mean()), z=float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))))
R = pd.DataFrame(rows); R["z_from_cell"] = R.groupby(["cell", "map"]).z.transform(lambda z: z - z.median()); R.round(6).to_csv(sys.argv[2], index=False)
L = []
for name, g in R.groupby("map"):
    L.append(f"{name}: {g.cell.nunique()} cells x {g.chrom.nunique()} autosomes; |z from the cell's own median| above 4: {int((g.z_from_cell.abs() > 4).sum())} of {len(g)} "
             f"({(g.z_from_cell.abs() > 4).mean():.4f}); largest {g.z_from_cell.abs().max():.2f} ({g.loc[g.z_from_cell.abs().idxmax(), 'cell']}, {g.loc[g.z_from_cell.abs().idxmax(), 'chrom']})")
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
