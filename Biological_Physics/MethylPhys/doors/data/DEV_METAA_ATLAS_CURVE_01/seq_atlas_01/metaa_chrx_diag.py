"""Sorted-cell Met-A with and without the X chromosome's identity sites, per healthy sample, from the chain's own per-site tables
(NAME_metaa_sites.csv.gz, written by the one command with --dev-metaa-sites). Diagnostic, added after pass 5 was scored (the noise maps
showed donor halves of different sex in 16 cells); nothing scored. Usage: python3 metaa_chrx_diag.py LAND_DIR OUT.csv OUT.txt"""
import os, sys, glob, re, numpy as np, pandas as pd
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, CH); import stage_metaa_atlas as SA, pyarrow.parquet as pq
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe").chrom_hg19
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*_metaa_sites.csv.gz"), recursive=True)):
    t = pd.read_csv(f, usecols=["site", "H_measured", "H_expected"]); x = t.site.map(K).to_numpy() == "chrX"
    hm, he = t.H_measured.to_numpy(), t.H_expected.to_numpy(); ok = np.isfinite(hm) & np.isfinite(he)
    a = hm[ok].sum() / he[ok].sum(); b = hm[ok & ~x].sum() / he[ok & ~x].sum()
    rx = hm[ok & x].sum() / he[ok & x].sum() if (ok & x).any() else np.nan
    rows.append(dict(gsm=re.search(r"GSM\d+", f).group(0), sites=int(ok.sum()), chrX_sites=int((ok & x).sum()), A_all=round(a, 5), A_without_chrX=round(b, 5), chrX_ratio=round(rx, 4)))
D = pd.DataFrame(rows).drop_duplicates("gsm"); D.to_csv(sys.argv[2], index=False)
L = [f"{len(D)} healthy samples; Met-A median with chrX {D.A_all.median():.4f}, without {D.A_without_chrX.median():.4f}; largest change {(D.A_all - D.A_without_chrX).abs().max():.4f}",
     f"the chrX sites alone: median ratio {D.chrX_ratio.median():.3f}; samples above 2: {int((D.chrX_ratio > 2).sum())}; below 0.5: {int((D.chrX_ratio < 0.5).sum())}"]
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
