"""Plan 35.30: the whole-genome atlas's machine time, scaled from IAM-Atlas v2.1's measured block (block_measure_v21.json: block 0, 1,163
sites, 2,001 s on 8 cores, peak 15.6 GB), with v2.1's own per-site cost; the whole genome's observations per site are not yet known, so
this is a floor. Sites: hg19's CpGs (CANON chromosome ranges), either one per CpG or neighboring CpGs fitted together in fours (the block
rule in the build plan). Prices: the AWS sheet read by the author (cost_v21.txt, m7a.32xlarge spot $3.32-3.53 per hour, 128 cores).
Usage: python3 wg_cost.py OUT.txt"""
import sys, os, json
H = os.path.dirname(os.path.abspath(__file__)); B = json.load(open(os.path.join(H, "block_measure_v21.json")))
R = json.load(open(os.path.join(H, "..", "..", "..", "chain", "Runtime Matrices", "IAM_A_Positions", "hg19_cpg_chrom_ranges.json")))["ranges"]
cpg = sum(b - a + 1 for a, b in R.values()); core_s = B["wall_s"] * B["cores_per_process"] / 1163
L = [f"measured: {B['wall_s']} s on {B['cores_per_process']} cores for 1,163 sites = {core_s:.1f} core-seconds per site; peak {B['peak_rss_gb']} GB per block",
     f"hg19 CpGs in the chromosome ranges: {cpg:,}", "sites | core-hours | GitHub machine-hours (4 cores) | days on 20 GitHub machines | AWS spot dollars (one atlas) | with the two half-atlases"]
for lab, n in (("every CpG", cpg), ("CpGs in fours", cpg // 4)):
    ch = n * core_s / 3600; gh = ch / 4
    L.append(f"{lab} ({n:,}) | {ch:,.0f} | {gh:,.0f} | {gh / 20 / 24:.1f} | {ch / 128 * 3.32:,.0f}-{ch / 128 * 3.53:,.0f} | x2")
L.append(f"memory: a block peaked at {B['peak_rss_gb']} GB against a GitHub machine's 16 GB, so on GitHub a block must be smaller than v2.1's")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
