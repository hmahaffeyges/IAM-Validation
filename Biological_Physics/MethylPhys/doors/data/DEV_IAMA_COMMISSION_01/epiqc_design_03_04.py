"""DEV-IAMA-COMMISSION-01 synthetic tests 3 and 4, metadata only (no read file opened). Input: epiqc_runs.csv (NCBI SRA runinfo for
SRR13050956-SRR13051274, the run range printed in the EpiQC paper's data statement, Foox et al. 2021 Genome Biol 22:332; library name = run alias
KIT_LINE_LAB_REP_SEQ). Per library (kit, line, lab, rep; lanes and sequencers summed): sequenced bases, depth x = bases / 3.1e9.
Test 3, depth: the Loyfer file (GSM5652313; raw reads are in EGA, not SRA) carries 738,262,569 CpG calls on 319,081,751 molecules over hg19's
28,217,448 CpGs: mean CpG coverage 26.16x (measured from the .pat, 2026-10-11). Synthetic test 1 put counting noise at <= 0.0071 at 1 % of that
file, a CpG coverage of 0.26x. Each EpiQC library's CpG coverage is taken, conservatively, as half its raw sequencing depth (an assumed allowance
for bisulfite mapping loss and duplicates, stated here, not measured). Bar (written before running): every library >= 0.26x.
Test 4, conversion: Stage Q0's limit 0.98 (C1) against each kit's non-CpG conversion as printed in the paper (Results: near 100 % for all
libraries except TruSeq ~2 % CHG / ~1 % CHH methylation and MethylSeq ~0.75 % CHH). Reported, not decided: the per-library value is read by Q0
from each alignment. Writes epiqc_design_03_04_output.csv and epiqc_design_03_04_summary.txt. Usage: python3 epiqc_design_03_04.py"""
import os, sys, re, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); E = pd.read_csv(os.path.join(HERE, "epiqc_runs.csv"))
m = E.alias.str.extract(r"^(?P<kit>[A-Za-z]+)_(?P<line>HG00\d)_(?P<lab>LAB\d+)_(?P<rep>REP\d+)_(?P<seq>[A-Z]+)"); E = pd.concat([E, m], axis=1)
WGS = {"TruSeq", "MethylSeq", "SPLAT", "EMSeq", "TrueMethylBS"}       # whole-genome 5mC libraries Stage Q can read (oxBS and capture/ATAC excluded)
L = E[E.kit.isin(WGS)].groupby(["kit", "lab", "line", "rep"]).agg(bases=("bases", "sum"), runs=("Run", "size"), readlen=("avgLength", "median")).reset_index()
FLOOR = 0.01 * 738_262_569 / 28_217_448                       # CpG coverage at 1 % of the Loyfer file
L["depth_x"] = L.bases / 3.1e9; L["cpg_cov_assumed"] = L.depth_x / 2
L["test3_pass"] = L.cpg_cov_assumed >= FLOOR
CONV = {"TruSeq": "CHG ~2 %, CHH ~1 % methylation reported (conversion ~0.98-0.99)", "MethylSeq": "CHH ~0.75 % (conversion ~0.99)",
        "SPLAT": "near 100 %", "EMSeq": "near 100 %", "TrueMethylBS": "near 100 %"}
L["conversion_paper"] = L.kit.map(CONV); L.to_csv(os.path.join(HERE, "epiqc_design_03_04_output.csv"), index=False)
out = [f"whole-genome libraries {len(L)} ({', '.join(f'{k} {v}' for k, v in L.groupby(['kit', 'lab']).size().items())})"]
for (k, lab), g in L.groupby(["kit", "lab"]):
    reps = g.groupby("line").rep.nunique()
    out.append(f"{k} {lab}: lines {g.line.nunique()}, lines with 2 reps {int((reps >= 2).sum())}, depth {g.depth_x.min():.1f}-{g.depth_x.max():.1f}x, "
               f"CpG coverage assumed {g.cpg_cov_assumed.min():.1f}-{g.cpg_cov_assumed.max():.1f}x (floor {FLOOR:.2f}x), test 3 pass {int(g.test3_pass.sum())}/{len(g)}")
out.append("SYNTHETIC TEST 3 " + ("PASSED" if L.test3_pass.all() else "FAILED") + " (every whole-genome library above the counting-floor depth)")
out.append("TEST 4 (reported): TruSeq sits at the 0.98 limit by the paper's non-CpG figures; C1 must be decided on identical DNA (copy error against conversion across kits) before C2-C5")
open(os.path.join(HERE, "epiqc_design_03_04_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
