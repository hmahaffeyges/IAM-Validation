"""DEV-FINGERPRINT-02 sealed-pair section under rule 02b, written from the committed outputs only. Usage: python3 note_section_02b.py PAIR"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); P = sys.argv[1]; S = json.load(open(os.path.join(HERE, f"sealed_{P}.json")))
tag = S["normal_cell"].replace(" ", "_"); PL = json.load(open(os.path.join(HERE, f"plant02b_{P}_output.json"))); eps = [r["eps"] for r in S["normal_files"]]
weak = {"breast_hmec": " Weaker pair: MCF-7 is luminal, the mammary epithelial cells basal-like."}.get(P, "")
two = " The normal holds two donors (the donor null)." if P == "lung_bronchial" else " The normal is one donor; the donor term in the bar comes from the bronchial donor null."
cs = "; ".join(f"{c} {o['VERDICT']} (excess {min(o['cancer_excess'].values()):.4f} to {max(o['cancer_excess'].values()):.4f})" for c, o in PL.items())
print(f"""
## Pair {P} ({S['cancer_cell']} against {S['normal_cell']}), rule 02b: sealed before any cancer file is aligned or read

**Sealed:** `data/DEV_FINGERPRINT_02/sealed_{P}.json` (scorer `{os.environ.get('SCORER', 'score_fingerprint_02b.py')}`, sha256 in the seal; the score step refuses unless the seal is
committed unchanged). Normal: {len(eps)} RRBS files, Stage Q copy error {min(eps):.4f}-{max(eps):.4f}; arrays {', '.join(S['normal_arrays'])}.{two}
**Arm B decides.** Bar {S['armB_bar']:.4f} = max(normal-library excess {S['armB_bar_libraries']:.4f}, healthy donor-to-donor excess {S['armB_donor_excess']:.4f}).
Run-loss sensitivity on these molecules (planted 0.10): {S['armB_sensitivity_run010']:.4f}. Planted test of the sealed scorer (`plant02b_{P}_output.json`): {cs}.
FINGERPRINT if every one of the {len(S['cancer_files_listed'])} {S['cancer_cell']} files listed in the seal reads above the bar; no fingerprint if every file reads at or below
it; otherwise UNDECIDED. Arm A (Met-A against curve B) is computed and reported, not deciding.{weak}
**Synthetic test:** DEV_FINGERPRINT_02/plant_fingerprint_02b.py, DEV_FINGERPRINT_02/plant02b_{P}_output.json, DEV_FINGERPRINT_02/donor_null_02.py, DEV_FINGERPRINT_02/donor_null_02_output.json, DEV_FINGERPRINT_02/prep_normal.sh, DEV_FINGERPRINT_02/normal_{tag}/runloss_sim.txt, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align.py, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align_result.csv""")
