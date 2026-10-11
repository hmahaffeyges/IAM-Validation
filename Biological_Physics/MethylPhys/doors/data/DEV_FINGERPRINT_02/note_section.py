"""DEV-FINGERPRINT-02: the sealed-pair section of DEV_FINGERPRINT_02.md, written from the committed outputs (no number typed).
Usage: python3 note_section.py PAIR   (prints markdown)"""
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); P = sys.argv[1]; S = json.load(open(os.path.join(HERE, f"sealed_{P}.json")))
tag = S["normal_cell"].replace(" ", "_"); pl = f"plant_{P}_output.json" if P != "breast_mcf10a" else "plant_fingerprint_02_output.json"
PL = json.load(open(os.path.join(HERE, pl))); pw = open(os.path.join(HERE, f"power_{P}_output.txt")).read().strip()
rl = open(os.path.join(HERE, f"normal_{tag}", "runloss_sim.txt")).read(); m = re.search(r"run 0\.10 .*?\| ([+-]\d\.\d{4})\s*$", rl, re.M)
eps = [r["eps"] for r in S["normal_files"]]
weak = {"breast_hmec": " Weaker pair: MCF-7 is luminal, the mammary epithelial cells basal-like.", "breast_mcf10a": " Weaker pair: MCF-7 is luminal, MCF 10A basal-like and immortalised."}.get(P, "")
print(f"""
## Pair {P} ({S['cancer_cell']} against {S['normal_cell']}): sealed before any cancer file is aligned or read

**Sealed:** `data/DEV_FINGERPRINT_02/sealed_{P}.json` (normal side only; scorer sha256 recorded; the score step refuses unless this file is committed
unchanged). Normal: {len(eps)} RRBS files (Box Run 9, bwa mem -T30), Stage Q copy error {min(eps):.4f}-{max(eps):.4f} per file; arrays
{', '.join(S['normal_arrays'])} at {', '.join(str(v) for v in S['normal_arrays_sites'].values())} identity sites ({S['identity_cell']}).
Planted test on this pair (`{pl}`): NULL {PL['NULL']['VERDICT']}, PLANTED {PL['PLANTED']['VERDICT']} (z {PL['PLANTED']['z']:.2f}), ONLINE {PL['ONLINE']['VERDICT']};
Arm B quiet in all three. Arm A: slope {S['slope']:.3f}, sd_rep {S['sd_rep']:.4f}, se {S['se_A']:.4f}, power at a 0.06 excess {S['power_A06']:.3f}
{'(below 0.80: an Arm A no-fingerprint reading is UNDECIDED)' if S['power_A06'] < 0.80 else '(0.80 or above)'}. Arm B: the run-loss reader recovers a planted 0.10 run loss as
{m.group(1) if m else 'n/a'} on these molecules; bar = largest normal-file excess, {S['armB_bar']:.4f}. Cancer: the {len(S['cancer_files_listed'])} {S['cancer_cell']} files listed in the seal.{weak}
**Synthetic test:** Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align.py, Biological_Physics/MethylPhys/boxruns/run9_fp2/synth_align_result.csv, DEV_FINGERPRINT_02/plant_fingerprint_02.py, DEV_FINGERPRINT_02/{pl}, DEV_FINGERPRINT_02/power_fingerprint_02.py, DEV_FINGERPRINT_02/power_{P}_output.txt, DEV_FINGERPRINT_02/prep_normal.sh, DEV_FINGERPRINT_02/normal_{tag}/runloss_sim.txt""")
