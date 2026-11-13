"""Both sides of the methionine lever (DEV-METHIONINE-01, exploration on the committed counts in out/): depleted over control for the marked
side's losses (Stage Q2 held_err_length), the unmarked side's stray marks (gain_err: a lone methylated call between two unmethylated ones on
molecules held unmarked) and the share of half-marked molecules (mol_mid_share). Writes score_sides_summary.txt."""
import os, json
H = os.path.dirname(os.path.abspath(__file__)); q = lambda g: json.load(open(os.path.join(H, "out", g + "_q2.json")))
L = ["Both sides, depleted over control (held = marked-side losses; gain = unmarked-side stray marks; mid = half-marked molecules)"]
for line, c, d in (("MZGC1", "DRR322659", "DRR322660"), ("MZGC3", "DRR322663", "DRR322664")):
    C, X = q(c), q(d)
    L.append(f"{line}: held {X['held_err_length'] / C['held_err_length']:.3f}, gain {X['gain_err'] / C['gain_err']:.3f}, mid {X['mol_mid_share'] / C['mol_mid_share']:.3f}")
open(os.path.join(H, "score_sides_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
