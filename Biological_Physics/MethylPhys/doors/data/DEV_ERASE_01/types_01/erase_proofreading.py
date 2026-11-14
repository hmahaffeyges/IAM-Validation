"""Register A18 (exploration on spent counts; 2026-11-14). Is the eraser the cell's proofreading, a Hopfield/Ninio discard step that removes
wrongly written marks? Prediction written first (6:57 AM Pacific): if it is, with the writer's discrimination D = 47.4-80, stray marks on the
unmarked side (chain Stage Q2 gain_err) rise in TET knockouts by about 1 + D (at least 10 times allowing for the meter's own false marks);
if the write is one selection step with no proofreading by the eraser (register A15), knockout over wild type stays within 0.7-1.5. The
ratio decides. Counts: the committed knockout count files (PHY-085 blood progenitors, PHY-086 P4 mature B and P5 iNKT, PHY-090 LSK).
Writes erase_proofreading_summary.txt."""
import os, glob, json, statistics as S
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(H)
def g(gsm): return json.load(open(glob.glob(os.path.join(D, "**", f"{gsm}*_count.json"), recursive=True)[0]))["gain_err"]
U = [("blood progenitors LSK, TET2 (PHY-085)", ["GSM7111686"], ["GSM7111688"]),
     ("blood progenitors Lin-neg, TET2 (PHY-085)", ["GSM7111689"], ["GSM7111691"]),
     ("LSK, TET2+TET3 (PHY-090)", ["GSM1866966", "GSM1866967", "GSM1866968"], ["GSM1866969", "GSM1866970", "GSM1866971"]),
     ("young iNKT, TET2+TET3 (PHY-086 P5)", ["GSM1855585", "GSM1855586"], ["GSM1855587", "GSM1855588"]),
     ("mature B, TET2+TET3 (PHY-086 P4)", ["GSM4907469", "GSM4907470"], ["GSM4907471", "GSM4907472"])]
L, R = [], []
for name, w, k in U:
    a, b = S.mean(g(x) for x in w), S.mean(g(x) for x in k); R.append(b / a)
    L.append(f"{name:42} wild type {a:.5f}  knockout {b:.5f}  knockout/wild type {b / a:.3f}")
v = "S (one selection step; the eraser is not proofreading the unmarked side)" if all(0.7 <= r <= 1.5 for r in R) else \
    "P (proofreading)" if all(r >= 10 for r in R) else "neither branch cleanly"
L += [f"range {min(R):.3f}-{max(R):.3f}: branch {v}",
      "Exploration on spent counts; the 0-30 % rise suggests the erasers also remove a small share of stray marks."]
open(os.path.join(H, "erase_proofreading_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
