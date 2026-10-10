"""DEV-WRITER-01 scoring by the rule sealed 2026-10-10 15:42 PDT (doors/DEV_WRITER_01.md). Values as printed in the papers:
Adam et al. 2023 (Nucleic Acids Res 51:6622): full-length murine DNMT1, competitive rates, HM/UM "80-fold on average" (abstract; Fig. 2B 87).
Yokochi & Robertson 2002 (J Biol Chem 277:11735), Table II: full-length human DNMT1, steady-state k_cat/K_M^CG 19.9 (hemi) and 0.42 (unmethylated).
Excluded by the rule: Bashtrykov et al. 2012 (Chem Biol 19:572) 30-mer, ~10-fold at ~1 uM DNA, above DNMT1's K_M (0.089-0.36 uM, Yokochi
Table II), so a k_cat ratio, not k_cat/K_M; its 40-mer 'at least 60-fold' is a lower bound only. Usage: python3 score_writer_01.py"""
import math, statistics as st
Q = {"Adam 2023 (abstract average)": 80.0, "Yokochi 2002 Table II": 19.9 / 0.42}
for alt in (80.0, 87.0):
    v = dict(Q); v["Adam 2023 (abstract average)"] = alt; m = st.median(v.values())
    band = "consistent with Model A" if 15 <= m <= 60 else ("below: partners add discrimination" if m < 15 else "above: cells lose more than the writer's errors")
    print(f"Adam value {alt:.0f}: D = {[round(x, 1) for x in v.values()]}, median {m:.1f}, E_hold = ln D = {math.log(m):.2f} k_BT, "
          f"eps = {1 / (1 + m):.4f} | cells 3.41 k_BT, eps 0.032 | {band}")
