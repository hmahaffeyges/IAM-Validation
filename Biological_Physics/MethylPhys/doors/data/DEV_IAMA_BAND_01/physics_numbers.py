"""The physics numbers quoted in DEV_IAMA_BAND_01.md (2026-11-20 section), computed so every number in the record has a committed source.
Derived chain (register A1-A4, book p6_08 ch:iama): M = dG_ATP / (R T); push x = 2 M (two ATP per restore, A2); held/lost = (x/2) / (1 - (1 - e^-x)/x)
(the slow-write rule, identical to the driven passage, A4); eps0 = 1 / (1 + held/lost). The counting uncertainty of A is the test library's own value
(GSM5652214, one-button bundle iam_a.counting.A_sd = 0.00143)."""
import json, math
R, T, dG = 8.314, 310.15, 54000.0
M = dG / (R * T); x = 2 * M; odds = (x / 2) / (1 - (1 - math.exp(-x)) / x); eps0 = 1 / (1 + odds)
out = dict(R_J_per_mol_K=R, T_K=T, dG_ATP_J_per_mol=dG, M_kicks=round(M, 2), push_x_kicks=round(x, 2), held_over_lost=round(odds, 2), eps0=round(eps0, 5),
           counting_sd_of_A_test_library=round(0.00143, 4), counting_sd_source="GSM5652214 one-button bundle iam_a.counting.A_sd (0.00143)")
json.dump(out, open("physics_numbers.json", "w"), indent=1); print(json.dumps(out))
