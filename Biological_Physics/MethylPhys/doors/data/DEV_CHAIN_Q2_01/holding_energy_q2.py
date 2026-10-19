#!/usr/bin/env python3
"""The cells' holding energy on the commissioned counter (2026-10-19). Inputs (committed): shape_out/GSM*.json, chain Stage Q2's counts
of 153 healthy samples of the published sequencing atlas (GSE186458, 56 cell types; reproduced from a fresh clone, reproduce/ iama_line),
and loyfer_samples.csv (each sample's cell type). Held error = isolated unmethylated interior calls / interior calls on mostly-methylated
molecules, the chain's definition (Stage Q2 count_pat): an end CpG has one neighbor and cannot be isolated, so it is no opportunity.
Per cell type: opportunity-weighted held error over its samples; E_hold = ln((1-eps)/eps) kT; then the mean and SD over the 56 types,
eps0 = 1/(1+e^E_hold), H_ref = H(eps0), phi = E_hold / M (M = 20.94, CANON M_cell). Nothing is fitted. Writes holding_energy_q2.json.
Replaces the PROC-CHANNEL-01 values (3.41 kT, 0.032, 0.163, 0.2043 bits), whose counter divided by every CpG on a molecule, ends included,
so read about 0.75 of the per-opportunity rate (denominator_check_2026-10-19.json)."""
import os, json, glob, csv, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); M_CELL = 20.94
cell = {r["sample"]: r["cell"] for r in csv.DictReader(open(os.path.join(H, "loyfer_samples.csv")))}
agg = {}
for f in sorted(glob.glob(os.path.join(H, "shape_out", "GSM*.json"))):
    j = json.load(open(f)); c = cell[os.path.basename(f)[:-5]]; a = agg.setdefault(c, [0.0, 0.0, 0])
    a[0] += j["held_err"] * j["held_opp"]; a[1] += j["held_opp"]; a[2] += 1
eps = {c: a[0] / a[1] for c, a in agg.items()}; E = np.array([np.log((1 - e) / e) for e in eps.values()])
Hb = lambda e: float(-e * np.log2(e) - (1 - e) * np.log2(1 - e)); m, sd = float(E.mean()), float(E.std(ddof=1)); e0 = 1 / (1 + np.exp(m))
out = dict(cells=len(eps), samples=sum(a[2] for a in agg.values()), counter="chain Stage Q2 count_pat (interior opportunities)",
           E_hold_kT=round(m, 3), E_hold_sd_kT=round(sd, 3), E_hold_range_kT=[round(float(E.min()), 2), round(float(E.max()), 2)],
           E_hold_median_kT=round(float(np.median(E)), 3), E_hold_Landauer=round(float(m / np.log(2)), 2), eps0=round(float(e0), 5), H_ref_bits=round(Hb(e0), 4),
           phi=round(m / M_CELL, 4), phi_sd=round(sd / M_CELL, 4), T_equivalent_K=round(310.15 * M_CELL / m), held_error_range=[round(min(eps.values()), 4), round(max(eps.values()), 4)],
           one_floor=dict(rule="H(eps_cell)/H(eps0) per cell type; Normal 0.95-1.05", min=round(float(min(Hb(e) / Hb(e0) for e in eps.values())), 2),
                          max=round(float(max(Hb(e) / Hb(e0) for e in eps.values())), 2), median=round(float(np.median([Hb(e) / Hb(e0) for e in eps.values()])), 2),
                          inside_normal=int(sum(0.95 <= Hb(e) / Hb(e0) <= 1.05 for e in eps.values()))),
           per_cell={c: dict(held_error=round(e, 5), E_hold_kT=round(float(np.log((1 - e) / e)), 3), samples=agg[c][2]) for c, e in sorted(eps.items())},
           supersedes=dict(source="PROC-CHANNEL-01 (every CpG in the denominator)", E_hold_kT=3.41, eps0=0.032, phi=0.1628, H_ref_bits=0.2043, T_equivalent_K=1905))
json.dump(out, open(os.path.join(H, "holding_energy_q2.json"), "w"), indent=1); open(os.path.join(H, "holding_energy_q2.json"), "a").write("\n")
print({k: v for k, v in out.items() if k not in ("per_cell", "supersedes")})
