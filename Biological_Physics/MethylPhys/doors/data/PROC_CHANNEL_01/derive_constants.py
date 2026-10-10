"""PROC-CHANNEL-01 -> CANON: holding energy, phi and eps0 from the per-cell genome-wide table.
channel_cells_genomewide.csv = channel.py's channel_cells.csv (56 cell types, 153 samples, 399 windows, 2026-09-30) with
E = ln((1-eps)/eps) [kT] and phi = E/M (M = 20.94) added, as on 2026-09-30 11:27. Run: python3 derive_constants.py"""
import json, math, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); M = 20.94
C = pd.read_csv(os.path.join(HERE, "channel_cells_genomewide.csv"))
for col in ("copy_err", "denovo"):
    E = np.log((1 - C[col]) / C[col]); assert np.allclose(E, C["E_" + col]) and np.allclose(E / M, C["phi_" + col])
    print(f"{col:9s} eps {C[col].min():.3f}-{C[col].max():.3f} | E_hold {E.mean():.2f} ± {E.std(ddof=1):.2f} ({E.min():.2f}-{E.max():.2f}) kT | "
          f"phi {(E / M).mean():.3f} ± {(E / M).std(ddof=1):.3f} ({(E / M).min():.3f}-{(E / M).max():.3f})")
E = C.E_copy_err.mean(); eps0 = 1 / (1 + math.exp(E))
print(f"E_hold_meth {E:.4f} -> {E:.2f} | phi {E / M:.4f} | eps0_meth = 1/(1+e^E) = {eps0:.5f} -> {eps0:.3f}")
canon = json.load(open(os.path.join(HERE, "../../../../../CANON/iam_canon.json")))["constants"]
for k, v in (("E_hold_meth", round(E, 2)), ("phi", round(round(E, 2) / M, 4)), ("eps0_meth", round(eps0, 3))):   # CANON: phi = E_hold_meth / M_cell
    print(f"  CANON {k} {canon[k]['value']} | derived {v} | {'match' if abs(canon[k]['value'] - v) < 1e-9 else 'DIFFERS'}")
