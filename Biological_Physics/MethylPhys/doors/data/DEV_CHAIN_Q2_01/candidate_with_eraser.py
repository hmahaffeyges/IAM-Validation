"""The candidate with the eraser's measured share of the loss added, compared like for like (PHY-083; exploration; found after the forward
model, so not a sealed test). The candidate (CANON eps0_candidate) is the loss of a mark held against thermal kicks alone. The eraser (TET)
adds loss on top: in five eraser-removal units (DEV_ERASE_01/recount_01/recount_units.csv, commissioned counter; ES and neural stem cells),
the held error with the eraser removed is a fraction r of the control's. A cell with its eraser is then predicted at candidate / r. Two
comparisons, each on its own scale: (a) the candidate's counter readings at the healthy clustering (forward_out, forward model on GitHub)
divided by r, against the healthy readings (lone-loss 0.04526, CANON eps_healthy_pool, 95 % 0.04375-0.04657; per-call 0.05048, 0.04887-
0.05208); (b) the per-site loss, candidate / r, against the per-site loss the healthy readings imply. Writes candidate_with_eraser_summary.txt."""
import os, glob, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
E = float(json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]["eps0_candidate"]["value"])
U = pd.read_csv(os.path.join(H, "..", "DEV_ERASE_01", "recount_01", "recount_units.csv")); r = U.ratio_recount.to_numpy()
F = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(H, "forward_out", "*.csv"))]); F = F[F["mode"].isin(["ind", "seq"])].groupby(["mode", "param"]).mean(numeric_only=True).sort_values("coupling")
lone, pc = np.interp(1.46, F.coupling, F.held_err_length), np.interp(1.46, F.coupling, F.held_e_call)
rm, lo, hi = float(np.median(r)), float(np.sort(r)[1]), float(np.sort(r)[3])   # median and the middle three units
L = [f"eraser removed / control, five units: {', '.join(f'{x:.4f}' for x in r)}; median {rm:.4f}; middle three {lo:.4f}-{hi:.4f}",
     f"(a) candidate through the counter at the healthy clustering, with the eraser: lone-loss {lone / rm:.5f} ({lone / hi:.5f}-{lone / lo:.5f}) against healthy 0.04526 (0.04375-0.04657)",
     f"    per-call {pc / rm:.5f} ({pc / hi:.5f}-{pc / lo:.5f}) against healthy 0.05048 (0.04887-0.05208)",
     f"    without the eraser: lone-loss {lone:.5f}, per-call {pc:.5f}",
     f"(b) per-site loss with the eraser {E / rm:.5f} ({E / hi:.5f}-{E / lo:.5f}); the healthy readings imply {E * 0.04526 / lone:.5f} (lone-loss) and {E * 0.05048 / pc:.5f} (per-call)"]
open(os.path.join(H, "candidate_with_eraser_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
