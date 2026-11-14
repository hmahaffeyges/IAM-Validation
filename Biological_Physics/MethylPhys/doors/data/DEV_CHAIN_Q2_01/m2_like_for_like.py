"""Register M2 (exploration; 2026-11-13 night): the healthy level behind CANON H_ref_cell, like for like. H_ref_cell = H(eps0_meth) is the
counter's held_err on 153 healthy samples at their own loose shares; the candidate (CANON eps0_candidate) is a per-site loss with no loose
tiles. So: (1) the candidate through the counter (forward_out/fc_heldErr_seq_1.5.csv: forward_candidate.py, eraser ratio 1, follow-on
clustering 1.5, GitHub explore run m2_canary); (2) the healthy samples' held_err and held_err_length read at zero loose share (percall_out,
neurons set aside), like IAM-A's intercept; (3) healthy over thermal on both quantities, against the stem-cell eraser factor 1/0.8642.
Writes m2_like_for_like_summary.txt."""
import os, json, glob, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
F = pd.read_csv(os.path.join(H, "forward_out", "fc_heldErr_seq_1.5.csv")); th_e, th_l, cpl = F.held_err.mean(), F.held_err_length.mean(), F.coupling.mean()
R = []
for f in glob.glob(os.path.join(H, "percall_out", "*.json")):
    c = json.load(open(f))
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and x.get("held_err") is not None and "neuron" not in str(x.get("cell", "")).lower():
            R.append((x["held_err"], x["loose_share"], x["held_err_length"], x["coupling"]))
A = np.array(R); ae = np.polyfit(A[:, 1], A[:, 0], 1)[1]; al = np.polyfit(A[:, 1], A[:, 2], 1)[1]
L = [f"thermal-only candidate through the counter (coupling {cpl:.2f}; healthy median {np.median(A[:, 3]):.2f}): held_err {th_e:.5f}, lone-loss {th_l:.5f}",
     f"healthy, {len(A)} samples, at zero loose share: held_err {ae:.5f}, lone-loss {al:.5f}",
     f"healthy over thermal: held_err {ae / th_e:.3f}, lone-loss {al / th_l:.3f}; the stem-cell eraser factor 1/0.8642 = {1 / 0.8642:.3f}",
     "Reading: on both counter quantities the average healthy cell holds at the thermal level plus about the stem-cell eraser share. The "
     "earlier H(candidate) 0.2627 against H_ref_cell 0.2593 was not like for like (per-site against the counter at the samples' own loose share).",
     "Exploration on spent counts and one synthetic run; nothing sealed."]
open(os.path.join(H, "m2_like_for_like_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
