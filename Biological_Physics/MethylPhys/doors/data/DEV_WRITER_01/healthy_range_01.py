"""DEV-WRITER-01: the healthy range quoted in the note, from PROC-CHANNEL-01's 56 cell types (data/PROC_CHANNEL_01/rerun_2026-10-10_channel_cells.csv):
copy error min-max and E_hold = ln((1 - eps) / eps) min-max, in k_BT. Writes healthy_range_01_output.txt. Usage: python3 healthy_range_01.py"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); C = pd.read_csv(os.path.join(HERE, "../PROC_CHANNEL_01/rerun_2026-10-10_channel_cells.csv"))
e = C.copy_err; E = np.log((1 - e) / e)
out = [f"cell types {len(C)}", f"copy error {e.min():.4f}-{e.max():.4f}", f"E_hold {E.min():.3f}-{E.max():.3f} k_BT"]
open(os.path.join(HERE, "healthy_range_01_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
