"""DEV-LINK-IAMA-METAA-02 outcome numbers: Stage Q's in-silico response on the vehicle molecules (insilico_vehicle_SRR25322252.csv) read at the
share of molecules decitabine left readable. Reports (i) the share readable when a scattered loss raises IAM-A to the 30 nM reading, (ii) IAM-A
a scattered loss would give at the 30 nM share read and at 0.60. Writes scattered_equivalent_02_output.txt. Usage: python3 scattered_equivalent_02.py"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V = pd.read_csv(os.path.join(HERE, "insilico_vehicle_SRR25322252.csv"))
D = pd.read_csv(os.path.join(HERE, "iama_dose_02_rows.csv")); d30 = D[D.treatment.str.contains("30 nM")]
a30, s30 = d30.IAMA_rel.mean(), d30.share_read.mean(); Vs = V.sort_values("share_read")
out = [f"30 nM measured: IAM-A_rel {a30:.4f}, share read {s30:.4f}",
       f"scattered loss reaching IAM-A_rel {a30:.4f}: share read {np.interp(a30, V.IAMA_rel, V.share_read):.4f}",
       f"scattered loss at share read {s30:.4f}: IAM-A_rel {np.interp(s30, Vs.share_read, Vs.IAMA_rel):.4f}",
       f"scattered loss at share read 0.60: IAM-A_rel {np.interp(0.60, Vs.share_read, Vs.IAMA_rel):.4f}"]
open(os.path.join(HERE, "scattered_equivalent_02_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
