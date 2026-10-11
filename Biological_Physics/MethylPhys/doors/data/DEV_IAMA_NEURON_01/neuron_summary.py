"""DEV-IAMA-NEURON-01: per-series counts from neuron_check.csv (GEO/SRA metadata, 2026-10-11; no sample value read). Per series: units, healthy
neuron units (cell NeuN+/neuron and diagnosis control/healthy/normal, or blank where the paper states healthy donors), donors, raw SRA GB of
those units, the series it shares >= 1 GEO contributor with (rule9_shared). Writes neuron_summary.txt. Usage: python3 neuron_summary.py"""
import os, re, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); N = pd.read_csv(os.path.join(HERE, "neuron_check.csv"))
neu = N.cell.astype(str).str.contains(r"(?i)neun\+|neuron|^neu", regex=True) & ~N.cell.astype(str).str.contains(r"(?i)neun-|non-neur", regex=True)
ok = N.diagnosis.astype(str).str.contains(r"(?i)control|healthy|normal|^nan$", regex=True)
out = []
for s, g in N.groupby("series"):
    h = g[neu.loc[g.index] & ok.loc[g.index]]
    out.append(f"{s}: units {len(g)}; healthy neuron units {len(h)}; donors {h.donor.nunique()}; raw GB {h.sra_GB.fillna(0).sum():.0f}; "
               f"lab {str(g.lab.iloc[0])[:40]}; shares contributors with {g.rule9_shared.iloc[0]}")
open(os.path.join(HERE, "neuron_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
