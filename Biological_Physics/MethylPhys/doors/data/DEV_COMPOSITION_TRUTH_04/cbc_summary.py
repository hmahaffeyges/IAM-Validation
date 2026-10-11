"""DEV-COMPOSITION-TRUTH-04: what the GSE167202 clinical differential gives per sample (GEO sample characteristics, gse167202_cbc.csv; metadata
only, no methylation value read). Writes cbc_summary.txt. Usage: python3 cbc_summary.py"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); G = pd.read_csv(os.path.join(HERE, "gse167202_cbc.csv"))
six = ["segmented_neutrophils", "lymphocyte_percent", "monocyte_percent", "eosinophils_percent", "basophils_percent", "immature_granulocytes_percent"]
three = G[["segmented_neutrophils", "lymphocyte_percent", "monocyte_percent"]].notna().all(1); full = G[six].notna().all(1); tot = G[six].sum(1)
n = G.segmented_neutrophils[three]
out = [f"samples {len(G)}", f"neutrophil, lymphocyte and monocyte % given: {int(three.sum())}", f"full six-part differential: {int(full.sum())}",
       f"six parts sum to 100 +/- 2: {int((full & tot.between(98, 102)).sum())}",
       f"segmented neutrophil % where given: median {n.median():.1f}, range {n.min():.1f}-{n.max():.1f}, 10th-90th percentile {n.quantile(.1):.1f}-{n.quantile(.9):.1f}",
       "by COVID status (n with neutrophil %): " + ", ".join(f"{k} {int(v)}" for k, v in G[three].covid_status.value_counts().items())]
open(os.path.join(HERE, "cbc_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
