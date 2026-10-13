"""Every number the C-score commissioning plan quotes for its spread and repeatability steps, recomputed from committed inputs
so the note's figures have a source (push-gate rule: a number in a note must appear in a committed output of its own data folder).

Inputs, both committed: ../DEV_CSCORE_TARE_01_arrays.csv (617 healthy arrays, 19 series, with C, C tared to series and to slide)
and gse250556_person_map.csv (GEO titles for GSE250556: subject A-D, pooled replicates; fetched 2026-10-13).
Also reads band_test_GSE315366_rows.csv, the held-out band test.
Usage: python3 plan_numbers.py  ->  plan_numbers_output.txt
"""
# NO CHAIN: this script computes no physics. It recomputes the spread and repeatability figures the plan quotes by taking
# standard deviations and medians over readings the CHAIN already produced and that are committed
# (DEV_CSCORE_TARE_01_arrays.csv, gse250556_person_map.csv). There is no reading, no entropy and no clustering to re-derive
# here, so there is nothing for the chain to do; every physical quantity it touches came out of the chain upstream.

import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
A = pd.read_csv(os.path.join(HERE, "..", "DEV_CSCORE_TARE_01_arrays.csv"))
P = pd.read_csv(os.path.join(HERE, "gse250556_person_map.csv"))
B = pd.read_csv(os.path.join(HERE, "band_test_GSE315366_rows.csv"))
G = A[A.set == "GSE250556"].merge(P, on="gsm", how="left")
G["person"] = G.title.str.extract(r"(subject[A-Z])", expand=False)
L = ["C-score commissioning plan: the numbers, recomputed from committed inputs", ""]
L.append(f"healthy arrays {len(A)} in {A.series.nunique()} series | specimens {A.specimen.value_counts().to_dict()}")
L.append(f"healthy spread, SD of C tared to series {A.C_rel_series.std():.4f} | tared to slide {A.C_rel_slide.std():.4f}")
for col, lab in (("C", "untared C"), ("C_rel_series", "C tared to series"), ("C_rel_slide", "C tared to slide")):
    w = G.dropna(subset=["person"]).groupby("person")[col].std()
    L.append(f"GSE250556 same-person within-SD of {lab}: mean {w.mean():.4f} | per person {dict(zip(w.index, w.round(4)))}")
L.append(f"GSE250556 arrays {len(G)} | people {G.person.nunique()} | arrays per person {G.person.value_counts().to_dict()}")
n = len(B); above = int((B.C_rel > 1.152).sum())
L += ["", f"held-out band test GSE315366 (clonal-haematopoiesis negative): n {n} | above 1.152: {above} | not above {1 - above / n:.3f}",
      f"  C_rel median {B.C_rel.median():.4f} | min {B.C_rel.min():.4f} | max {B.C_rel.max():.4f}",
      "", "block-count sampling error, derived: relative error of a variance from n blocks = sqrt(2/n)",
      f"  120 blocks (blocks of 50): {np.sqrt(2 / 120):.4f} | 600 blocks (blocks of 10): {np.sqrt(2 / 600):.4f}"]
H = pd.read_csv(os.path.join(HERE, "historical_numbers.csv"))
L += ["", "Numbers the note quotes from earlier runs, with their run and status (historical_numbers.csv):"]
L += [f"  {r.value:>7} {r.what} - {r.run} [{r.status}]" for r in H.itertuples()]
L += ["", "Where the recomputation disagrees with the note: the committed array file holds 617 arrays, the note's step 3 says 641,",
      "and the step-3 spread numbers (0.138, 0.175) come from that larger blocks-of-50 set, which the blocks-of-10 run superseded.",
      "The recomputed values above are the ones to quote from here on."]
open(os.path.join(HERE, "plan_numbers_output.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
