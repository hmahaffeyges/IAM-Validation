"""Sample labels for the A-score database (CANON/ascore_labels/<test>.csv: gsm, series, condition, cancer_type, stage), built from each test's
committed sample list, so CANON/bundle_rows.py can carry them into the rows (author 2026-11-03: every score kept with its cancer type and
severity, for our own look at where healthy and each kind of cancer sit; nothing is declared from it). Nothing is typed per sample: each
label is a committed column or its committed GEO record. Stage is "not recorded" wherever the source records none; it is never inferred.
Usage: python3 CANON/ascore_labels.py"""
import os, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); D = os.path.join(ROOT, "Biological_Physics", "MethylPhys", "doors", "data")
OUT = os.path.join(ROOT, "CANON", "ascore_labels"); os.makedirs(OUT, exist_ok=True)
def rd(p): d = pd.read_csv(os.path.join(D, p)); d["gsm"] = d.gsm.astype(str).str.strip(); return d
def put(name, d):
    d = d[["gsm", "series", "condition", "cancer_type", "stage"]]; d.to_csv(os.path.join(OUT, name + ".csv"), index=False); print(name, len(d), dict(d.condition.value_counts()))
S = rd("DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/healthy_fullchain7_samples.csv")
put("healthy_pass7", S.assign(series=S.gse, condition=S.genotype, cancer_type="", stage=""))
S = rd("DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/selfree_samples.csv"); put("selfree", S.assign(series=S.gse, condition="healthy", cancer_type="", stage=""))
S = rd("DEV_ATLAS_V21_01/seq_atlas/outside_data/blueprint_healthy_test_samples.csv"); put("blueprint", S.assign(series="BLUEPRINT", condition="healthy", cancer_type="", stage=""))
S = rd("DEV_ONE_DRAW_01/plasma_seq_02/plasma_seq_02_samples.csv"); put("plasma", S.assign(series=S.gse, condition=S.genotype, cancer_type="", stage=""))
S = rd("DEV_ALL_INFO_01/full_depth/hopkins_fullchain_samples.csv"); T = rd("DEV_ALL_INFO_01/full_depth/hopkins_geo_traits.csv").set_index("gsm")
COND = {"leukemia": "cancer", "knockout": "cell line (UHRF1 knockout)", "control line": "cell line (control)", "cord-blood B cells": "cord-blood B cells"}
cy = S.gsm.map(T.cytogenetics).fillna("").replace("N/A", "")
put("hopkins", S.assign(series=S.gse, condition=S.genotype.map(COND).fillna(S.genotype),
                       cancer_type=["pediatric pre-B acute lymphoblastic leukemia" + (f", {c}" if c else "") if g == "leukemia" else "" for g, c in zip(S.genotype, cy)],
                       stage=["not recorded" if g == "leukemia" else "" for g in S.genotype]))
