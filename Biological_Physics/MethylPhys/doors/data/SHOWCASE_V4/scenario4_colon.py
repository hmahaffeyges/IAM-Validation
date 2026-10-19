#!/usr/bin/env python3
"""Showcase scenario 4 (written 2026-10-19 before any of its files is opened): healthy colon tissue read by IAM-Atlas v2.1's commissioned tissue
use, through its own scorer, not the one command (the one command has no tissue route; development/TEST_PLAN_TRACKS_2026-10-17.md, Part 7).
Samples: GSE151732, adenoma history "No", the first 3 by GSM accession (selected.json s4): GSM4589141, GSM4589144, GSM4589145; EPIC v1
(GPL23976). Each array: chain Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta, pinned methylprep), then, at the sealed composition-blind
sites of the colon tissue test (DEV-TISSUE-COLON-01 sealed_colon_tissue.json, sealed into the atlas commissioning as U2), the share of sites
within the sealed tolerance (0.05) of the atlas's colon-epithelium value (chain stage_atlas_u2.anchors and stage_atlas_commission.u2_share,
unchanged). The three together through stage_atlas_u2.score unchanged (its minimum is 3 arrays). The atlas's commissioned bar is 0.90.
A showcase reading, not a test: nothing is scored or sealed here; a reading below the bar is reported as it comes out.
Usage: python scenario4_colon.py <atlas_v21_summary.parquet> <idat dir> <out dir>"""
import os, sys, json, glob
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "../../../chain")); sys.path.insert(0, CH)
import numpy as np, pandas as pd
import stage_atlas_u2 as U2, stage_atlas_commission as A21
from stage_1_idat_calibration import calibrate_idat_to_beta
SAMPLES = json.load(open(os.path.join(H, "selected.json")))["s4"]
summary, idat, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
AN = U2.anchors(summary); a = AN["colon"]; tol = U2.SEAL["colon"]["tolerance"]
rows, betas = [], []
for g in SAMPLES:   # one at a time: the first array also installs methylprep's manifest, so it is not raced
    grn = glob.glob(os.path.join(idat, g + "_*_Grn.idat")); red = glob.glob(os.path.join(idat, g + "_*_Red.idat"))
    assert len(grn) == 1 and len(red) == 1, (g, grn, red)
    b, _ = calibrate_idat_to_beta(grn[0], red[0], verbose=False); b = b.iloc[:, 0] if hasattr(b, "columns") else b
    b = b.astype(float); b.index = b.index.astype(str); betas.append(b)
    v = b.reindex(a.index).values; k = np.isfinite(v) & np.isfinite(a.values)
    rows.append(dict(sample=g, base=os.path.basename(grn[0])[:-9], probes=int(b.notna().sum()), blind_sites=int(len(a)), blind_sites_measured=int(k.sum()),
                     share_within_tolerance=round(A21.u2_share(v, a.values, tol), 4), median_abs_diff=round(float(np.nanmedian(np.abs(v[k] - a.values[k]))), 4)))
    print(rows[-1], flush=True)
P = U2.score(AN, {"colon epithelium": betas}, ["colon"])["colon"]
R = dict(scenario=4, what="healthy colon tissue read by IAM-Atlas v2.1's commissioned colon tissue use (U2 scorer, unchanged)", series="GSE151732",
         rule="development/TEST_PLAN_TRACKS_2026-10-17.md Part 7 #4; selected.json s4", tolerance=tol, commissioned_bar=0.90, per_sample=rows,
         three_together=dict(share_epithelium=P["shares"].get("epithelium"), arrays=P["arrays"], verdict_by_the_scorer=P["verdict"], panel_cells_not_assessed=P["listed"]),
         note="A showcase reading, not a test. The scorer's own planted test: doors/data/DEV_ATLAS_V21_01/planted_u2_v21_summary.txt (PASSED).")
json.dump(R, open(os.path.join(out, "scenario4_colon.json"), "w"), indent=1); print(json.dumps(R["three_together"]))
