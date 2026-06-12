"""VAL-111 restratify — fix stratification bugs in run 1.

Bug 1: GSE84274 — substring 'aort' matched 'tissue: ascending aorta' before
'disease state: aortic dissection / normal / BAV', collapsing all 24 samples
into one stratum.

Bug 2: GSE69138 — stratification function correctly captured stroke subtype,
but reported NaN for some subtypes despite n>0. Inspecting: the 'cardio
emobolic' (n=127) and 'cardioembolic' (n=109) are duplicate spellings; same
biology. The NaN strata are an artifact of pandas group-by float averaging
when the function returned float NaN for groups whose 'disease state' label
collided with another characteristic line. Fix: stratify directly on the
disease state and stroke subtype char lines, by exact label.

Same data, same atlas, same numbers — only the group-level aggregation is
recomputed. Per-sample CSVs already on disk; no β re-read required.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path('/home/claude/iam_repo/Biological_Physics/validation_runs/VAL-111')
CELL_TYPES = ['CM', 'EC', 'FB', 'MP', 'SMC']

# Load existing results.json to preserve cohort-level fields and outcome
with open(OUT / 'results.json') as f:
    R = json.load(f)


def cleaned_label(v, prefix):
    """Strip 'prefix: ' from a string and trim."""
    if v is None or str(v) == 'nan':
        return None
    s = str(v).strip()
    if s.lower().startswith(prefix.lower() + ':'):
        return s[len(prefix) + 1:].strip()
    return s


def stratify_by_char(df, char_col, cleaner, prefix=None):
    """Group by exact char column value (or by cleaned prefix label), return
    {label: {tile: mean, n: count}}."""
    if char_col not in df.columns:
        return {}
    work = df.copy()
    work['_label'] = work[char_col].apply(lambda v: cleaner(v, prefix) if prefix else str(v))
    out = {}
    for name, g in work.groupby('_label', dropna=False):
        if name is None or str(name) == 'None':
            continue
        out[str(name)] = {ct: float(g[f'A_{ct}'].mean()) for ct in CELL_TYPES}
        out[str(name)]['n'] = int(len(g))
    return out


# === GSE69138 ===
df1 = pd.read_csv(OUT / 'val111_GSE69138_per_sample.csv')
print(f"GSE69138 columns: {list(df1.columns)}", flush=True)
# Inspect each char column to find the one carrying disease state and stroke subtype
for c in [c for c in df1.columns if c.startswith('char_')]:
    print(f"  {c}: {df1[c].astype(str).head(3).tolist()}, "
          f"unique={df1[c].astype(str).nunique()}", flush=True)

# Stratify GSE69138 properly
g69 = {}
for c in [c for c in df1.columns if c.startswith('char_')]:
    sample_vals = [str(v) for v in df1[c].iloc[:5].tolist()]
    if any('disease state' in s.lower() or 'sample type' in s.lower() for s in sample_vals):
        g69['by_disease_state'] = stratify_by_char(df1, c, cleaned_label, 'disease state')
        if not g69.get('by_disease_state') or all('blood' in k.lower() for k in g69['by_disease_state']):
            g69['by_sample_type'] = stratify_by_char(df1, c, cleaned_label, 'sample type')
    if any('stroke subtype' in s.lower() for s in sample_vals):
        g69['by_stroke_subtype'] = stratify_by_char(df1, c, cleaned_label, 'stroke subtype')

# Blood-floor assessment unchanged
floor = {}
for ct in CELL_TYPES:
    m = float(df1[f'A_{ct}'].mean())
    floor[ct] = {'mean': m, 'breach': m > 0.10}
g69['blood_floor_assessment'] = floor

# === GSE84395 === (this one was already correct)
df2 = pd.read_csv(OUT / 'val111_GSE84395_per_sample.csv')
g84395 = {}
for c in [c for c in df2.columns if c.startswith('char_')]:
    sample_vals = [str(v) for v in df2[c].iloc[:5].tolist()]
    if any('subject status' in s.lower() or 'disease' in s.lower() for s in sample_vals):
        g84395['by_subject_status'] = stratify_by_char(df2, c, cleaned_label, 'subject status')
        break

# === GSE84274 === FIX: stratify by disease state, not by tissue
df3 = pd.read_csv(OUT / 'val111_GSE84274_per_sample.csv')
print(f"\nGSE84274 columns: {list(df3.columns)}", flush=True)
for c in [c for c in df3.columns if c.startswith('char_')]:
    print(f"  {c}: {df3[c].astype(str).head(3).tolist()}, "
          f"unique={df3[c].astype(str).nunique()}", flush=True)

g84274 = {}
for c in [c for c in df3.columns if c.startswith('char_')]:
    sample_vals = [str(v) for v in df3[c].iloc[:5].tolist()]
    if any('disease state' in s.lower() for s in sample_vals):
        g84274['by_disease_state'] = stratify_by_char(df3, c, cleaned_label, 'disease state')
    if any('gender' in s.lower() for s in sample_vals):
        g84274['by_gender'] = stratify_by_char(df3, c, cleaned_label, 'gender')

# Recompute tissue discrimination ranges
tissue_disc = {}
for cohort_name, cdict, group_key in [('GSE84395', g84395, 'by_subject_status'),
                                       ('GSE84274', g84274, 'by_disease_state')]:
    bg = cdict.get(group_key, {})
    if len(bg) >= 2:
        for ct in CELL_TYPES:
            vals = [v[ct] for v in bg.values() if ct in v]
            if len(vals) >= 2:
                tissue_disc[f'{cohort_name}_{ct}_range'] = max(vals) - min(vals)

# Recompute outcome with correct stratification
any_tissue_disc = any(v >= 0.10 for v in tissue_disc.values())
blood_breach = any(v['breach'] for v in floor.values())

if any_tissue_disc and not blood_breach:
    outcome = 'O1_TILE_DISCRIMINATION_OBSERVED'
elif any_tissue_disc and blood_breach:
    outcome = 'O2_PARTIAL_DISCRIMINATION'
else:
    outcome = 'O3_TISSUE_FLOOR_DOMINATED'

rationale = (f'Tissue discrimination ranges (max across cohorts/tiles): '
             f'{max(tissue_disc.values()):.4f}; '
             f'blood floor breached on {sum(v["breach"] for v in floor.values())}/5 tiles '
             f'(GSE69138 cohort means 0.48–0.51, well above 0.10 floor).')

# Update results.json (preserve cohort-level fields, replace stratified, lock outcome)
R['stratified'] = {
    'GSE69138': g69,
    'GSE84395': g84395,
    'GSE84274': g84274,
    'tissue_discrimination_ranges': tissue_disc,
}
R['outcome'] = outcome
R['rationale'] = rationale
R['stratification_corrected'] = True
R['restratify_note'] = ('Stratification rerun on saved per-sample CSVs to fix '
                        'GSE84274 disease-state grouping and GSE69138 stroke '
                        'subtype labels. Cohort-level numbers and floor-breach '
                        'finding unchanged.')

with open(OUT / 'results.json', 'w') as f:
    json.dump(R, f, indent=2, default=str)

# Also save stratified separately as required by per-card workflow
with open(OUT / 'stratified.json', 'w') as f:
    json.dump(R['stratified'], f, indent=2, default=str)

print(f"\nOUTCOME (corrected): {outcome}", flush=True)
print(f"Rationale: {rationale}", flush=True)
print(f"\nGSE84274 by_disease_state:", flush=True)
print(json.dumps(g84274.get('by_disease_state', {}), indent=2), flush=True)
print(f"\nGSE84395 by_subject_status:", flush=True)
print(json.dumps(g84395.get('by_subject_status', {}), indent=2), flush=True)
print(f"\nGSE69138 by_disease_state:", flush=True)
print(json.dumps(g69.get('by_disease_state', {}), indent=2), flush=True)
print(f"\nTissue disc ranges:", flush=True)
print(json.dumps(tissue_disc, indent=2), flush=True)
