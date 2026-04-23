#!/usr/bin/env python3
"""
VAL-051 Step 1 — Deterministic 80/20 stratified split of AIBL.

Stratification: disease status × sex (6 cells: HC×F, HC×M, MCI×F, MCI×M, AD×F, AD×M)
Seed: 42
Output: val051_split_map.json — {gsm: 'train'|'holdout'}

NO β-VALUES TOUCHED IN THIS SCRIPT. This is pure metadata partitioning.
"""
import json, random, time

SEED = 42
TRAIN_FRAC = 0.80

manifest = json.load(open('aibl_manifest.json'))
# QC filter identical to VAL-050
joined = []
for s in manifest:
    if not s.get('disease status'): continue
    if not s.get('gender'): continue
    joined.append(s)

print(f"Samples post-QC: {len(joined)}")

# Stratify by (status, sex)
strata = {}
for s in joined:
    key = (s['disease status'], s['gender'])
    strata.setdefault(key, []).append(s['gsm'])

rng = random.Random(SEED)
split_map = {}
for key, gsms in sorted(strata.items()):
    gsms_shuffled = sorted(gsms)
    rng.shuffle(gsms_shuffled)
    n_train = int(len(gsms_shuffled) * TRAIN_FRAC)
    for g in gsms_shuffled[:n_train]: split_map[g] = 'train'
    for g in gsms_shuffled[n_train:]: split_map[g] = 'holdout'
    print(f"  {key[0]:<30} {key[1]:<8} n={len(gsms_shuffled):>3}  train={n_train:>3}  holdout={len(gsms_shuffled)-n_train:>3}")

# Summary counts
from collections import Counter
for split in ['train','holdout']:
    ad=mci=hc=0; fem=male=0
    for s in joined:
        if split_map[s['gsm']] != split: continue
        if s['disease status'] == "Alzheimer's disease": ad += 1
        elif s['disease status'] == "Mild Cognitive Impairment": mci += 1
        elif s['disease status'] == 'healthy control': hc += 1
        if s['gender'] == 'Female': fem += 1
        elif s['gender'] == 'Male': male += 1
    print(f"\n{split.upper():<8} n={ad+mci+hc}: AD={ad} MCI={mci} HC={hc}  F={fem} M={male}")

with open('val051_split_map.json','w') as f:
    json.dump({
        'seed': SEED,
        'train_frac': TRAIN_FRAC,
        'stratification': 'disease_status × gender',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'split': split_map,
    }, f, indent=2)

print(f"\nSplit map written: val051_split_map.json ({len(split_map)} samples)")
