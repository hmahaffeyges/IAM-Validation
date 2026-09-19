"""
Bridge Caggiano CelFiE TIM cardiac panel (genomic-region-indexed WGBS) to
HM450 array CpG IDs.

Caggiano TIM (`tim_matrix.txt`) is 1,581 rows × 19 cell types, indexed by
(chrom, start, end) WGBS regions of ~500bp. Each cell-type column has both
a methylated read count (`*_meth`) and total depth (`*_depth`).

To make this scoreable against array-based cardio cohorts, we identify
HM450 array CpGs whose hg19 coordinate falls within each Caggiano region
and aggregate the per-CpG β estimates.

Output: caggiano_tim_cpg_bridged.csv
  Columns: cpg_id, dendritic, endothelial, eosinophil, erythroblast,
           macrophage, monocyte, neutrophil, placenta, tcell, adipose,
           brain, fibroblast, heart, hepatocyte, lung, mammary,
           megakaryocyte, skeletal, small_intestine
  19 cell types. Each cg-indexed β = meth_count / depth for the parent region.

For atlas-quality, we also produce:
  caggiano_tim_INVENTORY.json with per-cell-type CpG counts, CHK-3.1C dedup
  status, source SHA-256.
"""
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path

OUT = Path('/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim')
TIM_PATH = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_2021/tim_matrix.txt'
MANIFEST = OUT / 'hm450_hg19_manifest.csv'

CELL_TYPES = ['dendritic', 'endothelial', 'eosinophil', 'erythroblast',
              'macrophage', 'monocyte', 'neutrophil', 'placenta', 'tcell',
              'adipose', 'brain', 'fibroblast', 'heart', 'hepatocyte', 'lung',
              'mammary', 'megakaryocyte', 'skeletal', 'small_intestine']


def main():
    print("=== Loading Caggiano CelFiE TIM ===", flush=True)
    tim = pd.read_csv(TIM_PATH, sep='\t')
    print(f"  Shape: {tim.shape}, regions: {len(tim)}", flush=True)
    # Compute β per region per cell type: meth / depth
    beta_data = {}
    for ct in CELL_TYPES:
        meth_col, depth_col = f'{ct}_meth', f'{ct}_depth'
        if meth_col not in tim.columns or depth_col not in tim.columns:
            print(f"  MISSING columns for {ct}, skipping", flush=True)
            continue
        depth = tim[depth_col].replace(0, np.nan)
        beta_data[ct] = tim[meth_col] / depth
    region_beta = pd.DataFrame(beta_data)
    region_beta['chrom'] = tim['chrom'].astype(str)
    region_beta['start'] = tim['start'].astype(int)
    region_beta['end'] = tim['end'].astype(int)

    # Quick sanity: distribution of β per cell type (should be in [0,1])
    print(f"  Per-cell-type β range:", flush=True)
    for ct in CELL_TYPES:
        if ct in region_beta.columns:
            s = region_beta[ct].dropna()
            print(f"    {ct}: n={len(s)}, range [{s.min():.3f}, {s.max():.3f}], mean {s.mean():.3f}", flush=True)

    print("\n=== Loading HM450 hg19 manifest ===", flush=True)
    manifest = pd.read_csv(MANIFEST)
    print(f"  Manifest: {len(manifest)} CpGs", flush=True)
    # Strip quotes from chr column if present
    manifest['chr'] = manifest['chr'].astype(str).str.strip('"')
    manifest['cpg_id'] = manifest['cpg_id'].astype(str).str.strip('"')

    print("\n=== Bridging regions to array CpGs ===", flush=True)
    # For each region, find CpGs whose pos is in [start, end]
    # Group manifest by chromosome for efficiency
    manifest_by_chr = {}
    for chrom, group in manifest.groupby('chr'):
        manifest_by_chr[chrom] = group[['cpg_id', 'pos']].sort_values('pos')

    cpg_records = []
    n_regions_with_cpgs = 0
    n_total_cpgs_mapped = 0
    for ri, row in region_beta.iterrows():
        chrom = row['chrom']
        if chrom not in manifest_by_chr:
            continue
        chr_manifest = manifest_by_chr[chrom]
        # CpGs in [start, end]
        in_region = chr_manifest[(chr_manifest['pos'] >= row['start']) &
                                  (chr_manifest['pos'] <= row['end'])]
        if len(in_region) == 0:
            continue
        n_regions_with_cpgs += 1
        for _, cpg_row in in_region.iterrows():
            rec = {'cpg_id': cpg_row['cpg_id']}
            for ct in CELL_TYPES:
                if ct in region_beta.columns:
                    rec[ct] = row[ct]
            cpg_records.append(rec)
            n_total_cpgs_mapped += 1
        if n_regions_with_cpgs % 200 == 0:
            print(f"    Mapped {n_regions_with_cpgs} regions, {n_total_cpgs_mapped} CpGs so far", flush=True)
    print(f"\n  Regions with ≥1 array CpG: {n_regions_with_cpgs}/{len(region_beta)}", flush=True)
    print(f"  Total CpG records: {len(cpg_records)}", flush=True)

    cpg_df = pd.DataFrame(cpg_records)
    print(f"  Pre-dedupe: {len(cpg_df)} rows, {cpg_df['cpg_id'].nunique()} unique CpGs", flush=True)

    # If a CpG falls in multiple regions (overlapping or adjacent), the same CpG appears
    # multiple times. Deduplicate by averaging β values per CpG per cell type.
    n_dups = len(cpg_df) - cpg_df['cpg_id'].nunique()
    print(f"  Duplicate rows (CpG in multiple regions): {n_dups}", flush=True)

    if n_dups > 0:
        # Aggregate by mean per CpG
        agg = cpg_df.groupby('cpg_id').mean(numeric_only=True).reset_index()
        print(f"  Post-dedupe: {len(agg)} unique CpGs", flush=True)
    else:
        agg = cpg_df

    # Set cpg_id as index, save CSV
    agg = agg.set_index('cpg_id')
    out_csv = OUT / 'caggiano_tim_cpg_bridged.csv'
    agg.to_csv(out_csv)
    print(f"  Wrote: {out_csv}", flush=True)
    print(f"  Final atlas: {agg.shape[0]} CpGs × {agg.shape[1]} cell types", flush=True)

    # Verify CHK-3.1C
    assert not agg.index.duplicated().any(), "CHK-3.1C FAIL: bridged atlas has duplicate CpG IDs"
    print(f"  CHK-3.1C passed: zero duplicate CpG rows", flush=True)

    # SHA-256
    def sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    inventory = {
        'atlas_id': 'Caggiano_CelFiE_TIM_cardiac_array_bridged_v1',
        'source': 'Caggiano C, Celona B, Garton F, Mefford J, Black BL, Henderson R, Lomen-Hoerth C, Dahl A, Zaitlen N. "Comprehensive cell type decomposition of circulating cell-free DNA with CelFiE." Nat Commun 2021;12:2717. doi:10.1038/s41467-021-22901-x',
        'source_atlas_path': TIM_PATH,
        'bridge_method': 'HM450 hg19 manifest CpG-in-region intersection; β = meth_count/depth per region; multi-region CpGs averaged',
        'manifest_source': 'IlluminaHumanMethylation450kanno.ilmn12.hg19 v0.6.1 (Bioconductor)',
        'n_source_regions': int(len(region_beta)),
        'n_regions_mapped_to_array_cpgs': int(n_regions_with_cpgs),
        'n_array_cpgs_total_pre_dedup': int(len(cpg_records)),
        'n_array_cpgs_unique_post_dedup': int(len(agg)),
        'n_cell_types': int(len(CELL_TYPES)),
        'cell_types': CELL_TYPES,
        'bridge_date': '2026-04-29',
        'sha256_bridged': sha256(out_csv),
        'sha256_source_tim': sha256(TIM_PATH),
        'chk_3_1c_passed': True,
        'cardio_relevance': 'heart_meth/heart_depth column = bulk heart tissue β reference; endothelial_meth = sorted vascular endothelial β reference. Both directly relevant for cardio Stage 2 cell-of-origin discrimination at array CpG resolution.',
    }
    with open(OUT / 'caggiano_tim_INVENTORY.json', 'w') as f:
        json.dump(inventory, f, indent=2)
    print(f"  Wrote: {OUT / 'caggiano_tim_INVENTORY.json'}", flush=True)


if __name__ == '__main__':
    main()
