# iamatlas_celltype_markers v0_1 — SUPERSEDED 2026-05-31

Superseded by **`iamatlas_celltype_markers_v0_2.json`** in this folder.

## What changed

v0_1 contained a `selection_logic_source` field with the value:
> `"val_093.py identify_marker_cpgs adapted from per-Loyfer-tile to per-IAMAtlas-cell-type"`

This was confusing — the field name + the appearance of "Loyfer" in the value
read like the markers were sourced FROM the Loyfer atlas. They were not.

**The actual data in v0_1 was already fully IAMAtlas REBUILD.** The `source_atlas`,
`source_sha256`, and `selection_criterion` fields all correctly anchored the data
to IAMAtlas. The only issue was that one provenance field carried a confusing
historical note about where the algorithm pattern was first implemented (in
val_093.py, which originally ran on Loyfer tiles).

## What v0_2 does

- Renames `selection_logic_source` → `algorithm_provenance`
- Reworded the value to make it unambiguously clear: "algorithm borrowed from
  val_093.py; APPLIED TO IAMAtlas REBUILD per-celltype posteriors"
- All marker CpG lists, H_min values, celltype-to-class mappings UNCHANGED

## Data identity check

Both files contain the same 115 cell types × 100 markers (115 × 100 = 11,500
CpG references). The marker data is bit-identical. Only metadata changed.

## Action

Use v0_2 going forward. v0_1 is kept for audit only.

v0_1 sha256: a56576cd5a7b2219d22d9a7a6efccd141a43c6d5fe4f5eb1d81e7375e1061ddc
v0_2 sha256: 46ea5be1db377f2b8773a02418a7f481a191630e0fa833d3294eab1fd19c47bd
