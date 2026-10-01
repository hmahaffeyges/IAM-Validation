# Methow River steelhead RRBS — design of the public dataset (Gavery et al. 2018, G3, doi 10.1534/g3.118.200458; BioProject PRJNA325786)

Read from the paper's methods and the SRA run metadata (2026-10-01).

| item | value |
|---|---|
| fish | 20 adult males: 10 hatchery-origin, 10 natural-origin |
| tissues per fish | red blood cells (RBC) and sperm — both single cell types (the authors' stated reason for choosing them) |
| collection | adults returning to the Methow River, Feb–Apr 2014, hook and line (USFWS); held at Wells Hatchery (WH) or Winthrop NFH (WNFH) until spawning |
| hatchery fish | WH age 3 (6), WH age 4 (2), WNFH age 4 (2); all marked (adipose clip / coded-wire tag); reared to yearling smolts |
| natural-origin fish | age 4 (6), age 5 (4); unmarked = at least one generation in the wild |
| sequencing | RRBS, Illumina single-end 101 bp; 109 runs; 17–99 M reads per specimen (median 38 M); 117 GB |
| genetics | RAD-seq on 72 fish, 936 SNPs: no population-level differentiation between hatchery and natural origin |
| published result | 85 differentially methylated 100-bp regions in RBC and 108 in sperm (methylKit; ≥ 20× in ≥ 7 of 10 per group); ~22 % of reads multi-mapped (genome duplication) |
| run → fish map | `methow_run_map.csv` (library name = fish_tissue_origin_lane_part) |

**Confounds a fish biologist will raise, built into the test:**
1. **Age:** hatchery fish are 3–4, natural fish 4–5 — origin and age overlap only at age 4 (2 + 2 hatchery vs 6 natural). Report readings by age; test origin within age 4 as a check.
2. **Two hatcheries** (WH, WNFH) — report separately.
3. **Males only; one return year (2014)** — no claim beyond that.
4. **Freshwater history** differs by design (hatchery yearling smolts vs longer wild residence) — that is the treatment, not a nuisance.
5. **Genotype:** C→T SNPs read as unmethylated — mask sites polymorphic in the RAD/WGS data or showing bisulfite-independent T in either strand context.
6. **Duplicated genome:** uniquely mapped reads only.
