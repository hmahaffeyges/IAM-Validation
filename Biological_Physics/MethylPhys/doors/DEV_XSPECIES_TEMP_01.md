# DEV-XSPECIES-TEMP-01 — does the copy error of one tissue rise with body temperature across species? (written 2026-10-09, before any read)

**Status labels.** The prediction is a PREDICTION from IAM (the kT ln 2 floor); its size is CALCULATED under a stated ASSUMPTION; the
reading method is DEVELOPMENT; nothing is fitted. All rules below are fixed before any read is processed.

**Data.** Klughammer et al., 580 animal species, RRBS (single-end 51 bp, HiSeq 3000), GEO GSE195869 / SRA PRJNA802599, one laboratory,
one protocol (all 3,023 runs already in S3). Tissues used: liver and heart (in most mammals and birds). Body temperature: AnAge
`Temperature (K)` (genomics.senescence.info, downloaded 2026-10-09): **45 mammal species** with liver/heart runs, 307.75–312.95 K. AnAge
has no bird body temperatures.

**Prediction (from IAM).** A methylation mark is held against thermal kicks with energy E; IAM reads it as E/kT = ln((1−ε)/ε). If the
absolute energy of maintenance is the same for one tissue across mammals (ASSUMPTION), a warmer body gives a smaller E/kT and a higher
copy error ε. \calculated At healthy ε ≈ 0.0385 (E/kT ≈ 3.22 at 310 K), d ln ε / dT = 3.22 × 310 / 310² ≈ **+1.0 % per K**, so about +5 %
across the 5.2 K mammal range. If cells adapt their maintenance to their temperature, the effect is absent: the test can fail.

**Reading (reference-free, one rule for every species).** Reads are grouped by their fully converted sequence (C→T in silico), which
is the same for every read of one fragment whatever its methylation; a position is a CpG in that fragment if any read of the group has
C followed by G there. Each read then gives C (methylated) / T (unmethylated) calls at those positions. The first 4 bases (MspI fill-in)
and the last 2 bases (end repair) are not called. Qualifying molecule: ≥ 4 calls, ≥ 80 % methylated; ε = isolated unmethylated interior
calls ÷ interior calls on qualifying molecules (the Stage Q definition, with ≥ 4 calls because reads are 51 bp). Per run: ε, number of
qualifying molecules; a run with < 20,000 qualifying molecules is not read. Per species and tissue: mean ε of its runs.

**Tests (fixed now).**
1. **Primary:** across the 45 mammal species, Spearman ρ(Tb, ε) > 0, one-sided p < 0.05, in liver AND in heart.
2. **Size:** slope of ln ε on Tb (ordinary least squares, species as points) between +0.5 % and +2 % per K in each tissue (consistent
   with the calculation); outside that range is reported as not consistent.
3. **Secondary (confounders, reported, not a pass/fail):** the same correlation controlling for log maximum longevity and log adult
   body mass (AnAge), and for qualifying-molecule count.
4. **Secondary:** birds (body temperature ~40–42 °C, literature, not AnAge) against mammals, per tissue — only one contrast between two
   lineages, so it cannot separate temperature from lineage; recorded only.
**Not done here:** fish (no body or water temperatures in AnAge yet; FishBase later).

---
**Status 2026-10-10: reading withdrawn.** The reference-free reader used here does not measure copy error (doors/data/DEV_SAM_LEVER_01/reader_check_01.md: ε depends on the CpG-calling threshold; fake CpGs from sequencing errors grow with reads per fragment). The results below are kept as run, but are not readings; a reread needs alignment to each species' genome.

## Results (2026-10-09; nothing above the line changed)
1,001 liver/heart runs read (mammals and birds), 1,000 with ≥ 20,000 qualifying molecules (median 792,614). Per-species table:
`data/DEV_XSPECIES_TEMP_01/species_tissue_eps.csv`; figure `fig_xspecies_temp.png`. Median ε 0.041 in mammal liver and heart.

| test | bar | liver (44 species) | heart (43 species) | met |
|---|---|---|---|---|
| 1. ρ(Tb, ε) > 0 one-sided p < 0.05, in BOTH tissues | both | ρ 0.29, **p 0.028** | ρ 0.12, p 0.22 | **no** (liver only) |
| 2. slope of ln ε on Tb within +0.5 to +2 %/K | both | +6.3 %/K (95 % CI −0.4 to +12.9) | +4.1 %/K (−1.5 to +9.7) | **no** (point estimates larger; CIs include the predicted +1 %) |
| 3. controlling for longevity, body mass, molecule count | recorded | +5.5 %/K, p 0.15 | +3.5 %/K, p 0.21 | – |
| 4. birds vs mammals | recorded | birds 1.33 × mammals | birds 1.37 × | – (lineage, not temperature: +1 %/K predicts ≈ +3 %) |

\measured The direction is the predicted one in both tissues and significant in liver alone; the primary test (both tissues) is not met,
and the effect is not resolved: 5 K of body temperature across 44 mammals gives a confidence interval ten times wider than the predicted
slope. The data cannot exclude the IAM prediction (+1 %/K lies inside both intervals) and cannot confirm it. The bird–mammal gap (33–37 %)
is far larger than temperature predicts and is a lineage difference. What would decide it: a wider temperature range within one lineage
with the same tissue (fish at different water temperatures; hibernating vs active state of one species), read the same way.

## Design check by simulation (2026-10-09, after the result; development)
Measured spread of ln ε: between mammal species 0.26 (liver, after temperature); within one species 0.136 (median over species).
Simulated with a true +1 %/K: today's design (44 species over 5.2 K) detects it with power **0.11** — it could not have decided.
One species, same tissue: 2 groups × 10 animals 10 K apart 0.43; 4 temperatures × 6 over 20 K 0.65; **40 animals over 20 K 0.85**;
torpid vs active, 8 + 8, 30 K 0.99 (caveat: copy errors arise at cell division, which slows in torpor). Requirement for the next data:
one species, one tissue, ≥ 20 K and ≥ 40 animals (or a torpor contrast with dividing cells).

## Development reading — fish, water temperature (2026-10-09; not a test, ~50 % power)
342 runs of 56 fish species with a FishBase water temperature (preferred 50th percentile, else midpoint of min–max; 0.2–28 °C), same reader.
| tissue | species | ρ | one-sided p | slope %/K (95 % CI) |
|---|---|---|---|---|
| gills | 34 | +0.19 | 0.14 | +0.45 (−0.38 to +1.28) |
| muscle | 33 | +0.07 | 0.36 | +0.39 (−0.60 to +1.39) |
| heart | 15 | +0.20 | 0.24 | +0.45 (−1.33 to +2.23) |
| liver | 16 | −0.27 | 0.84 | −0.65 (−2.17 to +0.87) |
\measured Over a ~28 K range the three larger tissues lean positive and their intervals hold the predicted +1 %/K and exclude the
+6 %/K of the mammal liver reading; liver leans negative. Undecided; habitat temperature is a species average, not the animals' own.
