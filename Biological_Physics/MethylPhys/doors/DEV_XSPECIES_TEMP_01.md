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
