# PROC-SALMON-01 — pre-registration (written 2026-10-01, before any Methow read is aligned)

**Question.** Can the gate-error reading — how faithfully a cell holds its methylation pattern, counted on single DNA molecules — be read for each
individual steelhead, per tissue, and does it differ between hatchery- and natural-origin fish? No reference population, no class floor, no atlas.

**Data.** Gavery et al. 2018 (G3), PRJNA325786: 20 adult males (10 hatchery, 10 natural), RBC and sperm each, RRBS single-end 101 bp.
Design and confounds: METHOW_DESIGN.md. Run → fish map: methow_run_map.csv.

**Processing (fixed now).** Trim Galore --rrbs; Bismark (bowtie2, directional) to the O. mykiss reference Omyk_1.0 (GCF_002163495.1); unique
best alignments only; the first 3 and last 3 read bases ignored (end-repair / M-bias). Read = one molecule.

**Statistics (same definition for every specimen and every comparison — the PROC-MOLECULE-01 lesson).**
- Qualifying molecule: ≥ 6 CpG calls, ≥ 80 % of them methylated.
- Isolated error: an unmethylated CpG call whose two neighbouring CpG calls on the same molecule are methylated. Opportunities = n − 2.
- Copy error ε = isolated errors / opportunities, pooled per specimen.
- Instrument: conversion failure c = methylated non-CpG calls / all non-CpG calls; sequencing error s from Bismark mismatch rate.
  Reported ε_corr = ε − s (as in PROC-ENCODE-01); c is reported beside every reading.
- Genotype mask: a CpG site is dropped for a fish if > 30 % of that fish's qualifying molecules covering it (≥ 5) carry an error there
  (a C→T variant, not a stochastic copy error).
- Holding energy E = ln((1 − ε_corr)/ε_corr) kT; M_fish = ΔG_ATP / RT at 10 °C = 22.94; φ = E / M.

**Predictions.**
- P0 (the instrument reads one fish consistently — null test): each specimen's runs split into two halves by run; ε_corr of the halves agrees with
  intraclass correlation ≥ 0.80 across the 40 specimens, and the median |difference| is smaller than the between-fish SD in each tissue.
- P1 (each fish has its own reading): between-fish SD of ε_corr exceeds the within-fish run-half SD in each tissue.
- P2 (temperature — IAM's law): if the copying machinery's energy gap is the same as in human cells, its value in kT units scales with 1/T.
  Human healthy cells read 0.029–0.036 on this statistic (PROC-MOLECULE-01), i.e. E = 3.29–3.51 kT at 310.15 K → predicted at 283.15 K:
  3.60–3.85 kT → **ε_corr 0.021–0.027** in RBC. Pass: the median RBC fish falls in that window.
- P3 (origin): hatchery vs natural, ε_corr per tissue, Mann–Whitney two-sided, Bonferroni over 2 tissues (α 0.025 each). A difference is
  attributed to origin only if, within age-4 fish (4 hatchery, 6 natural), the median difference has the same sign. If natural age-4 vs age-5 fish
  differ by as much as the origin effect, age is reported as an unresolved explanation.
- P4 (difference map vs published regions): per-CpG difference map hatchery − natural, in units of its own per-site noise. Threshold = the 95th
  percentile, over all 126 splits of the natural fish into 5 vs 5, of the largest |z| in each split map (look-elsewhere). Report how many of the
  published 85 RBC / 108 sperm regions exceed it; no prediction on the number.

**Stated limits now.** 20 males, one return year, two hatcheries; age and origin overlap only at age 4. RRBS reads CpG-dense regions, which may
hold their pattern differently from the whole-genome molecules the human value came from (P2). A difference is not proof of a fitness effect.

## Deviation recorded 2026-10-01 (before any fish was scored)
The pilot (one specimen, 10 M reads) mapped 0.6 % as directional. The study's own methods state PBAT-type libraries (EpiNext post-bisulfite
adapter tagging), Trim Galore --rrbs --non_directional and Bismark --pbat --score_min L,0,-0.2. Processing changed to those settings; strand
collapse now uses the XG tag. Statistics and predictions are unchanged.
Pilot with these settings (fish 100, RBC, natural origin; 10 M reads, before the full run): mapping 29.0 %; CpG methylation 88.8 %; non-CpG
methylation 0.7–0.9 % (conversion failure 0.0071); A/T mismatch 0.0074; raw ε 0.0429, ε_corr 0.0355. This value was seen before the full run and
is disclosed here; predictions are not changed (P2's window 0.021–0.027 stands).
