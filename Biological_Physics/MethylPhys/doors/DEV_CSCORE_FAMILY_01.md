# DEV-CSCORE-FAMILY-01: two new members of the C-Score family (design)

**What the C-Score reads.** Not how large a change is, but how it is laid out. The commissioned statistic (`conductor_v3.stage_mc_cscore`):
take the sample's departure at each site from its healthy reference, in units of the healthy spread (z); cut the sites into blocks of
consecutive neighbors; multiply each block's mean by the square root of the block size (`_clustering`: `mean(block) * sqrt(w)`); the C-Score
is the variance of those scaled block means over the variance of the sites (the same as the variance of the block means times the block
size), divided by the same number in healthy samples. Scattered departures (heat's kicks, one site at a time) give about 1; departures that move together within
blocks give more. The physics: a held mark that slips under thermal noise slips alone; a program that stops paying for a whole set of marks
moves them together.

**The family today:** (a) commissioned, at Met-A's neutrophil sites, blocks along the genome; (b) the same statistic on the globe's
whole-genome change map against IAM-Atlas v2.1 (development); (c) IAM-A's version, held-error departures along the genome (development).

## (d) The C-Score on the information sphere

**Its job:** find a program failing across many chromosomes at once.

| Part | Rule |
|---|---|
| Neighbors | sites in the same pixel of the information sphere: sites that behave alike across the healthy cell types, wherever they sit on the DNA |
| Statistic | the commissioned one, unchanged; only the blocks change, from runs along the genome to pixels on the sphere |
| Healthy baseline | the same statistic on healthy samples of the same cell, so healthy reads 1 |
| Why it adds something | (a) to (c) see a change only where its sites are neighbors on the DNA; a cell program's sites are scattered over the genome but sit together on the information sphere |

**The synthetic, to be written and sealed with the code:** healthy inputs with three plants on the same number of sites: (1) scattered at random; (2) a run
along the genome; (3) the sites of one information-sphere neighborhood, spread over many chromosomes. What it must show: healthy reads inside its own
band; plant (3) reads clustered on (d) and not on (a); plant (2) reads clustered on (a); plant (1) reads clustered on neither.

## (e) The whole-genome C-Score on sequencing

**Its job:** the same reading as (b), at every CpG of a sequenced sample, against the IAM Sequencing Atlas.

| Part | Rule |
|---|---|
| Departures | each CpG's level against the person's own expected healthy self (the atlas's cells at the sample's shares), in units of the healthy spread counted at the sample's own reads |
| Healthy baseline | from the atlas's own healthy covariance (checklist D8) and genome-distance spectrum (D6): healthy neighbors are correlated at short range (the spectrum's result in DEV_ATLAS_V21_01.md), so a healthy sample's blocks are not independent, and the baseline must carry that |
| Masks | the sex chromosomes and the poorly mapping sites, as the sky statistics use (D5) |

**The synthetic, to be written and sealed with the code:** healthy inputs, and the same inputs with a planted clustered loss and a planted scattered loss of the
same total size. What it must show: healthy within its band; clustered above it; scattered within it.

**When:** both are code in the chain, so they are written after the B and T and Johns Hopkins readings finish (one chain version). Their
synthetics then run on GitHub. A real-data check on a known clustered gain: GSE306917 (chosen genes methylated on purpose, with same-donor
controls; held unread).
