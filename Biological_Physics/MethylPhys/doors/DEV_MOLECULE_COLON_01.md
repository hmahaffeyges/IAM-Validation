# DEV-MOLECULE-COLON-01 — reading colon cells' copy error from their own molecules (development, 2026-10-01)

**Why.** To read IAM-A for one cell type inside a mixture (stool, blood, tissue) without unmixing: count copy errors only on molecules that can only
have come from that cell. IAM-A uses methylated molecules (≥ 6 CpGs, ≥ 80 % methylated), so in blocks where colon epithelium is methylated and every
other source is unmethylated, the qualifying rule itself selects colon molecules.

**Marker blocks** (Loyfer 2023 .beta, hg19 CpG index rebuilt from the genome and checked: 895,713/895,713 probes agree; lifted to GRCh38).
Colon epithelium (left + right, 5 samples) against 15 blood cell groups, colon fibroblasts and colon macrophages:
- strict (colon ≥ 0.80, all others ≤ 0.10, ≥ 4 CpGs within 100 bp): 163 blocks, 1,905 CpGs;
- loose (colon ≥ 0.70, others ≤ 0.20, ≥ 3 CpGs): 1,030 blocks, 8,091 CpGs.

**Read on the existing tumour and normal site tables** (20 M reads per specimen, early-onset CRC and oral SCC; no re-alignment).
- **Molecules in the blocks are too few at whole-genome depth.** Loose blocks give 9–3,385 qualifying opportunities per specimen (110–1,300 per million).
  At ε ≈ 0.04 that is 0–135 error events, so the per-specimen error is ±10–100 %. The tumour/normal ratios in the blocks scatter from 0.6 to 2.7. That scatter is counting noise; it is not a signal.
- **All-molecule copy error reproduces the earlier tumour result**: tumour above normal in 6/6 CRC pairs (1.02–1.44).
- **The blocks are colon-specific against blood and stroma, not against other epithelia.** Oral tumour and normal read 12–37 % "colon content".
  Tumours also gain methylation in these blocks, which inflates their content estimate.

**What this teaches.**
1. The method is sound in principle: in these blocks, a qualifying molecule is a colon-epithelial molecule. But whole-genome sequencing
   puts only ~0.1 % of molecules there. That is too few to read one person.
2. **The design that works is targeted capture of the blocks.** At 1,000× per block over the 1,030 loose blocks, ~6 M opportunities come from colon
   molecules in pure colon tissue, or ~600 k if colon cells are a tenth of the human DNA. That puts ε at ±1 %. This is the route for stool.
3. For stool, the blocks must also exclude the other sources of human DNA in the gut (small-intestine and gastric epithelium, squamous cells)
   and must be checked against tumour-gained methylation, since CRC hypermethylates many colon-methylated regions.
4. Next: rebuild blocks against all 39 Loyfer cell groups; check them on a public stool or colon-lavage methylome if one exists. Otherwise, mix
   colon and blood reads in silico at known fractions and read the error back.
