# The Sequencing Atlas Globe's noise maps at every CpG

Version 1.2 of the IAM Sequencing Atlas adds the noise maps of its 57 cell types (the megakaryocyte column's are in
`globe_wg_noise_maps_megakaryocytes_n256.npz`, one of the Sequencing Atlas Globe's files, also in this version; see README_globe.md). On the sky, a noise map is made by splitting the observations in two
and subtracting the halves: the sky cancels and only the noise remains. The same is done here at every one of the 28,217,448 CpGs.

## The two maps

- **Map 1, donor halves** (`<cell>|map1 donor halves`, 49 cells, every cell with two or more healthy donors): the cell's donors split into two
  halves, each half's level computed from its reads, and half their difference taken at each CpG. What remains is how healthy people differ
  from one another, plus counting noise.
- **Map 2, read halves** (`<cell>|map2 read halves`, all 57 cells): the same cell's reads split into two halves at random. Only counting noise
  remains.

Each map is stored at HEALPix NSIDE 256 (786,432 pixels, NESTED ordering), in the Sequencing Atlas Globe's own genome-order layout: CpG i of n goes to pixel
floor(i x 786,432 / n), and each pixel holds the average of its CpGs' half-differences (about 36 CpGs a pixel). Pixels with no measured CpG,
and the sex chromosomes (the last pixels, since X and Y come last in genome order), are NaN: the sex chromosomes are left out of the tools as
a healthy foreground, as a bright galaxy is masked on the sky.

## What was measured on them

- `globe_wg_noise.json`: per cell and map, computed on every CpG (not on the pixel maps): mean and mean square, the spectrum along the genome
  (correlation by distance in bases), the sphere null test (each band's power against 20 within-mask shuffles, apodised mask) and the sign-flip
  null (each band's power against 20 random sign flips of the same values).
- `globe_wg_noise_summary.txt`, `globe_wg_flip_summary.txt`: the readings of those numbers. Read halves show no structure, only counting;
  donor halves show regions, not CpG-by-CpG differences.
- `seq_globe_wg_noise_synthetic.txt`, `seq_globe_wg_flip_synthetic.txt`: each tool's synthetic, run on all 28,217,448 constructed CpGs before it
  read the atlas, both met.

## Rebuilding them

From this atlas's pieces (the `pieces57_set*_of_12.tar` files here) and the repository (https://github.com/hmahaffeyges/IAM-Validation):
`Biological_Physics/MethylPhys/atlas/tools/seq_globe_wg.py noise-part` on each of the atlas's 113 pieces, then `noise-join` (the workflow
`.github/workflows/globe_noise.yml` runs it on GitHub's machines). The maps deposited here are that workflow's output, run 42339899387,
SHA256 6226e6a464293e13ce48a23a4ec8f732d4088c47ff2886eabd87aeae6cbdfdd7. `CHECKSUMS_v1.2.txt` gives every added file's SHA256 and size.
