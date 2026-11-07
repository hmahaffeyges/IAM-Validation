# The Sequencing Atlas Globe at every CpG

The IAM Sequencing Atlas (Zenodo 10.5281/zenodo.23934666) placed on a sphere: all 28,217,448 CpGs of the hg19 numbering, for each of its 57
healthy human cell types, on a HEALPix sphere at NSIDE 2048 (50,331,648 pixels, NESTED ordering). The CpGs are laid out in genome order,
chromosome 1 to 22, then X and Y. CpG i of n goes to pixel floor(i x 50,331,648 / n), so every CpG has its own pixel and genome neighbors
stay sky neighbors. Pixels that hold no CpG are blank.

## Why a sphere

A map of the cosmic microwave background and a methylome are each a field of very many points held near an expected state, and in each the
job is to find a real departure in the noise. Both fields have the same structure because the same law holds both. Every point is a record
held against thermal noise, and IAM's law sets what holding it costs and how it is lost. The statistics built to find a faint signal on the
sky therefore apply to the genome unchanged: the sphere and its pixelization, masks, noise maps from split halves, null tests and the
spectrum. This globe carries the atlas into that form.

## What each pixel holds

For each cell type, at each CpG, three values: the healthy level, and the low and high ends of its 95 % range. These are the level of the
cell's average and its range by donors and by counting, as `stage_seq_atlas.range_of_average` computes them from the atlas's counts with
each cell's number of healthy inputs. All three are stored as unsigned 16-bit integers, level x 10,000, with 65535 meaning blank. Nothing is
fitted. Every value comes from the atlas's counted reads.

## Files

- `globe57_setNN_of_12.tar`: one tar per published atlas piece set. Each holds one `.npz` per atlas piece of 250,000 CpGs, with that piece's
  stretch of the globe (`p0`, its first pixel; for each cell, `mu`, `lo` and `hi`) and its sums and counts at NSIDE 256.
- `globe_wg_textures_n256.npz`: the same globe at NSIDE 256 (786,432 pixels, each the mean of its CpGs), the size the Globe Viewer draws.
- `globe_wg_check.json`: the check. At every one of the 813,951 IAM-Atlas v2.1 array-layout sites, for all 57 cells, this globe's level,
  low and high equal the earlier Sequencing Atlas Globe's (139,185,168 values, all equal).
- `seq_globe_wg_synthetic_run2.txt`: the builder's test on a constructed atlas of the same size, met on all three bars.
- `seq_globe_wg.py`: the builder, as committed in the repository (`Biological_Physics/MethylPhys/atlas/tools/`).
- `CHECKSUMS.txt`: every file's checksum as Zenodo returned it, and its size.

## Rebuilding it

From the published atlas alone: `python3 seq_globe_wg.py slice seq_atlas_inputs.csv OUT piece_*.parquet` for each piece set, then
`python3 seq_globe_wg.py assemble OUT2 OUT...`. The code and the inputs table are in https://github.com/hmahaffeyges/IAM-Validation.
