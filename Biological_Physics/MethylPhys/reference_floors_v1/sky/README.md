# One neutrophil on the sky — Met-A residual maps (development, 2026-10-02)

Six physical Salas EPIC purified neutrophil arrays (GSE110554; GSE167998 re-deposits the same six and is not read), our Stage 1.
Each array is read against the other five. Script `sky_neut6.py` (box job fcb79e1e); figures `make_sky_figs.py`.

- Map sites: across-array SD <= 0.05 among the five, mean beta 0.75-0.95 or 0.05-0.25 (about 450,000-464,000 per held-out array).
- Site z = (H(beta) - mean H of the five) / shrunk SD. All EPIC probes in genome order -> HEALPix NSIDE 64 RING pixels in sequence
  (865,873 probes, 17.6 per pixel); pixel value = sum z / sqrt(n).
- Met-A and C-score: the chain's 6,000 frozen identity sites (metA_floors_v1_3.json); C-score = clustering of z in 50-site genome blocks
  divided by the healthy baseline 1.1104 (neutrophil_reference_v1_1.json).

| specimen (6 arrays) | Met-A | C-score |
|---|---|---|
| healthy | 0.993-1.008 | 0.71-1.23 |
| 2 % blur toward beta 0.5 at every site | 1.096-1.110 | 0.78-1.23 |
| 5 % blur inside 10 genomic regions (5 % of sites) | 1.005-1.020 | 12.5-15.7 |

1. Met-A sees damage spread everywhere; the C-score sees damage concentrated in regions, which Met-A averages away. Report both.
2. GSM2998057 reads as female (40 % of chrX sites at beta 0.3-0.7 vs 8 % in the other five; 121 chrY probes detected vs ~540). Read against
   five male arrays, its whole-array map lights up chrX (pixel mean z +21 on chrX, about -1 elsewhere). The 32 chrX identity sites read the
   same beta on all six arrays, so Met-A and the C-score are unaffected (0.993, 0.93). Whole-array maps need a sex mask or a same-sex reference.
3. Whole-array map clustering for healthy arrays is 1.6-2.1 (neighbouring sites share residuals); the maps involving the female array read
   12-18 because of chrX.
