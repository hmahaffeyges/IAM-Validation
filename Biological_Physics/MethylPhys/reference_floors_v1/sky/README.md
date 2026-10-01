# One neutrophil on the sky — Met-A residual maps (development, 2026-10-01; `sky_neut.py`)
Salas EPIC purified neutrophils (12). Held-out specimen vs the other 11. ~457,000 reference sites (across-donor SD ≤ 0.05, β 0.75–0.95 or
0.05–0.25). Site z = (H(β) − reference mean H) / reference SD (shrunk). All EPIC probes in genomic order → HEALPix NSIDE 64; grey = no site.
Clustering = variance of z averaged over 50 consecutive sites, × 50, over the site variance (1 = independent scatter).

| map (GSM5121412; GSM2998021) | A | clustering |
|---|---|---|
| healthy | 1.004; 0.994 | 2.2; 1.7 |
| null, healthy minus healthy | — | 1.9 |
| 2 % blur everywhere | 1.062; 1.052 | 2.2; 1.9 |
| 5 % blur in 10 regions (5 % of sites) | **1.011; 1.001** (Normal) | **6.8; 6.4** |

1. A and the map measure different things: A sees damage spread everywhere; the map's clustering sees damage concentrated in regions,
   which A averages away. Report both.
2. Healthy and null maps already cluster at about 2: neighbouring sites share residuals (donor genetics, regional array effects). That is the
   floor any real feature must exceed; it is measured from healthy specimens, not assumed to be 1.
3. Next: the same maps for real disease neutrophils (AML EPIC in S3), and the IAM-A map from sequencing for the cross-correlation.
