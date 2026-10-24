# Methods (draft; plan 50.2; not for circulation)

Every constant below is copied from the chain or the book's formula appendix (`docs/book/appendices/app_E_formulas.tex`), with its source
named, so that the paper's generator can replace each with a value read at build time. Marked `[check]`: a statement still to be checked
against its record before the draft is final.

## Data

All data are public. Whole-genome bisulfite sequencing of pediatric pre-B acute lymphoblastic leukemia, the Reh line with a non-targeting
control and with UHRF1 knocked out, and umbilical cord blood CD19+ B cells: GEO GSE116229 (36 samples with reads). The reference cell
types, the laboratories and the platforms behind IAM-Atlas v2.1, and every other series used, are listed with their accessions in the
supplementary data table, generated from `CANON/data_spent.csv`, which also records for each series whether it was used to develop a
method or held out to test it.

## A thermodynamic bound on a maintained mark

A methylated CpG on one strand is a stored binary record. Erasing or rewriting one bit dissipates at least k_BT ln 2 (Landauer); at body
temperature (310.15 K) this is 2.968 x 10^-21 J. The cell's drive per maintenance event is set by ATP hydrolysis, and we express it as
the dimensionless ratio M = ΔG_ATP / (R T_b) = 54,000 / (8.314462618 x 310.15) = 20.94 (formula appendix entry 485), so one ATP can pay
for M / ln 2 = 30.21 Landauer bits (entry 487).

A held mark is a balance between loss (rate f per site per division) and restoration (rate g). The steady-state error is
ε = f / (f + g), so (1 - ε) / ε = g / f, and the free energy that holds the mark is E_hold = k_BT ln(g / f) (entry 493). Measured from
single molecules in 56 healthy human cell types, E_hold = 3.08 k_BT, the same across cell types within their spread, a fraction
φ = E_hold / (M k_BT) = 0.147 ± 0.006 of one ATP (entry 492; `doors/data/DEV_CHAIN_Q2_01/holding_energy_q2.json`). This is consistent
with the 2.3 to 3.9 k_BT implied by measured maintenance error rates of 2 to 10 % (Genereux et al. 2005; entry 491).

The bound defines the scale on which every reading is placed: a floor, where thermal noise alone sets the error of a mark held with the
full drive, and a ceiling, where a site has lost its mark and its Shannon entropy H(β) = -β log2 β - (1 - β) log2(1 - β) reaches 1 bit.
`[check: state the floor equation used by each instrument exactly as the chain computes it; the qubit-gauge form p_eq = 1/(1 + e^M),
entry 475, is the two-level analogue.]` Prior information-thermodynamic treatments of methylation (the work the book's Landauer chapter
discusses under "information thermodynamics of methylation", entries 494-495) are cited and compared here. `[check: full reference.]`

## Counting molecules: the loose share and the held error (IAM-A)

Reads are aligned by bwa-meth to hg19 and converted to per-molecule methylation patterns (wgbstools .pat format) by the chain's alignment
script (`doors/data/DEV_ERASE_01/erase_gh.sh`); for fixed-depth readings a fixed seed draws the same number of read pairs (10 million) from
each sample. Stage Q2 (`chain/stage_q2_selfref.py`) then reads each molecule once, alternating lines between two disjoint uses so that no
molecule informs both quantities:

- **The loose share L.** The genome is cut into tiles of 2,000 CpGs; a tile with at least 200 calls is loose when its mean methylation is
  below 0.70. L is the share of loose tiles. A molecule-level share (molecules with at least 6 calls whose methylated fraction is below
  0.70) is recorded beside it, because it adds linearly when cell types mix and the tile share does not (synthetic test cfDNA 2).
- **The held error ε.** On molecules held marked (at least 6 calls, at least 80 % methylated), ε is the rate of isolated unmethylated
  calls between two methylated neighbors, over all interior positions. The mirror quantity on molecules held unmarked (at most 20 %
  methylated) is recorded as the gain error.

**The self-reference line.** A cell's design leaves a share L of its genome loosely held, and its expected held error follows from that
share: ε_pred(L) = ε_firm + s L, with ε_firm = 0.04026 and s = 0.03462 estimated once by the Theil-Sen estimator on 153 samples of 56
healthy cell types (`chain/Runtime Matrices/IAM_A_SelfRef/selfref_line_v1.json`). A sample's reading is IAM-A = ε / ε_pred(L), from its
own molecules; no control group enters. `[check: why the line is not circular; the synthetic tests and the held-out reproductions to
cite.]`

## IAM-Atlas v2.1

**Model.** For each cell type c and site l, the mean methylation μ_cl is modeled on the logit scale with a site-level hierarchical prior,
logit μ_cl ~ Normal(ν_l, τ), ν_l ~ Normal(0, 2), τ ~ HalfNormal(2), and the person-to-person spread of the cell at the site as
log σ_cl ~ Normal(log 0.03, 0.7). An array observation y has mean μ_cl + d_lab and variance σ_cl² + t² + 0.01²; a sequencing observation
has mean g(μ_cl), with g the measured monotone transfer between the two platforms (a piecewise cubic Hermite curve bounded in [0, 1],
estimated on 767,672 sites measured both ways in the same 17 cell types), and variance g'(μ)² σ_cl² + h(1 - h)/coverage + n s(μ)² +
t² + 0.01², carrying the read-count noise, the transfer's own measured scatter s and the source's sample count n
(`doors/data/DEV_ATLAS_V21_01/atlas_block_v21.py`).

**Laboratory and platform terms.** Each array laboratory's offset d_lab, each source's extra scatter t and the transfer g are estimated in a
separate calibration stage on cell types shared between sources and held fixed in the posterior sampling, with one laboratory as the
anchor (d = 0). Sampling the terms inside every block was tested on synthetic data generated from the model's own likelihood; its bars
were not met, and the fixed terms were kept (`doors/data/DEV_ATLAS_V21_01/labterms_synthetic_v21_summary.txt`).

**Sampling.** The 813,951 sites are fitted in 700 blocks with the No-U-Turn sampler (NumPyro; target acceptance 0.9, maximum tree depth 8,
600 warm-up and 400 draws on each of 4 chains, seed 1000 + block). R-hat and the effective sample size are stored for every site and cell:
0.42 % of values have R-hat of 1.01 or above, and the fit had 13 divergent transitions in 700 blocks; flagged values are kept with their
flag. A site a cell's sources never measured is left unmeasured, never imputed. One hundred thinned posterior draws per block are public
(Zenodo 10.5281/zenodo.23422688). **Split-half null test:** the atlas refitted on two halves of every cell's donors; the difference map
must be noise. `[check: the null result to cite.]`

## Deconvolution

Each sample's cell-type shares are estimated by weighted non-negative least squares on marker sites, the sites that best separate each cell
type from its nearest neighbor in the panel in both directions, z = |μ_c - μ_nearest| / sqrt(σ_c² + σ_nearest²). Each site's weight is
1 / (Σ_k f_k² sd_k² + lab_sd²), refined over three passes from equal shares; the reference spread sd² combines the posterior uncertainty of
the mean, the donor spread and a measured laboratory term. Shares are normalized to sum to one, with the raw sum reported; 95 % intervals
come from refits on resampled marker sites and on cell profiles redrawn from the atlas's own spread (`chain/stage_deconv_v21.py`). What no
healthy mix explains (the residual) is reported rather than discarded, and a share of a cell type the atlas lacks is flagged.

## Met-A

For one cell type in one specimen, Met-A = mean over the cell's identity sites of H(β) / H_ref, where H_ref is the healthy reference
value on the same platform. Identity sites: across-array SD at most 0.05 and mean β of 0.75 to 0.95 (methylated) or 0.05 to 0.25
(unmethylated), up to 3,000 per direction. Normal is 0.95 to 1.05 (`chain/stage_m_met_a.py`). In a mixture, each cell's expectation is
built from the atlas at the sample's own shares. **The healthy line** (Met-A's healthy reference derived laboratory by laboratory with no
cohort): `[pending: DEV-METAA-HEALTHY-01 bar 2 and layer 1]`.

## The territories, the genome-wide projection and the embedding

`[to write after 49.5: the territory rules (islands; loosely held domains) from stage_metaa_atlas.territory_sets; the equal-area
genome-wide projection, every site at a fixed address; the information-geometric embedding (Fisher-Rao distance on per-site departures);
the sensitivity of each site, dH/dε at its held level, derived analytically.]`

## Pre-registration and reproducibility

Every test was written, with its question, data, bars and expected outcomes, and committed to the public repository before it ran; each
record and its result are public with the commit that produced them. Data used to develop a method are recorded as spent and are not used
to test it. Every figure and value in this paper is produced by one command (`run_sample.py`) and the paper's generator, from public data.

## Statistics

`[to write: tests used per result (two-sided Mann-Whitney for group comparisons in R4), intervals, multiple-comparison control for
region-level results.]`
