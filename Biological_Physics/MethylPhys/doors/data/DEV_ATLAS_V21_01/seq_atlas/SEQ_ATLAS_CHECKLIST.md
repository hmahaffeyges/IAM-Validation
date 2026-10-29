# IAM Sequencing Atlas: what is baked in

A separate atlas for sequenced samples only. IAM-Atlas v2.1 stays as commissioned for arrays; nothing here changes an array reading.

**Its job:** a healthy sequenced sample reads Normal on every instrument, read against the healthy cells of its own type, measured from sequencing.

**Its rule:** measure, not fit. Every value is counted from healthy donors' reads. Nothing is filled where nothing was measured.

## A. What is stored at every CpG of the genome

| # | Stored | Why | Source |
|---|---|---|---|
| A1 | Each healthy donor's marked and total read counts, kept per donor, never averaged away | Any depth rule can be applied later without rebuilding; the common-depth test decides the rule, the atlas keeps the raw counts | Met-A depth findings |
| A2 | Each donor's measured disorder at the site | Healthy disorder is measured, not computed from a mean level: a 0.02 donor difference raised Met-A to about 1.3 | fixed-depth synthetic |
| A3 | Per cell: median level, donor spread, number of donors, labs and reads | Sample variance: the next person, not only the average | v2.1 tool; Gift 3 (ATL-051) |
| A4 | Blank where no donor measured the site, with the count beside every value | Masks, never filling | v2.1 tool; map row 16 |
| A5 | Each laboratory's offset per cell, where a cell has two or more labs | One map from many instruments | v2.1 tool (destriping) |
| A6 | The molecules themselves (read patterns), per donor | Held error (IAM-A), nearest-neighbor coupling, molecule-level markers for cell-free DNA | IAM-A; coupling; ATL-068 |
| A7 | Each donor's sex, age and other recorded traits | Foregrounds are annotated, never subtracted | map rows 18, 19, 47 |
| A8 | Counts stored at full range (16-bit), not capped at 255 | Deep samples lose information in an 8-bit count | the set-aside builder capped at 255 |

## B. A healthy reference for every sequencing instrument

| # | Instrument | What the atlas gives it |
|---|---|---|
| B1 | IAM-A | Healthy held error per cell type, and its per-cell band (only a pooled band today) |
| B2 | Coupling-A | Healthy coupling per cell type (today from 153 samples) |
| B3 | Met-A on sequencing | Identity sites chosen on the sequencing atlas by the same rule, sealed before testing; healthy Met-A per cell |
| B4 | Territories | Healthy islands and domains per cell type |
| B5 | C-Score | Healthy limits per cell on sequencing, including the neighbor correlation of each healthy map |
| B6 | IAM-Deconvolver | Cell columns in sequencing units, for sequenced mixtures and plasma; a megakaryocyte column when a source exists |
| B7 | Every cell in a mix, one blood draw | The same columns, read per cell |
| B8 | Tumor readings (ceiling gauge) | The cancer's cell of origin as its healthy reference |
| B9 | Group-A | The healthy group statistics per cell |

## C. The spheres

| # | Stored | Why |
|---|---|---|
| C1 | Every CpG of the genome placed on the HEALPix sphere by genome address, at a resolution fine enough for the whole genome | The sky sphere and Globe Explorer (map row 15) |
| C2 | Every site's healthy position on the information sphere per cell, with its Fisher-Rao distance | The information sphere reads a sample's distance from its own healthy self |
| C3 | Noise maps for both spheres (from the half-atlases and donor resampling) | ATL-046: tells signal from noise on the sphere |
| C4 | The three spheres compared: atlas v2, v2.1 and the sequencing atlas | ATL-048 |

## D. CMB tools carried over

| # | Tool | In the sequencing atlas | Map row / record |
|---|---|---|---|
| D1 | Half-mission split | **Two half-atlases**: every cell's donors split in two by a rule sealed before any build; half A minus half B is a pure noise map; crossing the halves gives spectra without noise bias. | row 22; ATL-062 |
| D2 | Second noise split, read halves | Each donor's reads split in two at random: a noise map from counting alone, available even for cells with one donor (most cells have 1 to 4) | row 22, row 31 |
| D3 | Noise maps | From the half differences and donor resampling, published with the atlas | v2.1 tool; ATL-046 |
| D4 | 95 % range per value | From reads and donors, by counting | v2.1 tool |
| D5 | Masks and apodization | SNP sites, sex chromosomes, blacklisted and hard-to-map regions; smooth mask edges | row 16 |
| D6 | Genome-distance spectrum C(d) | Healthy spectra per cell in log-spaced bands, and the mode coupling from masked sites | rows 1, 17, 23, 25 |
| D7 | Cosmic variance | The cell-count floor: copies per site behind each reading; needed for plasma | row 3 |
| D8 | Covariance | Cross-site covariance of healthy donors | row 29; v2.1 tool |
| D9 | Transfer function and end-to-end simulation | Constructed samples built from the atlas measure what the chain absorbs | rows 14, 30 |
| D10 | Null tests | Half differences and shuffled nulls for every map statistic | row 31 |
| D11 | Frequency coverage and cross-calibration | Bisulfite (WGBS) and enzymatic (EM-seq) kept apart and compared; the bridge to v2.1 on sites both measure | row 4; Gift 2 (ATL-050) |
| D12 | E/B separation | 5mC against 5hmC where oxidative data exist | row 49 |
| D13 | Higher-order statistics | Three- and four-site shapes, Minkowski functionals on healthy maps | rows 53 to 55 |
| D14 | Isotropy tests | Per-chromosome healthy means | row 57 |
| D15 | Blinding and held-out skies | Held-out labs and donors reserved before the build and never read by it; every bar sealed before it is scored | rows 32, 61, 78 |
| D16 | Reproducibility | Built through the chain from committed code and listed inputs; bit-identical rebuild; own version and DOI; re-commissioned before the chain switches | row 79; ATL-044 |
| D17 | The guidebook | Each tool, its origin in cosmology, and how it applies | ATL-063 |

## E. Rules for the build

1. **Sources:** every healthy sorted-cell sequencing data set we can find, recorded in the spend ledger. Held-out sets for commissioning stay unread by the build.
2. **Units:** sequencing units only. Nothing is converted from arrays.
3. **Coordinates:** the chain's hg19 CpG numbering, both strands combined, with an hg38 map.
4. **Depth:** raw counts kept (A1). The common-depth test sets the reading rule.
5. **Compute:** GitHub's machines plus Google spot machines (C3D, Dallas, about 2 cents an hour for a 4-CPU machine). Timed on one machine before any large run.

## F. Only where the data hold it

| # | Stored | Note |
|---|---|---|
| F1 | Fragment features (nucleosome occupancy, fuzziness, WPS, DELFI fragment size) | Healthy plasma only. Sorted-cell libraries are cut by sonication, so their fragments carry no nucleosome information |
