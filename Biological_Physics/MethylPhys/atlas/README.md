# IAM-Atlas v2: a reference map of healthy human cells

IAM-Atlas v2 is the reference every IAM instrument reads against. For each of **74 purified healthy human cell types**, at every CpG on the
Illumina methylation array, it gives the cell's methylation and **how sure we are of it**: a mean, a standard deviation, a 95 % interval and
20 posterior draws, per site, per cell.

**Last atlas search for new purified cells: 2026-10-12.** A search is due at least every 6 months; the push gate refuses a push when this
date is more than 6 months old.

## How it was made

One joint Bayesian fit (NumPyro, NUTS, 4 chains) over the purified healthy samples that met its entry rule in public data: 352 samples from seven
sources, two kinds of measurement (arrays and bisulfite sequencing), several laboratories.

| source | measurement | samples |
|---|---|---|
| Loyfer et al. 2023, Nature (GSE186458) | whole-genome bisulfite sequencing | 207 |
| Salas et al. 2022 (GSE167998) | EPIC array | 56 |
| Salas et al. 2018 (GSE110554) | EPIC array | 35 |
| Moss et al. 2018 (GSE122126) | 450K/EPIC arrays | 27 |
| GSE63409 (sorted healthy bone-marrow stem and progenitor cells; the series' leukemia arrays are not used) | EPIC array | 18 |
| Tian et al. 2023 (GSE215353) | pooled brain sequencing | 7 |
| ENCODE | whole-genome bisulfite sequencing | 2 |

Three things set the fit apart:
- **Every laboratory and platform has its own measured offset.** The same cell measured on an array in one laboratory and by sequencing in
  another is one cell; the model estimates the difference between sources instead of leaving it as noise.
- **Every site carries an uncertainty, and it was checked.** 5 % of the measurements were hidden, the atlas refitted, and the hidden values
  predicted: the 90 % intervals held them as often as stated (record: [`PROC_V5_HELDOUT_OUTCOME.md`](../doors/PROC_V5_HELDOUT_OUTCOME.md)).
- **Nothing is filled in.** A cell never measured at a site is marked NOT MEASURED, never given a guessed value.

Every step, script by script, with the machine and package versions: [`v2/README.md`](v2/README.md). Why each choice was made:
[`v2/DECISIONS.md`](v2/DECISIONS.md). What was run on the finished atlas: [`v2/postbuild/README.md`](v2/postbuild/README.md).
The model and its acceptance tests, written before the build: [`../doors/ATLAS_V2_SPEC.md`](../doors/ATLAS_V2_SPEC.md).

## What it can do that a tissue atlas cannot

Atlases of whole tissues are built to name the organ a sample came from. Each organ's profile already averages that organ's own mixture of
cells, so a change in the mixture and a change in how one cell type holds its pattern look alike. This atlas holds each purified cell type
on its own, which allows three things a tissue profile does not:
- **telling a change of state from a change of mix:** a test reads only sites where every cell type of the tissue agrees, so a shift in
  the mixture cannot move the reading (the sealed tissue tests are built this way);
- **reading one cell type inside a mixture:** whole blood is read for its neutrophils against an expectation built from that person's own
  cell mixture ([book, atlas chapter](https://hmahaffeyges.github.io/IAM-Validation/pdf/IAMs_Law_and_Order.pdf));
- **saying how far a reading can be trusted,** from the uncertainty at every site.

Naming a sample's organ of origin is a task tissue atlases are built for and this one is not. And the atlas has gaps: 74 cell types, and
no lung fibroblasts or lung smooth muscle, so lung tissue tests cannot yet be protected against lung stroma
([`DEV_TISSUE_LUNG_03.md`](../doors/DEV_TISSUE_LUNG_03.md)).

## Why it matters to IAM, and what is new

The IAM instruments measure how far a cell's methylation pattern has drifted from healthy. That needs to know what healthy is, for each
cell type, with an honest uncertainty, so a reading can say whether a change is larger than what is known. The atlas also chooses the sites
a tissue test reads: sites where every cell type agrees, so a change in the mix of cells in a sample cannot move the reading. Every cancer
fingerprint result on patients' tissue rests on those sites.

**What is new, as far as we know.** Other atlases are larger or deeper, and each is built for a different job:
- a 2025 compendium of 16,959 healthy 450K samples is built to classify **tissues** (mostly whole tissues, arrays only);
- Loyfer et al. 2023 sequenced 39 purified cell-type groups (this atlas is built in large part on it);
- CelFiE-type methods model the reference probabilistically, but **re-estimate it from the sample being read**.

IAM needs the opposite of the last one: a healthy reference **fixed before any sample is read**, so a specimen is measured against it and
never fitted to it; a reference that bends toward a sick sample hides the very drift being measured. We know of no other public reference
of purified healthy cells that fits arrays and sequencing from several laboratories in one model, measures each laboratory's offset,
carries an uncertainty at every site that held-out data confirmed, and is frozen before use. Literature checked 2026-10-12; a correction is
welcome as a GitHub issue.

## Questions the atlas builders would ask

**What does this add to the sequencing atlas of purified cells (Loyfer et al. 2023)?** That atlas is the deepest public map of purified human
cells, one laboratory and one platform, and most of this atlas's samples come from it. This atlas places it on the same scale as the
array references of other laboratories (Moss 2018, Salas 2018 and 2022, GSE63409) and the pooled brain sequencing of Tian 2023, in one model,
and gives every cell type at every site a posterior: a mean, a 95 % interval and 20 draws. As far as we know, no other public purified-cell
reference publishes an interval at every site, and none has tested its intervals on data held back from the fit: here 90 % intervals covered
92.7 % of 342,716 hidden values, every one of 20 test blocks between 92.3 % and 93.1 % ([`PROC_V5_HELDOUT_OUTCOME.md`](../doors/PROC_V5_HELDOUT_OUTCOME.md)).

**How can references from different laboratories and platforms be combined at all, when each laboratory's offset is larger than the
differences being measured?** Each source (one published data set on one platform) gets its own term in the model, what that laboratory's
chemistry and pipeline add to every cell it measured. The term is measured, not left in the noise: on ten cell types measured both by
sequencing and by arrays, the transfer between platforms was one line for every cell type (slopes 1.03-1.09), not a per-cell difference
(book, atlas chapter). So the same cell type from two laboratories is one cell with two source terms, not two entries; a sample-level twin
test removes look-alikes and sorting impurities; and a cell type never measured at a site is left blank, never filled.

**What does it do that a marker reference does not?** Marker references hold one number per cell type at a few thousand sites chosen to
tell cell types apart, built to answer how much of each cell is in a sample. This atlas holds every array site with data, about 814,000,
on the same calibration a patient's array goes through, and answers a second question as well: how faithfully each cell type is holding
its own pattern, with a stated uncertainty.

**Is it finished?** No. It is version 2 of a reference meant to grow:
- every 6 months a new search for purified healthy cells (enforced by the push gate; latest above);
- each new set split before use into data that builds the atlas and data held back to test it;
- each addition a new version with its own DOI, re-commissioned before the chain uses it;
- first priorities: purified lung fibroblasts and lung smooth muscle (the lung tissue tests need them), second and third laboratories for
  cell types that have one, and the cell types waiting for a second donor.

## Get it

The fitted atlas is 4.67 GB (2,100 files: 700 blocks of sites, each with a summary, 20 posterior draws and its prior). It is too large for
git. **Zenodo DOI: to be added on deposit.** Every file's SHA-256 is in
[`v2/postbuild/records/ATLAS_V2_OUTPUT_MANIFEST.json`](v2/postbuild/records/ATLAS_V2_OUTPUT_MANIFEST.json); the fetch script checks each
one. Per-cell tables for chosen cells are made with [`tools/extract_posterior.py`](tools/extract_posterior.py).

## Commissioning

The atlas is commissioned use by use, on purified cells from laboratories that did not build it:
[`../doors/DEV_ATLAS_COMMISSION_01.md`](../doors/DEV_ATLAS_COMMISSION_01.md).

| use | status |
|---|---|
| U1 reference profiles | held-out check passed on the atlas's own sources; independent laboratories to come |
| U2 sites for tissue tests | development; first to be commissioned |
| U3 blood composition | development; CD4 and CD8 fractions not yet within the bar |
| U4 per-cell identity sites | development |

## In this folder

| folder | what |
|---|---|
| [`v2/`](v2/) | the build: scripts in run order, inputs, records, environments |
| [`v2/postbuild/`](v2/postbuild/) | tests run on the finished atlas, with scripts and records |
| [`tools/`](tools/) | reading the published atlas |
| [`sources/`](sources/) | source notes |
| [`healpix_mapping/`](healpix_mapping/), [`external_manifests/`](external_manifests/) | site-to-sky map and array manifests the chain uses |

## Removed

The class-era atlas (IAMAtlasREBUILD, v0.1: 115 cells in 8 classes, built 2026-05-28) and its class archives and build scripts were removed
from the current tree on 2026-10-12 (author decision): no current reading uses them. They remain in git history at commit `70c0e7b`. Its
lesson file, [`IAMAtlas_FLATNESS_LESSON.md`](IAMAtlas_FLATNESS_LESSON.md), stays as a frozen record.
