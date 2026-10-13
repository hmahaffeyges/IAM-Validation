# DEV-BREACH-01 — where past Normal does a lesion become dangerous? (design, 2026-10-13; NOT sealed, nothing read)

**The question (author, 2026-10-13).** "We need a number all cancers do not go below... Its not about detecting advanced cancer, its about
detecting it early enough to act... I want to deliver a warning that can stop it before it crosses the point of no return...that is the
breech line." The breach line has to be measured where the outcome is known: lesions caught BEFORE cancer, followed to see which ones
became cancer.

**Data: GSE281307** (GEO metadata only, `data/DEV_BREACH_01/breach_meta_01.py`, output `breach_meta_01_output.txt`): ductal carcinoma
in situ (DCIS), Illumina 450K, 196 arrays with raw IDATs, on 19 slides. Classified by later outcome: progressed to invasive cancer 93,
did not 92 (plus 7 adjacent normal, 4 normal). Found by the 2026-10-13 data search; not in CANON/data_spent.csv; never read.

**Two confounds found before design, both from metadata.**
1. **Treatment.** Radiation and hormone therapy prevent progression, so a treated "non-progressor" may be a progressor that was stopped.
   The test uses only lesions untreated by both: 40 progressed, 46 did not, on 19 slides, 15 of which carry both groups.
2. **Slides.** Unlike GSE287331, both outcome groups share slides, so every array can be tared to its own slide.

**The reading.** Met-A at the composition-blind breast sites sealed in DEV-TISSUE-BREAST-01 (atlas-derived, so a mix of cell types cannot
move it), each array divided by the median of the untreated arrays on its own slide. Stage 1 calibration unchanged.

**Draft bars, before any seal:**
| | bar | what it tests |
|---|---|---|
| B1 the gauge carries outcome | untreated progressors read above untreated non-progressors (one-sided rank test, p < 0.05) | a breach line can exist on this reading |
| B2 a line that holds | slides split at random into halves A and B (seed fixed in the seal); the line set on A (best separation) puts more progressors than non-progressors above it in B (one-sided Fisher, p < 0.05) | the line is not fitted to noise |

The line itself is reported as a position on the gauge only if B2 is met, with its interval, and labeled as one series of one cancer.

**What breaks it:** B1 not met - at this stage, on this reading, lesions that will become cancer read like lesions that will not, and
the breach is not visible at DCIS in breast.

**Before the seal, still to do:** (1) a planted test on spent arrays (TCGA breast normals, plate-tared): a planted shift of a stated
size in half of them must meet B1 at 40 against 46, and none must not; (2) the split, written into the seal.

**Prior reading:** GSE281307 has not been read. Only the atlas-derived site list of DEV-TISSUE-BREAST-01 is reused, no array.

**In plain words.** Early breast lesions are watched or treated, and nobody can yet say which ones will turn into cancer. This series
followed women for years and recorded which lesions did. If our reading of the lesion's DNA is higher in the ones that later became
cancer, that is the first measured sight of the breach line: the point past Normal where watching should turn into acting. We exclude
women who were treated, because treatment changes the outcome, and we compare arrays only against others on the same slide.
**What it would take:** the planted test, the seal, then one download and read.

**Synthetic test:** planned (TCGA breast normals, planted shift); not written yet.

**Not a milestone:** a design; nothing read, nothing sealed.
