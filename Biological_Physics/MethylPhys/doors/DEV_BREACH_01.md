# DEV-BREACH-01 — where past Normal does a lesion become dangerous? (sealed before any array was read; SCORED ONCE 2026-10-13)

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

**Planted test, run once before the seal** (`data/DEV_BREACH_01/planted_breach_01.py`, bars first; output `planted_breach_01_summary.txt`):
on spent TCGA breast normals, plate-tared, 40 against 46, 1,000 draws: nothing planted meets B1 in 5.0 % and B2 in 2.1 %; a 1 % loss
meets B1 in 100 % and B2 in 96 %. All three bars met.

**The seal** (`score_breach_01.py seal`, metadata only; `sealed_breach_01.json`, with the scorer's sha256): the 86 untreated lesions, the
6,000 blind sites, and the slide halves for B2. The scorer refuses to score unless the seal is committed unchanged.

## Outcome, scored once (2026-10-13): the breach is NOT visible at DCIS on this reading

`score_breach_01.py score` on the 86 sealed untreated lesions (196 arrays calibrated, Stage 1 unchanged, box run); outputs
`scored_breach_01.json`, `scored_breach_01_arrays.csv`.

| | result | bar | verdict |
|---|---|---|---|
| B1 progressors above non-progressors | median slide-tared reading 1.0058 against 1.000, p = 0.52 | p < 0.05 | **NOT MET** |
| B2 a line from half the slides holds on the other half | 11 of 23 progressors above it, 12 of 27 non-progressors, p = 0.52 | p < 0.05 | NOT MET |

No line is reported. The planted test showed the design finds a 1 % fidelity loss every time, so this is not a weak test: at the DCIS
stage, on the composition-blind average, lesions that later became invasive cancer read the same as lesions that did not.

**What this does and does not say.** It is one cancer, one series, one stage and one reading: the AVERAGE disorder at sites no cell mix
can move. It does not test the per-site surface share, the clustering meter, or single-molecule discordance, which is where the
two-phase picture (HYP-SATURATION-01, falsifier 7) puts the breach. Every lesion here is already a lesion, so all of them may sit past
the point this reading can separate. The next breach test should read one of those other quantities, or a series with healthy tissue
and earlier lesions in it.

**In plain words, the result.** We asked whether early breast lesions that later turned into cancer already read different from ones
that never did. On this reading, they don't: the two groups are the same. The test was strong enough to see a small difference, so this
is a real answer, for this kind of lesion and this one measurement.

**Prior reading:** GSE281307 was read once, by this test. GSE287331 is named only for its slide layout (spent by DEV-FIELD-01); no array of it is used here. Only the atlas-derived site list of DEV-TISSUE-BREAST-01 is reused, no array.

**In plain words.** Early breast lesions are watched or treated, and nobody can yet say which ones will turn into cancer. This series
followed women for years and recorded which lesions did. If our reading of the lesion's DNA is higher in the ones that later became
cancer, that is the first measured sight of the breach line: the point past Normal where watching should turn into acting. We exclude
women who were treated, because treatment changes the outcome, and we compare arrays only against others on the same slide.
**What it would take** next: a breach test on the per-site surface share or the clustering meter, or a series with earlier lesions and healthy tissue, sealed first.

**Synthetic test:** DEV_BREACH_01/planted_breach_01.py, DEV_BREACH_01/planted_breach_01_summary.txt

**Not a milestone:** scored once; B1 and B2 not met.
