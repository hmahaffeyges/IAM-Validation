# System operating limits study (2026-10-17)

**Asked by the author:** re-evaluate every limit on the instrument catalogue; find the bottlenecks; relax what is stricter than the job;
bring forward only what the goal needs. **The goal:** prove the physics works and that a donor can be read against their own reference,
then put the tools in experts' hands. Experts set the clinical lines on the gauge, not us.

**The rule going forward:** every tool gets one line, "The tool's job:", and at most three bars, each testing that job and nothing tighter.
Anything beyond the job is reported as information, labeled, and never gates a release.

## Where the limits were too tight

| Limit | Set by | The tool's job | Finding | Change |
|---|---|---|---|---|
| Deconvolver error-bar bar (B2: true share inside the bar 80 % of the time) | me | report each cell's share in a mixture | Shares met their bar twice (error 0.029, then 0.016, bar 0.03). The error-bar bar went beyond the job. | Commission the tool for shares; report the error bars as information, labeled not calibrated |
| Atlas per-use bars before the gift (U1 per cell-platform, U3 blood composition) | me | be a usable healthy reference | It beat both published references (array-reference, sequencing-atlas) on their own jobs, all three bars each. Per-use labels stand as scored. | No more atlas commissioning before the gift; per-use labels ship as they are |
| Sphere and noise maps before the gift (M01-M03, G05) | plan | (not a reference job) | The sphere construction has not passed its own synthetic test; this is research | Move to after the gift |
| Full guidebook before the gift (G08) | plan | explain the tools | The 13-tool story (origin, use, value to geneticists) is written and live | Ship the story as guidebook v1; the full manual after |
| Second cell types, neurons, age ranges, post-mortem checks (COM-METAA-001..036, COM-IAMA-013..050) | plan | (beyond the first proof) | None is needed to prove the physics or the same-donor reference | Park until IAM-A is commissioned and the gift ships |
| Physics derivations not needed for the proof (PHY-051, 057, 060, 070-074) | plan | (beyond the first proof) | Interesting; not on the critical path | Park (future work) |

## What stays (the limits that protect the public claims)

- IAM-A's grand test: its sealed rule (at least 0.75 of read healthy samples inside Normal, at least 8 read). This is the instrument's job.
- Synthetic first, the chain, and the manifest: they cost minutes and catch real errors (tonight they caught a disk failure, a wrong folder and a stale page).
- Every scored test keeps its score; nothing is rescored.

## The critical path after this study

1. **Tonight:** IAM-A grand test scored (11 of 13 samples counted at 11:07 PM). If met, IAM-A is commissioned.
2. **Then the gift (about a day of packaging, no new science):** verify the existing outputs (F01-F06, C01, G01-G03, G06, G09), the
   Zenodo deposit and its README (Z01), the letter (Z02), the deconvolver commissioned for shares, the story as guidebook v1.
3. **Then present it** to the experts.

## Cost

Both machines are on-demand (m7a.32xlarge and m7a.16xlarge). Once the grand test is scored, both stop; nothing parked needs them.

## The to-do list (author, 2026-10-17)

1. **The first commissioned chain** (OTH-068): one run through every commissioned instrument. Atlas v2.1, the deconvolution tool and IAM-A's self-reference are commissioned but not yet in the live run.
2. **The report** (OTH-069): professional, with the C-score drawn on the HEALPix sphere.
3. **One-button test from a fresh clone** (OTH-070).
4. **Front page and repository** (OTH-071): the commissioned chain front and center; older results to an archive folder.
5. **Book audit, the cellular part** (OTH-072).
6. **The temperature law** (PHY-057, unparked): the physicist's test, alongside the gift.
7. **Gift packaging:** deposit, README, letter (release checklist).
