# DEV-SYNTH-LEVERS-01 — synthetic driven-bit molecules: do the levers move IAM-A as the physics says? (development, 2026-10-09)

**Model.** Molecules of 8 CpGs copied division after division: a methylated site is lost with probability f, an unmethylated site
restored with probability g (independent sites). Read with Stage Q's exact rule (≥ 6 calls, ≥ 80 % methylated, isolated loss with both
neighbours methylated), optionally with instrument errors. 200,000 molecules, 80 divisions, from fully methylated.

| test | result |
|---|---|
| 1. what the rule reads | depends only on g/f (f,g = 0.02/0.24 and 0.01/0.12 both ε 0.050; 0.005/0.24 and 0.01/0.48 both 0.018). E from ε = ln(g/f) + a computable offset (+0.26 at ln(g/f) 3.18; +0.46 at 2.49; +0.14 at 3.87) from the rule's conditioning |
| 2. SAM lever (g ∝ S/(Km+S), Km 4.4 µM, from 20 µM) | 2×, 4×, 8× lower SAM: IAM-A_rel 1.10, 1.26, 1.50; ΔE −0.13, −0.31, −0.56 kT vs plain ln(g ratio) −0.17, −0.43, −0.82 |
| 3. instrument (0.5 % false loss, 1 % non-conversion) | ε 0.0312 → 0.0339; subtracting the false-loss rate over-corrects (0.0289): correction must use the model, not subtraction |
| 4. divisions | from fully methylated, steady state in ≈ 1/(f+g) ≈ 4 divisions (ε 0.009, 0.023, 0.029, 0.031 at 1, 4, 8, 16) |
| 5. molecules needed | SD of ε 3.2 % (5,000 qualifying molecules), 1.3 % (20,000), 0.8 % (80,000) |

**Lessons, used before any real data.** (a) Predictions must go through the rule's own mapping (computed, not fitted); the plain
ln(g/f) overstates the SAM response by ~30–45 %. (b) Instrument correction from a spike-in must be model-based. (c) Adult cells are at
steady state within a few divisions, so the generation-series prediction is: **ε constant across generations while the rates are
constant**; drift means the rates changed. (d) Molecule counts do not limit any test (WGBS gives millions); the limits are instrument
and laboratory offsets and, for temperature, between-animal variance.

## Simulation 2 — decitabine-like exposure (loss rate raised for 3 divisions), both readings on the same molecules
| loss during drug (× normal) | IAM-A | Met-A | share of molecules IAM-A can read |
|---|---|---|---|
| 1 | 1.00 | 1.00 | 100 % |
| 3 | 1.48 | 1.71 | 90 % |
| 6 | 1.87 | 2.47 | 70 % |
| 12 | 2.22 | 3.40 | 36 % |
| 25 | 2.46 | 4.10 | 7 % |
| 40 | 2.66 | 4.02 | 1 % |
(Toy sites near β ≈ 0.97; real identity sites differ, so only the shapes carry over.) **Lessons:** IAM-A saturates once most molecules
fall below the ≥ 80 %-methylated rule (it then reads only surviving molecules); Met-A turns over once sites pass β = 0.5.

## Simulation 3 — how large a disease change Met-A can detect (identity sites, real frozen means; 15 vs 15, healthy SD 0.020)
| identity sites changed | loss at those sites | Met-A shift | power |
|---|---|---|---|
| 0.06 % (≈ a 300-CpG published signature, if random) | 15 % | +0.0003 | 0.05 |
| 1 % | 15 % | +0.006 | 0.20 |
| 5 % | 10 % | +0.022 | 0.89 |
| 10 % | 10 % | +0.045 | 1.00 |
| all | 1 % | +0.055 | 1.00 |
**Lesson.** Met-A reads the neutrophil's identity pattern. It detects disease that shifts ≥ ~5 % of those sites by ≥ 10 %; a disease
signature confined to a few hundred genes elsewhere (e.g. interferon genes) is invisible to it by design.

## Simulation 4 — a genome-wide reading for diseases outside the identity pattern (400,000 sites, 15 vs 15, heavy-tailed noise,
whole-array contrast/offset sized to the commissioned 2–2.5 % array spread)
| disease sites changed (Δβ 0.2) | power, mean entropy over all sites (same-run tared) | power, deviation load* |
|---|---|---|
| 0 | 0.00 | 0.00 |
| 300 | 0.00 | 1.00 |
| 1,000 | 0.17 | 1.00 |
| 3,000 | 0.33 | 1.00 |
*Deviation load = number of sites where the array (after self-taring its contrast and offset to the reference means) departs from the
healthy cell's frozen per-site mean by > 5 healthy SD: the sites carrying information the healthy cell does not. Averaging entropy over the
genome dilutes a focal disease to nothing; counting departures does not. A new reading: must be simulated on real healthy arrays (false-load
rate), then commissioned. (Epimutation-load counts exist in the literature; the per-cell frozen healthy reference and same-run tare are the
chain's.)

## Simulation 5 — the two instruments tell mechanisms apart
| disease mechanism | IAM-A | Met-A |
|---|---|---|
| healthy | 1.00 | 1.00 |
| copier 1.5× worse (loss rate up) | 1.26 | 1.33 |
| copier 2× worse | 1.42 | 1.61 |
| 10 % of cells switched state | 1.00 | 2.36 |
| 25 % of cells switched state | 1.01 | 3.52 |
**A copying failure moves both; a change of cell state moves Met-A only.** The pair reads the mechanism, not only that something changed.

## Simulation 6 — the SAM (energy-supply) test design (repeatability SD 0.009; baseline SAM 20 µM, Km 4.4 µM)
| SAM drop | IAM-A | power n = 2 | n = 3 | n = 5 |
|---|---|---|---|---|
| 1.25× | 1.024 | 0.55 | 0.85 | 0.99 |
| 1.5× | 1.042 | 0.90 | 1.00 | 1.00 |
| 2× | 1.087 | 1.00 | 1.00 | 1.00 |
| 4× | 1.258 | 1.00 | 1.00 | 1.00 |
The lever weakens if nuclear SAM sits far above DNMT1's Km (halving from 60 µM: 1.037; from 200 µM: 1.005). **Design:** ≥ 3 per group,
a ≥ 1.5× measured SAM drop, measured SAM/SAH reported with the sequencing; the prediction is computed from the measured SAM change.

## Simulation 7 — the deviation load on REAL healthy arrays (450K neutrophils, 390,939 sites; 2026-10-09)
Reading: number of sites where a self-tared array departs from the healthy mean by |z| > cut, z on a shrunk per-site healthy SD, after
removing the array's own offset and contrast. Disease = a signature of k sites moved by Δβ added to a real healthy array.
**Against another laboratory's reference (GSE88824, 8 arrays): unusable.** Healthy arrays of GSE124565 already depart at a median 6,990
sites (max 10,889) and GSE224807 at up to 28,006: laboratory differences swamp any focal disease (caught 1/12 even at 3,000 sites).
**Against same-run healthy references (leave-one-out within GSE124565, 12 arrays):**
| cut | healthy load | 300 sites, Δβ 0.2 | 300 sites, Δβ 0.1 | 1,000 sites, Δβ 0.1 |
|---|---|---|---|---|
| 5 | 360–1,208 | 2/12 | — | — |
| 6 | 245–672 | 3/12 | — | — |
| 8 | 150–302 | 12/12 | 10/12 | 12/12 |
| 10 | 113–186 | 12/12 | 10/12 | 12/12 |
**What it sets.** The reading works only against same-run healthy references (the Stage T rule again), with a strict cut. The cut (8)
was chosen on these 12 arrays, so it is a development value: it must hold on a laboratory not used to choose it (false-trigger rate on
its healthy arrays, then a constructed signature), before any disease is read with it. Synthetic signatures are random sites; real
disease signatures cluster in regions, which the C-score reads.

## Search and simulate-first for the SAM and temperature levers (2026-10-09; nothing downloaded)
**SAM lever: GSE77079** (mouse liver RRBS, one laboratory, raw reads public SRX1539708-…): Mat1a knockout, liver SAMe depleted, placebo
(6); knockout given SAMe (5); wild type (8). Reference-free reader (rrbs_iama.py) applies. Simulation (restore ∝ SAM/(K_m+SAM),
K_m 4.4 µM, read by the chain's rule): IAM-A rises 1.025 / 1.05 / 1.10 for a 1.5 / 2 / 3-fold SAM drop at 60 µM, 1.13-1.25 at 20 µM.
Power, knockout vs wild type with the within-species spread of the 580-species liver reads (SD of ln ε 0.136): 0.42 at IAM-A 1.10,
0.97 at 1.25. **Decision rule before download:** the paper's measured liver SAMe fold drop and the wild-type within-laboratory spread
must give power ≥ 0.8; the SAMe-treated knockouts are the built-in reversal (they must move back toward wild type). Confounds: the
knockout develops steatohepatitis (cell mix, proliferation).
**Temperature, one species: GSE199815** (Syrian hamster liver WGBS, 3 euthermic, 3 late torpor, 3 early arousal). The two pictures of
the bit predict opposite outcomes: a passive bit held at the current body temperature would lower ε by ~30 % in torpor (≈ 30 K colder;
power 3 v 3 = 0.67); a driven bit renewed at copying predicts almost no change during a torpor bout, because liver cells barely divide
in it (power to see the small residual 0.06-0.07, i.e. a null). A clear fall in torpor would favour the passive picture; no change is
what the driven-bit derivation (DEV-FLOOR-HEIGHT-02) predicts. Labelled PREDICTION (driven bit: no torpor change; change only after
renewal). Needs a hamster read pipeline and conversion control; 9 WGBS runs.
**Not useful:** GSE152444 (sea bass; 4 K during development, read three years later: a memory test, and 4 K is below detection).
