# DEV-IAMA-BAND-01: IAM-A's own healthy band (COM-IAMA-054)

**Sealed 2026-10-16 by the author ("Seal as drafted"), before any Hansen count was read.** Manifest row COM-IAMA-054.

**The reading.** A = the sample's measured held error divided by what its own loose share predicts on the chain's self-reference line
(version 1). A = 1 is on the healthy line; above 1 is more holes than its own line predicts (right on the gauge). Chain Stage Q2 `reading()`.

**The band.** The 2.5th to 97.5th percentile of every healthy reading, both laboratories pooled: Loyfer's 153 samples and each Hansen donor
(GSE49627) that passes the sample-shape check (COM-IAMA-053). A refused donor is not pooled. Computed by chain Stage Q2 `build_band()`
(`data/DEV_IAMA_BAND_01/build_band_v1.py` writes `chain/Runtime Matrices/IAM_A_SelfRef/band_v1.json`, stamped). Nothing is chosen by hand;
the laboratory difference sits inside the band by construction. Loyfer alone, already counted, shown to the author before sealing (`data/DEV_IAMA_BAND_01/prior_loyfer.json`, by the chain):
median 0.95, 2.5th to 97.5th percentile 0.785 to 1.055, range 0.72 to 1.068.

**Prior reading:** GSE49627's tile shape was read on 2026-10-15 to plan DEV-SECONDLAB-02; no held error, loose share or reading of it had been computed when this rule was sealed. Loyfer's 153 counts had been read (above).

**Synthetic test:** DEV_CHAIN_Q2_01/q2_synthetic.py, DEV_CHAIN_Q2_01/q2_synthetic.json

(T7: samples planted exactly on a planted line read A = 1; a planted spread from two laboratories returns its known percentiles. MET.)

**What it would take to succeed:** the band is built by the chain from the stamped counts of both laboratories by the rule above. It is a
measurement, not a pass or fail; it becomes IAM-A's Normal mark when IAM-A is commissioned (COM-IAMA-052).

**In plain words:** the healthy range for IAM-A comes from healthy people measured in two different laboratories, not from a number we picked.

## Band v1 (2026-10-16), built by the chain by the sealed rule

`chain/Runtime Matrices/IAM_A_SelfRef/band_v1.json` (stamped; quoted here from `data/DEV_IAMA_BAND_01/band_v1_summary.json`): 156 healthy readings (Loyfer 153, Hansen 3, all three passing the shape check),
**Normal 0.7787 to 1.0551**, median 0.9488. The Hansen donors read 0.849, 0.805 and 0.78, inside.

## 2026-11-20 · Per-type band (the measured substitute for A8) against the pooled band, like for like, each library's donor left out

**Status.** Exploration on spent data (the IAM Sequencing Atlas's own healthy libraries, GSE186458). Not commissioning: commissioning needs unspent healthy data. Script, inputs and outputs: `doors/data/DEV_IAMA_BAND_01/per_type_band_loo.py`, `per_type_band_loo_inputs.csv`, `per_type_band_loo_libraries.csv`, `per_type_band_loo_types.csv`, `per_type_band_loo_summary.json` (anyone can rerun: `python3 per_type_band_loo.py`).

**The question.** IAM-A's healthy reading differs by cell type: about 81 % of its healthy spread lies between types (register A8). One pooled band must hold every type, so its width is set mostly by those type differences, not by noise. A per-type band takes each type's own measured level as its centre. Does that make IAM-A more sensitive without calling healthy libraries abnormal?

**The quantity and its physics (what is derived and what is measured).**
1. IAM-A counts isolated losses on held molecules: a molecule counts if it has at least 6 calls and is at least 80 % methylated; a loss counts only when both neighbours are marked (rule A14). Held error eps = isolated losses over chances.
2. The derived line (`derived_line_v1`) predicts each library's held error: p = f(coupling) + slope x loose share.
   - f is the held error from physics alone, read through the counter. **Derived (A1-A4):** M = dG_ATP / (R T) = 54,000 / (8.314 x 310.15) = 20.94 thermal kicks per ATP; a restore costs two ATP (A2), so the push is x = 2M = 41.9 kicks; the hold is the slow-write rule, held/lost = (x/2) / (1 - (1 - e^-x)/x) = 21.45, so eps0 = 1 / (1 + 21.45) = 0.04454.
   - The slope on the loose share is **measured, partly derived (A7)**.
3. A_derived = eps / p. If IAM's Law described every type completely, a healthy library would read 1.
4. **A8 is not derived.** Healthy types sit at their own level, 0.79 to 1.19 on this line, fixed with age. The per-type band uses each type's **measured** level as a substitute, like a scheduled value replacing a lost measurement in state estimation (a pseudo-measurement). Its basis is measured. It is not a derivation of A8, and it must never be described as one.

**The two bands, each rebuilt without the library being scored.**
- **Donor stand-in.** GEO gives no donor ID, so a library's donor is taken as its GEO lab, age and sex together. Every library sharing all three is removed before the bands are rebuilt.
- **Pooled band** (band v2's own rule): the 2.5th to 97.5th percentile of A_derived over all remaining libraries of all types. NOTE (2026-11-20, from the record): the pooled 2.5-97.5 % percentile rule behind band v1 and band v2 is the OLD rule; the agreed method (author 2026-11-07, 'no fixed band, 1 is the middle') is each type's measured healthy width with readings counted in widths from 1, as stage_metaa_atlas.width_state does for Met-A arrays. The per-type band here follows the agreed method; the pooled band is shown only as the old method it replaces. Band v2 in the chain (0.8744 to 1.1783) was built from 150 of these same libraries, so it is labelled 'in commissioning', not commissioned.
- **Per-type band:** centre m = median A_derived of the remaining libraries of the same type (at least 2 needed). Half-width w = t(0.975, df) x s x sqrt(1 + 1/n), the GUM prediction interval for one new library:
  - s = the within-type relative spread pooled over all types (each library's A over its type mean, minus 1); s = 0.035, df = 94;
  - sqrt(1 + 1/n) adds the uncertainty of a centre built from n libraries;
  - t widens the band for the finite number of libraries.
  - For comparison, the counting uncertainty of one library's A is about 0.0014 (0.14 %), so s is almost all person-to-person and lab-to-lab scatter.
- **Variant B** (s from the type's own libraries only) is also reported. With 2 to 7 libraries per type it is too uncertain, and the pooled s is the right estimator.

**Results.**
- **Scored libraries:** 123 of 152. The other 29 belong to 16 types left with fewer than 2 libraries once the donor is removed: aorta endothelium, bladder epithelium, breast luminal epithelium, colon enteroendocrine, colon fibroblasts, colon macrophages, effector memory cd8 t cells, endometrium epithelium, kidney glomerular epithelium, kidney tubular epithelium, lung alveolar macrophages, memory b cells, naive cd4 t cells, naive cd8 t cells, striated muscle, tonsil pharyngeal epithelium. Their per-type band rests on one library or on a lineage borrow, and must be flagged as such.

| | Per-type band | Pooled band (band v2's rule, rebuilt) |
|---|---|---|
| Healthy libraries inside (95 % band) | 113 of 123 (91.9 %) | 115 of 123 (93.5 %) |
| Width | +/- 8.5 % around each type's own centre | one window, 0.8744 to 1.1783, for every type |
| Smallest held-error rise flagged at the type's centre | 8.5 % in every type | median 10.8 %, range 0.7 % to 46.9 % by type |
| Types where a 10 % rise is flagged | 43 of 43 | 18 of 43 |
| Variant B (own-type spread) | 117 inside, half-width 26.0 % | |

- **Sensitivity is calculated, not fitted.** The prediction p does not depend on held error, so a rise r in held error multiplies A_derived by (1 + r). A library at its type's centre is flagged once (1 + r) passes the band's upper edge: r = w for the per-type band, and r = hi / m - 1 for the pooled band. The pooled band's sensitivity therefore depends on where the type sits. A type near its top edge is flagged by almost any rise: healthy central memory CD4 libraries (centre 1.164) need only a 0.7 % rise, so ordinary scatter already pushes 2 of 3 outside. A type near its bottom edge can take a large rise unnoticed: erythrocyte progenitors (centre 0.802) need a 46.9 % rise, and their healthy libraries already sit below the band (3 of 3 read 'below Normal' while healthy).
- **The misses differ in kind.**
  - The pooled band misses whole healthy types: cortical neurons, erythrocyte progenitors, pancreatic delta, t central memory cd4, t effector cell cd8. All three erythrocyte progenitor libraries read below it and both central memory CD4 libraries above it. That is A8 showing through.
  - The per-type band misses single libraries just past the edge (|z| 2.1 to 3.4): GSM5652223 (cortical neurons, z +2.12), GSM5652245 (pancreatic acinar, z -2.51), GSM5652283 (cd8 t cells, z -2.35), GSM5652288 (t effector cell cd8, z +2.40), GSM5652290 (t effector cell cd8, z -3.42), GSM5652291 (t effector memory cd4, z +2.15), GSM5652316 (b cells, z +2.28), GSM5652318 (b cells, z -2.12), GSM5652339 (prostate epithelium, z +2.33), GSM5652370 (colon epithelium, z +2.17).

**Limits, stated so nobody has to guess later.**
1. **Coverage.** 91.9 % is a little under the 95 % the band is built for, so the within-type spread is slightly underestimated. Candidates: lab-to-lab scatter inside a type (the between-lab term, about -5 %, PHY-084, is not in the band), heavier tails than a normal distribution, or the donor stand-in not removing every library of a donor. One type (CD8 effector T cells) holds misses on both sides (z +2.40, -3.42), so its own spread is wider.
2. **Donor stand-in.** Lab, age and sex may merge two donors, which removes more data than needed and so is conservative. It cannot split one donor. Donor IDs from the atlas's supplementary table (COM-IAMA-039) are owed.
3. **One sequencing source.** All libraries come from one atlas, though from several tissue-source labs. Readings from another sequencing lab need the between-lab term.
4. **Spent data.** These libraries also built the type positions (type_positions_v1.json) and band v2. Leaving the donor out removes the library's own influence on its band; it does not make this commissioning.
5. **A separate, out-of-sample check** exists (register F248). The hold alarm against each type's position, judged with a jackknife floor, read 40 of 41 healthy libraries within, on 41 libraries that built neither band.

**Correction of an earlier statement (2026-11-20, in chat).** I first reported 'per-type 113 of 123 at 8.5 %, commissioned 116 of 123 at 17.8 %; a 10 % rise is caught per type, the commissioned band needs about 18 %'. Three things were wrong:
- band v2 is not commissioned;
- its 116 of 123 was scored in-sample, against a per-type band scored with the donor left out;
- the 17.8 % was the band's upper edge measured from A = 1, which is no type's centre.

The like-for-like numbers are those in the table above.

**Power-system reading (analogy).** The pooled band is one system-wide voltage limit applied to every bus. The per-type band is each bus's own voltage schedule, with a deviation limit set from that bus's measured scatter. Where a schedule cannot be derived, the measured schedule stands in as a pseudo-measurement until it can be.

## 2026-11-20 · The measured substitute for A8, in full: how it is built, the out-of-sample test, and why A8 is not yet derived

This section exists so that a reader can check the substitute without asking anyone. It adds three things to the band comparison above.

**1. How the type positions are built (`chain/Runtime Matrices/Seq_Atlas/law_sphere/type_positions_v1.json`, edit 212).**
- **Data:** the one-button joint-run readings of the IAM Sequencing Atlas's own healthy libraries (GSE186458), 150 libraries in 56 cell types: 42 types with 3 libraries, 10 with 2, 4 with 1 (effector memory cd8 t cells, lung alveolar macrophages, naive cd4 t cells, naive cd8 t cells).
- **Quantity:** each library's A_derived = its held error over IAM-A's derived line (the law's 0.04454 plus the measured slope on the loose share, section above).
- **Position of a type:** the centre of its libraries' A_derived, with the relative uncertainty of that centre at k = 1 (GUM), stored with the number of libraries. Centres run from 0.804 to 1.171.
- **Use in the chain:** the Law Sphere's hold alarm (`stage_law_sphere.hold_alarm`) compares a single-type library's held record with its own line multiplied by its type's position, judged against a measured floor (below). Mixtures are not read this way (NOT_RUN).
- **Basis: measured.** It is a pseudo-measurement, the scheduled value a power-system operator substitutes for a lost meter, used until the value can be derived. It is never described as a derivation of A8.

**2. The out-of-sample test (data: `doors/data/DEV_IAMA_BAND_01/hold_alarm_out_of_sample.csv`).**
- **Libraries:** 41 healthy sequenced libraries in 21 types from the same GEO series that the atlas never used, so **none of them built the type positions**.
- **Reading:** each library's held error on the Law Sphere's units, from its own molecules, against three references: the law alone (0.04454), IAM-A's own line alone, and the own line times the type's position (the substitute).
- **Floor:** a block jackknife over the sphere's units, which carries the measured overdispersion between units (median standard error 0.034 in log), combined by GUM with the type position's own uncertainty. z = log(held / reference) over that floor.
- **Result:**

| Reference | Libraries within abs z <= 2 | Within 3 | Median abs z |
|---|---|---|---|
| The law alone | 24 of 41 (0.585) | 28 | 1.78 |
| IAM-A's own line alone | 9 of 41 (0.22) | 17 | 3.53 |
| Own line x type position (the substitute) | **40 of 41 (0.98)** | **41** | **0.98** |

  The one library past 2 is pancreatic acinar, z +2.12. The substitute predicts unseen donors' held records within their measured uncertainty; the law alone and the line alone do not, which is A8 showing.
- **Earlier step on the same libraries (exploration):** the spread of the sphere's held error over each reference (sd of log) fell from 0.139 (law) to 0.102 (own line) to 0.051 (substitute).
- **Limits:** one sequencing source; the between-laboratory term is not in the floor; 4 types rest on one library; spent data, so this is exploration, not commissioning.

**3. Why A8 is not yet derived: every candidate cause tested so far.** The derived part of IAM-A (the hold from M and the two-coin restore) predicts one held error for every cell. Healthy types sit at their own level, so the law is missing a per-type term (A8). Each row below was tested as that term or is listed to be. Status only: each result's numbers are in the project's findings ledger under the F-numbers given, and are published here only with their scripts, so that no number on this page lacks a committed derivation.

| Candidate cause of A8 | Where it would enter IAM's Law | Status | Findings |
|---|---|---|---|
| Coin size (dG_ATP) | push = coins x M | closed | F026, F195 |
| Temperature | M = dG_ATP / RT | closed | F195 |
| Cell division rate | copy layer | closed | - |
| Age within a type |  | closed | F034, F226 |
| Eraser share by 5hmC (TAB-seq) and TET knockouts | eraser_share | closed | F031 |
| Writer precision by type | D (selection) | closed | - |
| Open-chromatin extent (traffic as extent) |  | closed | F035, F181 |
| Packing (ChromHMM packed share) |  | closed | F037 |
| Double loss, run lengths, damaged minority | coupling, CP1 | closed | F048 |
| Location in late-replicating DNA | copy timing | closed | F168, F170 |
| Tissue oxygen | loss rate | closed | F192, F193 |
| Hormone and sugar uptake (GLUT4, INSR) |  | closed | F204 |
| Synchronous orders (stretch or region rewrites) |  | closed | F200 |
| Replication-independent histone H3.3 |  | closed | F218 |
| Immediate-early (activity) genes |  | closed | F225 |
| Within-type asynchrony |  | closed | F205 |
| Cell lifespan across tissues |  | closed | F224 |
| Donor effect |  | closed | F226 |
| Repeats in unused dense tiles |  | closed | F227 |
| Coupling (losses next to each other), corrected | coupling-A | closed | F200 |
| Duty-cycle crew form |  | closed | - |
| Own-programme dense places |  | closed | F220 |
| Unused islands (written in no type) | write layer | closed | F241 |
| Finite-time price (restore window by type) | the push family: a shorter restore window delivers less of each coin (the author's orders / duty-cycle idea) | closed | F249 |
| Eraser route (islands gain vs held loss) | eraser_share, eraser_clean | lead or met | F195, F196, F228, F240 |
| Crew expression, DNMT1 + UHRF1 RNA | restore rate g | lead or met | F201 |
| Activation history, asynchronous orders | orders retarget the record | lead or met | F002, F013, F198, F200 |
| Slow residue per day spent ordered |  | lead or met | F222, F223 |
| Clock sites (call-up count) | clock layer | lead or met | F235 |
| Dense-over-sparse held error (feeder slope) | hold by CpG density | lead or met | F219, F221 |
| Open-chromatin held floor | hold layer | lead or met | F208 |
| Delivery share (gentleness of restores) | phi, push | lead or met | F210, F211, F212, F242 |
| Processive crew (recruitment saving) |  | lead or met | F216 |
| Whole held record offset (Law Sphere, atlas side) | push by type | lead or met | F240, F242 |
| Input fingerprints on three readings (push, eraser, writer) | held, block gain, island gain | lead or met | F243 |
| One lever at a time in the simulator, with the pair triple's Koide reading | held, strays, U/H/M, Koide Q, angle, delta | lead or met | F244 |
| Coins per restore by type (adenylate kinase away from equilibrium) | coins | candidate (untested) | F240 |
| Restore and loss RATES by type (A8R) | g and f | candidate (untested) | F190 |
| Double-strand (hairpin) snapshot by type | maintenance vs de novo in one sample | candidate (untested) | - |
| Traffic as a RATE (not extent) |  | candidate (untested) | - |
| Crew PROTEIN by type: DNMT1, UHRF1 | restore rate g | candidate (untested) | F201 |
| Replication-independent DNMT1 recruitment (UHRF1 ubiquitylated H3, the backup mode) | restore in resting cells | candidate (untested) | - |
| USP7 (deubiquitylates H3; helps and limits DNMT1) | restore rate g | candidate (untested) | - |
| LIG1 and PAF15 (fork-coupled recruitment) | restore during copying | candidate (untested) | - |
| H3K9me3 coverage (boosts UHRF1) | restore by region | candidate (untested) | - |
| DNMT3A/3B (de novo backup; also at repair sites) | write layer and repair | candidate (untested) | - |
| DNA damage and repair synthesis by type (restore at repair patches) | a loss-and-restore cycle without division | candidate (untested) | - |
| Methylation potential SAM/SAH by tissue (MAT2A, AHCY) | the methyl donor's pressure | candidate (untested) | F026 |
| Adenylate energy charge by tissue | coins per restore | candidate (untested) | - |
| Light (UV) damage written by the cell, partly unrepairable | a record copied forward, not restored | candidate (untested) | F160 |
| Restore speed vs energy vs error (speed-dissipation-error tradeoff) | how gently each restore is made | candidate (untested) | - |
| UHRF1 sensing accuracy at the half-marked site (energy-speed-accuracy of sensing) | the reader that finds a lost mark | candidate (untested) | - |
| Barrier hold vs paid pump (Kramers escape) | whether part of the hold is a wall, not a pump | candidate (untested) | - |
| Opposite-value overwrite (flip) share by type | the price per held mark: erase-then-remark cycles cost about four times a restore | candidate (untested) | - |
| 5hmC in neurons | meter | not assessed | - |

The leads point to the hold itself (how well each type holds even where access is easiest), in the direction of the push family. The derivation route the long form names, the finite-time price of each restore, needs restore speeds measured by healthy cell type, which are not openly available. Until then the measured position stands in, labelled as measured, tested out of sample, and replaced as soon as a derivation passes.
