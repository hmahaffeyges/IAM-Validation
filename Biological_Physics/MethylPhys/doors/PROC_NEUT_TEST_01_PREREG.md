# PROC-NEUT-TEST-01 — pre-registration (written 2026-10-01, before any array below is read by chain v3)

**Instrument.** Chain v3 as pushed (commit bc4a651), `run_sample.py --engine v3`, unchanged during this test. Each specimen: IDAT pair →
intake → calibration → composition → Met-A → C-score → report. Whole blood is tared in a second pass, against the healthy references in the
same slide (≥ 3); if a slide has fewer than 3, they come from the same batch (the dataset's own healthy arrays, leave-one-out). Normal = 0.95–1.05. No setting is changed after reading.

**T1 Real healthy whole blood, FACS-counted (GSE112618; 6 adults, one slide).**
- T1a: ≥ 5/6 tared A_rel in Normal.
- T1b: the Stage A neutrophil fraction is within 0.05 of the FACS fraction, median over the 6.

**T2 Isolated neutrophils from one person, repeated (GSE247193, 30-year-old man; GSE247195, 54-year-old man; 8 times of day × 3 arrays each). Another lab.**
Read against the frozen floor with no tare, as the chain rule states.
- T2a: ≥ 80 % of the 48 arrays in Normal. This tests the no-tare rule for isolated cells from another lab.
- T2b: within one person, the SD of A over his 24 arrays is ≤ 0.010.
- T2c: technical replicates (same time point) have median |ΔA| ≤ 0.010.
- Descriptive: whether A changes with time of day.

**T3 Technical variability, whole blood (GSE250556; 64 arrays, 4 men aged 24–66, pooled replicates).**
- T3a: ≥ 90 % of tared A_rel in Normal.
- T3b: the within-subject SD of A_rel is ≤ 0.010 for each subject.

**T4 COVID-19 whole blood (GSE179325; SEVERE, MILD and SARS-CoV-2 NEGATIVE adults).** Severe COVID-19 mobilises immature and activated neutrophils.
Tare against the NEGATIVE arrays on the same slide (≥ 3), or else against all NEGATIVE arrays (leave-one-out).
- T4a: SEVERE individuals read above Normal (A_rel > 1.05) more often than NEGATIVE individuals (one-sided Fisher, p < 0.05).
- T4b: MILD lies between them (descriptive).
- T4c: ≥ 80 % of NEGATIVE individuals read in Normal.
- Descriptive: the C-score by group, and how often A is withheld for neutrophils under 50 %.

**Stated limits now.** T1 is the same lab as the floor and profiles. T2 has two people. The T4 NEGATIVE adults are hospital-tested people, not
healthy volunteers, and lymphopenia in severe disease raises the neutrophil fraction. A difference means the neutrophils' pattern changed in
those people. It does not tell which neutrophil state caused it.

---
**Amendment 1 (2026-10-01, written after reading GEO's FACS counts and before any array was read by the chain; original sha bab2f5e367ac40ad).**
In GSE112618 the FACS neutrophil fractions are 0.34–0.65, and 3 of the 6 are below 0.5. The chain therefore withholds A for those 3. That leaves each
readable specimen with only 2 same-slide references, and the tare needs 3. T1 as written cannot be run. T1 is replaced by:
- T1a: the dominant-cell gate agrees with FACS (neutrophils ≥ 0.5 or not) for ≥ 5/6 specimens.
- T1b: unchanged (the Stage A neutrophil fraction is within 0.05 of FACS, median).
- T1c: the untared A of the readable specimens lies within the offset this lab showed on the known mixtures, 0.93–0.98.
T2–T4 are unchanged.
