# PROC-FOREIGNSCORE-01 — outcome: NOT ADOPTED. No scoring floor is set; a detected foreign cell stays "detected, fraction f, not scored".

Scored 2026-09-27 against [`PROC_FOREIGNSCORE_01_PREREG.md`](PROC_FOREIGNSCORE_01_PREREG.md). Script `kit/PROC_FOREIGNSCORE_01.py`;
results `kit/results/PROC_FOREIGNSCORE_01.json`. 12 healthy GSE87571 hosts × 6 foreign cells × 5 fractions (2, 5, 10, 20, 50 %) = 360
spikes; the spike is the cell's atlas profile at the loci where it is measured and the host's β elsewhere.

| measured | |
|---|---|
| raw A of the spiked cell inside ±0.05 of its own-profile reading | Kidney and β-cells: only from 10–20 %; Hepatocytes: never; Colon and the neural pair: never (|ΔA| 0.13 even at 50 %) |
| inverted A (dilution line inverted with the solver's f̂) | worse than raw and **non-monotonic in f** (Colon 0.064 at 10 %, 0.164 at 20 %) — a defect in the inversion as written, not a physical result |
| detection of the spiked cell | epithelia 100 % from 5 %; the neural pair **0 % at f ≥ 10 %** after 83 % at 5 % — a twin label switch (Glia ↔ Cortical) at high fraction, to be confirmed |
| B4 both readings inside 0.02 at f = 0.50 | **FAILED** |
| B5 blood cells' own A unmoved by the spike | **FAILED** — max shift 0.031 |

## What it decides
The fraction confound is as large as FRACTION_AND_A predicted and larger for some cells: a foreign cell detected at 2–5 % of the
DNA cannot be read on its identity loci in whole blood, raw or by this inversion. The chain keeps its current behaviour — a detected
non-blood cell is printed with its fraction and **no A** — and no floor file is written.

## Queued diagnostics (cheap; before any re-run)
1. The inversion arithmetic: the blood expectation at the foreign cell's identity loci must be the *host's own* composition at those loci; check whether it was the class mean, and why the error is non-monotonic in f.
2. The neural pair at f ≥ 10 %: which template the joint fit names; if the twin, the detector reports the family and the test counts the family.
3. **B5, the one that matters for the reading**: which identity loci the blood cells share with each spiked profile, and the size of the shift per blood cell per foreign cell — a real epithelial signal in a patient's blood would move the leukocyte readings by this much, and the report must either correct it or print it.
