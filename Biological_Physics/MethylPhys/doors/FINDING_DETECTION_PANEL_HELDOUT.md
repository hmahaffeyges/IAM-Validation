# FINDING — the foreign-cell detection panel, held out on 732 healthy blood arrays (2026-09-27)

**Register row B-12: NOT VERIFIED.** detection_panel_v1.json (archived privately) was commissioned on 12 arrays per laboratory (PROC-MF-02/03,
2026-09-26). Run on all 732 Uppsala whole-blood arrays (kit/HELDOUT_2D.py (archived privately), one shard per array, 1 s each; results
`kit/results/HELDOUT_2D_*.{json,csv}`), 720 of them never seen by the panel:

| | measured |
|---|---|
| arrays with ≥ 1 foreign cell "detected" | **354 of 732 (48 %)** |
| arrays the detector itself flags UNSPECIFIC (≥ 10 of 21 templates fire together) | **176 (24 %)** — healthy blood called a "substrate mismatch" one time in four |
| per-cell false-positive rate on the 556 arrays it calls OK | 0.9 % (Breast) to **12.6 % (Colon)**; median ~5 %; design was ~1 % |
| panel σ vs the σ the cohort shows | 0.0012 vs **0.0031** — twelve arrays gave a σ 2.6× too small |

**Structure, not noise.** Detections per array are bimodal: median 0, p90 = 20 of 21. Half the arrays fire on nearly every
template at once. That is a **per-array common mode** — a shift shared by all 21 templates, which is no cell. It is not the
chip, not the SNP tare, not age, not A (all-fire vs none differ in none of these; HELDOUT_2D_per_array.csv (archived privately)).

**Diagnostic (after the measurement, not a bar): remove the common mode.** Subtract each array's median f̂ across the 21
templates before applying the lines (HELDOUT_2D_commonmode_diagnostic.json (archived privately)):

| | raw | common mode removed |
|---|---|---|
| arrays ≥ 10 fires | 176 | **9** |
| Bladder / Breast / Kidney / Left atrium / β-cells / Upper GI | 18–29 % | **1.4–2.9 %** |
| Cortical neurons / Glia / endothelial / adipocyte / prostate / uterus / duct | 21–29 % | 3–7 % |
| **Colon 23 %, Hepatocytes 19 %, Lung 17 %, gastric families 12–13 %, Thyroid 10 %** | | **stay high** |

The cells that stay high are the thin 6,105-locus source family (PROC-COV-01): on healthy blood their f̂ is *biased* positive,
not noisy — a template built on 1.3 % of the array correlates with the blood residual. No line fixes a bias.

## What this decides

1. The detector, as commissioned, does not hold on held-out input and is **not to be trusted on any specimen** until changed.
   On the report the Stage 2d rows stay printed with this finding named; the composition check (PROC-FOREIGN-01) is unaffected —
   it reads the class fraction, not this detector.
2. **Chain change to pre-register (PLAN item 5):** (a) a per-array common-mode term — a shift common to every template is
   subtracted before any template is read (this specimen's own numbers, no population); (b) lines re-measured on the full
   admitted set with a stated quantile, not on 12 arrays — noting that a detection line measured on arrays known to lack the cell
   is the **instrument's noise floor**, the one place a set of arrays legitimately defines a number, and the author decides
   whether that is acceptable or whether the line must come from the array itself (its SNP-probe noise, as the sky's σ does);
   (c) the thin-source cells (Colon, Hepatocytes, Lung, Thyroid, the gastric families, Pancreas) are **NOT DETECTABLE on this
   block** and print as such until atlas v2 gives them full-coverage profiles.
3. The kit test for Stage 2d (PLAN item 5) takes its contract from this file: per-cell FP ≤ 1 % on the OK set, UNSPECIFIC ≤ 1 %
   of healthy arrays, thin-source cells report "not detectable".
