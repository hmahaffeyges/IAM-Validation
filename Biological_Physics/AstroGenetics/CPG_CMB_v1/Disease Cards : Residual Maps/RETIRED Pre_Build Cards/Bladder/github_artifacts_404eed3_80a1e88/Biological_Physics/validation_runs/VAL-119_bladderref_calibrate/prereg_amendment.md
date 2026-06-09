# VAL-119 — Pre-Registration Amendment (atlas SHA correction)

**Amendment ID:** VAL-119_AMENDMENT_001
**Original prereg:** `prereg.md` SHA-256 `04d9d0d36faaf3bc6051c5f852f3bcb9d2ab48c437982b714ff8573d4cad178a` sealed `2026-05-01T03:35:46Z`
**Amendment timestamp:** [SEAL_TIMESTAMP at amendment seal]
**Amendment SHA:** [computed at amendment seal time]

---

## What is being amended

The atlas SHA-256 referenced in `prereg.md` is being corrected. **No biological content, no thresholds, no outcomes, no cohort assignments are changing.** The change is purely a serialization fix to the bridged CSV format so that the calibration script's `load_atlas` function can parse NaN cells correctly.

---

## Why

The bridge script `bridge_bladderref_to_array.py` was written to mirror the prostate `bridge_prostateref_to_array.py` template but used `pandas.DataFrame.to_csv()` with default na_rep behavior (empty string for NaN cells). The prostate bridge output, by contrast, contains the literal string `"nan"` for NaN cells. The val_calibrate.py script's `load_atlas` function expects to parse the literal `"nan"` (which Python's `float("nan")` accepts) — empty strings cause `ValueError: could not convert string to float: ''`.

This is a serialization-format compatibility bug between the bridge and the calibration script, NOT a difference in atlas content. Source `BladderRef__mrefBladder_m.csv` (163 rows × 5 columns + weight, 29 of which contain NaN) is unchanged. Source SHA-256 `f73fbeab74dfbe5aec2829303757908df569bb969101180c2875a46505a3e758` is unchanged. The 5 source EIDs without 450K CpG mapping (1880, 2252, 26521, 51699, 54829) are unchanged. The 2,696 bridged CpG count is unchanged. The 158 unique EID coverage is unchanged. The 4 cell-type list (EC, Epi, Fib, IC) is unchanged. The CHK-3.1C dedup PASS gate is unchanged.

The fix is a one-line bridge script change: pass `na_rep='nan'` to `to_csv()` so NaN cells serialize as the string `"nan"` rather than `""`.

---

## What changes

### Bridge script
- `bridge_bladderref_to_array.py` line modified at the `bridged.to_csv(...)` call to add `na_rep='nan'`.

### Bridge output file SHA
- **OLD bridge output SHA-256** (in original prereg.md): `26b7ee3cb7254e28c1dab5bb4bd2c405f35c46f856f429b40aeab087d7f2ca16`
- **NEW bridge output SHA-256** (after na_rep fix): `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`

### val119 calibration script
- `val119_bladderref_calibrate.py` constant `ATLAS_SHA` updated from `26b7ee3c...` to `3005663b...` to match the corrected bridge output. No other change to the script.

---

## What does NOT change

| Item | Status |
|---|---|
| Source `BladderRef.rda` SHA | unchanged: `a357383a492ebd6ec6262cb0bfba45f970c6a266ef2a1b83f813f31164a42135` |
| Source `BladderRef__mrefBladder_m.csv` SHA | unchanged: `f73fbeab74dfbe5aec2829303757908df569bb969101180c2875a46505a3e758` |
| Source `probeInfo450k.rda` SHA | unchanged: `1b4d0bb8ebd0de3a5bd8b1c9cbf170599fce920da399076182070bdd93b57ca8` |
| Bridged CpG count | unchanged: 2,696 |
| Bridged unique EID count | unchanged: 158 of 163 |
| Cell types | unchanged: EC, Epi, Fib, IC |
| H_min anchors | unchanged |
| Calibration cohort | unchanged: TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 |
| CHK-3.1A/B/C thresholds | unchanged |
| Pre-locked outcomes O1/O2/O3/O4/O5 | unchanged |
| Tissue-floor-dominated threshold (range < 0.02) | unchanged |
| RNG seed | unchanged: 20260420 |
| Pre-registered audit chain | unchanged |

---

## CCL-041 compliance statement

The original prereg `04d9d0d3...` was sealed BEFORE any β file was read. No β file has been read since the original seal. The val119 calibration script execution was halted at `load_atlas(...)` with a parse error before any β file load was attempted — the script verified atlas SHA, ran CHK-3.1C dedup audit, then errored at atlas parse. **No data has been observed under the original prereg.** This amendment seals BEFORE re-execution and BEFORE any β file is read.

This is the same amendment pattern as VAL-117 prereg_amendment (CHK-3.1B threshold correction caught BEFORE re-execution). The strict CCL-041 discipline is preserved.

---

## SHA-256 of this amendment

To be computed at amendment seal time and recorded in `PREREG_AMENDMENT_SEAL.txt` before val119 re-execution.

---

**No outcome may be added post-hoc. No threshold may be relaxed post-hoc. No exception under CCL-041.**
