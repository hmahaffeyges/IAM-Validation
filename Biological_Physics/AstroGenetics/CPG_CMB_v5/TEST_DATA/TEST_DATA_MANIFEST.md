# CPG CMB v2 — Test Data Manifest

All testing used real, public GEO IDAT files run through the full chain. Calibration
(Stage 1) used methylprep 1.7.1. In this development container (Python 3.12) a small
pandas-compatibility shim (`pdshim.py`) is required because methylprep 1.7.1 predates
pandas 2.x; the doctor's Python 3.11 environment runs methylprep natively with no shim.
The chain logic is identical either way. Calibrated betas are cached in
`betas_cache.pkl` so the chain can be re-run without re-calibrating.

## Samples used

| GSM | Series | Array | Substrate | Role in testing |
|-----|--------|-------|-----------|-----------------|
| GSM1051525 | GSE42861 | 450K | whole blood | RA case — trajectory + per-cell signal |
| GSM1051526 | GSE42861 | 450K | whole blood | RA case (spare) |
| GSM1051533 | GSE42861 | 450K | whole blood | control — trajectory baseline |
| GSM1051534 | GSE42861 | 450K | whole blood | control (spare) |
| GSM2333901 | GSE87571 | 450K | whole blood | healthy 58M — clean within-band demo |
| GSM2333905 | GSE87571 | 450K | whole blood | healthy 67F — genuine stem_adult elevation demo |
| GSM2333950 | GSE87571 | 450K | whole blood | healthy 43M — adjudicator before/after |
| GSM8772491 | GSE288652 | EPIC (850K) | colon tissue (high-grade adenoma) | secretory-positive + EPIC support |
| GSM8772492 | GSE288652 | EPIC (850K) | colon tissue (high-grade adenoma) | secretory-positive + EPIC support (spare) |
| GSM5065990 | GSE166212 | EPIC (850K) | colorectal carcinoma (stage 1) | secretory-positive — genuine carcinoma |
| GSM5065985 | GSE166212 | EPIC (850K) | colorectal carcinoma (stage 4) | secretory-positive — genuine carcinoma, advanced |

Source: NCBI GEO, `https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM<prefix>nnn/<GSM>/suppl/`.

## What each test established

**Adjudicator fix (Mahalanobis presence gate).** On healthy whole-blood donors, trace
non-substrate classes (stem_pluri, terminal, cycling at <2% abundance) were inflating the
class-level Mahalanobis. With the gate (a class counts only if abundance ≥ 3% AND outside
the NORMAL band [0.95,1.04)): GSM2333950 went from a false d=42.9 "beyond band" to d=0.0
"within band"; GSM2333905 kept its genuine stem_adult=1.101 elevation (d=3.0). False
positives removed, real departures preserved.

**EPIC support.** GSM8772491 / GSM8772492 (EPIC, ~865K probes) calibrated end-to-end
(490K / 602K CpGs after QC) and ran the full chain. EPIC is supported.

**Secretory readability (the key question).** High-grade colon adenoma tissue (GSM8772491)
deconvolved to: cycling 35.4% (A=0.989), immune 25.3% (A=0.815), stem_pluri 16.1% (A=0.601),
**secretory 12.2% (A=0.959)**, terminal 11.2% (A=0.858). Confirmed on genuine colorectal
**carcinoma** (GSE166212, EPIC): stage 1 (GSM5065990) → secretory 12.9% (A=0.970), cycling 35.7%,
immune 34.0%; stage 4 (GSM5065985) → **secretory 24.8% (A=1.032)**, cycling 48.1% (A=1.069, at the
Warburg line), immune down to 21.9%. The secretory and cycling signal both rise with stage and
their A-scores elevate toward the Warburg/breach lines. The chain reads and scores secretory and
cycling cleanly whenever epithelial DNA is present. Secretory reading ~0 in healthy whole blood is
therefore correct (blood carries no epithelial DNA), not a chain limitation.

## Substrate implication (for the report and the paper)

- **Whole-blood leukocyte DNA (buffy coat):** the immune-architecture readout. Detects the
  systemic immune field-effect signature of disease. Epithelial/secretory cells are absent
  by biology.
- **Plasma cfDNA (liquid biopsy):** the shed-tissue readout. Carries epithelial/tumour DNA,
  so the secretory/cycling signal a tumour produces appears directly — the substrate to use
  when the goal is not to miss a solid tumour (CRC, breast) by reading the tumour itself.
- **Tissue:** secretory/cycling resolve strongly (demonstrated above) — the positive control.

The report must state which substrate a given run used so the cell readout is interpreted
correctly and the right tube is drawn.
