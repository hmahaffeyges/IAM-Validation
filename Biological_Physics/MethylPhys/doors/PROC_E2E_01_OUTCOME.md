# PROC-E2E-01 — outcome: the commissioned chain reproduces the test package, and the run found three defects

**Sealed 2026-09-23** against the bars fixed in [`PROC_E2E_01_PREREG.md`](PROC_E2E_01_PREREG.md) before any
array was read. Nine IDAT pairs, four cohorts, two platforms, three substrates, each through
[`run_sample.py`](../chain/MethylPhys_Interface/run_sample.py) exactly as a reviewer would type it.
Evidence: [`PROC_E2E_01_score.json`](../kit/results/PROC_E2E_01_score.json),
[`PROC_E2E_01_calibration_vs_cache.json`](../kit/results/PROC_E2E_01_calibration_vs_cache.json),
[`PROC_E2E_01_geo_metadata.json`](../kit/results/PROC_E2E_01_geo_metadata.json), and the scripts
[`PROC_E2E_01_run.py`](../kit/PROC_E2E_01_run.py) / [`PROC_E2E_01_score.py`](../kit/PROC_E2E_01_score.py).

## Verdict

| bar | result |
|---|---|
| B1 intake behaves | **met**, 9 of 9. The five whole-blood and two adenoma arrays PROCEED; the two GSE166212 arrays QUARANTINE with `INCOMPLETE_MANIFEST:declared_sex,declared_chronological_age`, which is correct — their cohort publishes neither. |
| B2 calibration is stable | **met**, 9 of 9 **bit-identical**: max abs difference 0.000e+00 on every shared locus, 415,863 to 757,003 loci per array. The package's two spare arrays have no IDATs here and are **not assessed**, named. |
| B3 composition reproduces | **met**, 11 of 11 documented fractions, every one to **0.0 percentage points**. |
| B4 substrate claims | **met**, all four: secretory 0.000 % in all five whole-blood arrays, 12.2–24.8 % in all four tissue arrays, secretory 12.9 → 24.8 % and cycling 35.7 → 48.1 % from stage 1 to stage 4, and both EPIC arrays calibrate to the documented counts (490,390 against ~490 K; 601,825 against ~602 K). |
| B5 adjudicator fix | **half met, half not assessable — stated as such.** GSM2333950 reads d = 1.022, inside the band: the false d = 42.9 does not return. The other half cannot be assessed: **stem_adult has no commissioned band on this chain**, so it is not gauged, and there is no A to compare with the documented 1.101. Its fraction is 4.0 %. This is a commissioning gap, not a disagreement. |
| B6 report renders | **met**, 9 of 9: 18 tabs, the sky plate, the Stage 0 record naming its decision, the Troubleshooting tab at 9 KB, 6.36 MB per report. |
| B7 A-scores (recorded, cannot pass or fail) | Only the immune class is gauged, and only where a laboratory zero exists. GSE288652 and GSE166212 have none, so **seven of the eight documented A-scores cannot be compared at all** — the chain reports them as not reportable rather than printing a number. The one comparable value differs by surface: immune on GSM8772491 reads A_mapped 1.0922 against a documented 0.815, identity loci against marker union. |

**The headline: every measurement the package documents, the chain reproduces — and the deconvolution
reproduces exactly, not approximately.** Eleven class fractions across three tissue arrays, each to the
decimal the record states.

## Three defects the test found, and what each would have cost

**1. The detection background was estimated with mean and standard deviation, and EPIC negative controls
have outliers.** Measured on the package's own arrays:

| array | NEGATIVE controls | median | p99 | max | mean / sd | median / MAD | detection |
|---|---|---|---|---|---|---|---|
| GSM2333901 (450K) | 613 | 324 | 615 | 801 | 343 / 97 | 324 / 83 | 0.9998 either way |
| GSM8772491 (EPIC) | 411 | 134 | 382 | **29,110** | 235 / **1,467** | 134 / **34** | **0.5041 → 0.9999** |
| GSM8772492 (EPIC) | 411 | 141 | 356 | 28,575 | 220 / 1,401 | 141 / 36 | 0.8153 → 0.9998 |

Half a per cent of EPIC negative-control addresses carry real signal. With mean/sd that lifts σ to 1,676 and
**fails half the probes on a good array** — both EPIC arrays were quarantined on detection and call rate
before the fix. The estimator is now median and MAD-scaled σ, which agrees with mean/sd on a clean 450K array
(0.9998 against 1.0000), so this is a construction fix and no threshold moved. Both estimates are returned in
`neg_control_stats` so nothing is discarded silently.

**What this does to PROC-STAGE0-02.** Its published distributions were computed with the old estimator. Spot-
checked on twelve Uppsala arrays: eleven rose or stayed equal, one fell by less than 1e-5, and every array
remains at or above 0.9991 against a 0.99 bar — so the sealed verdicts stand, every array that passed still
passes, and no reading changes. The published distribution should nonetheless be re-derived on the robust
estimator before it is quoted again: **Future Goal, GATE 1.**

**2. A free-text specimen description crashed the report.** `--specimen "colorectal tumour tissue"` reached
report prose and the vocabulary guard refused to render — correctly, because a condition name has no place in
a reading, but the failure mode was a traceback rather than a message. The laboratory's own words now stay in
the custody record and only a controlled token (`whole_blood`, `plasma_cfDNA`, `tissue`, `unknown`) reaches
the bundle the report renders from.

**3. `--age` accepted only whole years.** GEO publishes `72.0` as readily as `72`, and the runner exited with
an argparse usage error on every array in the first attempt. The flag now takes a float.

## A metadata correction, recorded before the run

The test package describes its three Uppsala arrays as "healthy 58M", "healthy 67F" and "healthy 43M". GEO's
own sample characteristics say **GSM2333901 = 72 M, GSM2333905 = 74 M, GSM2333950 = 81 M**. This run used the
published metadata. Any age-corrected number from the June 2026 run was therefore corrected against the wrong
age — which is one reason a per-array A from that era should not be compared with one from this chain.

## What this run does not establish

It does not test the tissue substrates against a healthy reference, because none exists for them: both tissue
cohorts lack a laboratory zero, so their readings are composition and per-cell only, with placement and tier
withheld. It says nothing about disease detection — no bar here concerns it. And the RA case and its control
are a pair of arrays, not a cohort: GSM1051525 reads immune A″ 1.0104 and GSM1051533 reads 1.0237, both
NORMAL and in band, which is a statement about two specimens and nothing more.
