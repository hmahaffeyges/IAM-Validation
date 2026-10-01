# PROC-WARBURG-01 — outcome (2026-09-30). Pre-registration: PROC_WARBURG_01_PREREG.md (sha a4683f03c3ca92de), written before any array was read.

479 EPIC arrays (GSE179847) through our Stage 1 (479/479 ok); 340 evaluable (Seahorse of the same line, treatment and part within 10 days, and a
time-matched standard). Δf_g = glycolytic ATP fraction of the array's culture − that of its three time-matched untreated controls.

| prediction | n | result | verdict |
|---|---|---|---|
| P1 switched (Δf_g ≥ +0.25) read A_meth ≥ 1.07 in ≥ 80 % | 77 | 19.5 % (median 1.014, range 0.974–1.187) | **FAIL** |
| P2 not switched (|Δf_g| < 0.10) read A_meth in Normal in ≥ 80 % | 133 | 97.0 % (median 1.000) | **PASS** |
| P3 A_meth rises with Δf_g, ρ > 0, p < 0.01 (within-line permutation) | 340 | ρ = 0.154, p = 0.018 | **FAIL** (right sign, not significant at the set level) |

Sensitivity (not pre-registered): arrays with call rate ≥ 0.93 only (half the set; median call rate 0.929): P1 11.9 % (n 42), P2 98.6 % (n 69). Same verdicts.

**Read as pre-registered:** P2 passes, P1 fails. The time-matched standard is stable (untreated cultures read 0.998–1.005 across 200 days), and a
metabolic switch on its own does not bring the methylated channel to 1.07. Hypoxia raises the glycolytic share by +0.33 and leaves A_meth at 0.995.
The ~1.08 seen at transformation (BJ HRAS, HBEC) is therefore not explained by the metabolic switch alone. The line is not moved by this test.

**Exploratory (found after reading; not evidence until pre-registered on new data):** under CHRONIC forced glycolysis the methylated channel climbs
with time, while untreated controls stay flat.

| days grown | control A_meth (n) | oligomycin A_meth (n) | mitochondrial-substrate blockers A_meth (n) |
|---|---|---|---|
| ≤ 30 | 0.998 (39) | 1.002 (3) | 1.002 (3) |
| 30–60 | 1.005 (34) | 1.009 (4) | 1.014 (5) |
| 60–90 | 0.999 (19) | 1.056 (5) | 1.094 (5) |
| 90–120 | 0.996 (7) | 1.060 (4) | 1.005 (3) |
| 120–200 | 0.998 (10) | **1.105** (3) | **1.176** (3) |

Candidate hypothesis: the gauge reads the ACCUMULATED cost of running on glycolysis, not the switch itself — error on the methylated sites builds with
time spent in the glycolytic state. Groups are 3–5 arrays; to be pre-registered as a dose × time prediction and tested on an independent series.
