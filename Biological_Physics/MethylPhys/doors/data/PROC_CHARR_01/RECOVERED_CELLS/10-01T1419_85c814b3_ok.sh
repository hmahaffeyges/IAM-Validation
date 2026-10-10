mkdir -p charr && cp hpc/4acef3b3-c341-46fa-ae67-908f79daaf48/{charr_readings.csv,charr_score.json} charr/ && cp remote_jobs/charr/score_charr.py charr/ && cat > charr/PROC_CHARR_01_OUTCOME.md <<'EOF'
# PROC-CHARR-01 — outcome (2026-10-01). Brook charr sperm, 40 males, scored as pre-registered (sha 1460b52e5f4bb676)

**Data.** 40 males (2017 and 2018; selected and control lines; ambient and +2 °C warm), WGBS paired-end, 5 M read pairs per fish, our alignment.
Copy error on single molecules (the same statistic as the salmon and tumour runs). E = ln((1 − ε)/ε) kT.

**Run faults, recorded first.**
- GSM7094128: only 297 read pairs arrived (84 qualifying molecules). Its download failed, but the driver logged it as ok. Treated as a failed run below.
- GSM7094137: the first attempt read 0 pairs. The re-run got one of its two runs, so it has one half only.
- The driver now refuses to log "ok" without a site table. A partial download is still not caught; the minimum read count needed is added to the to-do list.

| prediction | result |
|---|---|
| **P0** halves agree, ICC ≥ 0.80 | **0.924 on the 36 fish with both halves: PASS.** The scorer printed NaN because 4 fish have one run; computed here on complete fish |
| **P1** the fish, not the library: \|ρ\| < 0.30 with conversion failure, duplicates, mask | conversion −0.01, **duplicates 0.45** (0.57 without the failed fish), mask 0.29: **FAIL** |
| **P2** scale: median ambient fish 3.8–4.3 kT | **3.816 kT** (ε 0.0216): **PASS**, at the low edge |
| **P3** temperature (+2 °C) | warm/ambient ratio 0.92 (95 % CI 0.79–1.05), p 0.24. Underpowered. **Not interpreted (P1 failed)** |
| **P4** line | −0.0015 (CI −0.0045 to +0.0015), p 0.31. **Not interpreted (P1 failed)** |

**What this shows.**
- The instrument repeats itself. Halves of the same fish agree with ICC 0.92, and the fish span only 3.74–3.88 kT (between-fish SD of ε 0.0007).
- Copy error tracks the library's duplicate fraction, so the small differences between fish can't yet be told apart from library preparation.
  That is the same lesson as the Methow steelhead.
- The scale matches across species and labs: brook charr sperm 3.82 kT, Methow steelhead sperm 4.02 kT. Human blood cells sit at about 3.8–3.9 kT (ε ≈ 0.02).
- Next in the fish line: deeper reads or a duplicate-aware statistic (count errors only on molecules with unique start positions per strand pair), before any fish difference is read.
EOF
echo ok