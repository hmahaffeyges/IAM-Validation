mkdir -p tumour && cp hpc/38699005-d325-407e-9477-345a2a087bfc/{tumour_pairs.csv,tumour_readings.csv,tumour_score.json} tumour/ && cp remote_jobs/tumour/score_tumour.py tumour/ && cat > tumour/PROC_TUMOUR_01_OUTCOME.md <<'EOF'
# PROC-TUMOUR-01 — outcome (2026-10-01). Scored exactly as pre-registered (sha 5ab460cf5af368d5)

**Pre-registration error, recorded first.** The pre-registration said 7 early-onset CRC patients had both tumour and adjacent normal. In ENA
(PRJNA1198593), patient 7 has a normal sample only. The tumours deposited are patients 1–6 and 8–10, and patient 8's run is empty. There are **6 true pairs**.
The scoring code ran unchanged. CRC7 appears in the table as normal-only.

**Statistic.** Copy error on single molecules: an isolated unmethylated CpG between two methylated neighbours, on reads with ≥ 6 CpGs and ≥ 80 %
methylated. ε_corr subtracts the sequencing-error rate. Per-patient genotype mask. 20 M reads per sample, our alignment (Bismark, GRCh38).

## Early-onset colorectal cancer, tumour vs the same patient's adjacent normal (WGBS)

| patient | ε_corr normal | ε_corr tumour | ratio | conversion-failure difference | instrument |
|---|---|---|---|---|---|
| CRC1 | 0.03514 | 0.04668 | 1.328 | 0.0027 | ok |
| CRC2 | 0.03550 | 0.04269 | 1.203 | 0.0009 | ok |
| CRC3 | 0.03268 | 0.03865 | 1.183 | 0.0032 | ok |
| CRC4 | 0.03275 | 0.03646 | 1.113 | 0.0001 | ok |
| CRC5 | 0.03715 | 0.03963 | 1.067 | 0.0051 | limited (> 0.005) |
| CRC6 | 0.03866 | 0.04216 | 1.090 | 0.0001 | ok |

- **P1** (tumour > normal in ≥ 6 pairs): **6 of 6** true pairs. Met as written.
- **P2** (median ratio ≥ 1.10): **1.148**. Met.
- **P3** (instrument): 5 of 6 pairs have a conversion-failure difference < 0.005; CRC5 is just over (0.0051). On the 5 clean pairs: 5/5, median ratio 1.183.

## Oral squamous cell carcinoma (descriptive)
- WGBS: tumour higher in **4/4**, median ratio 1.090.
- oxWGBS (5mC only, 5hmC removed): tumour higher in **3/4**, median 1.059. OSCC2's oxWGBS tumour has only 76k qualifying molecules and reads 0.85.
  So most of the excess survives with 5hmC removed: it is in the 5mC copy itself.

## What this shows and what it does not
- In every patient with a true pair, the tumour's methylated pattern carries more copy errors than the same person's adjacent normal tissue,
  by 7–33 %. This is read directly from molecules, with no reference population and no classifier.
- Not shown: why. Tumour purity is unknown. Adjacent normal can carry field effects, which would make the difference smaller, not larger.
  The tissue mix also differs between tumour and normal (stroma, immune cells).
  6 pairs from one lab and 4 from another. The reading is the copy-error channel (IAM-A input). The IAM-A floor is not yet decided (separate comparison in progress).
EOF
echo ok