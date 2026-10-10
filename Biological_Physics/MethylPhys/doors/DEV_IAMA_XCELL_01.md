# DEV-IAMA-XCELL-01 — commissioning IAM-A without ≥ 3 same-cell donors per laboratory: cross-cell same-run tare and a spike-in standard
(development; written 2026-10-09 before any reading below)

**Problem.** IAM-A carries a laboratory-and-kit offset (DEV-IAMA-KIT-01: same neutrophils, Swift 1.16 vs TruSeq 1.05). The author chose the
same-run tare (≥ 3 healthy references read the same way). Search for open read-level neutrophil WGBS with ≥ 3 healthy donors per laboratory
and kit (2026-10-09; SRA, GEO, ENCODE): none besides Loyfer 2023 (3 granulocytes, the files P is measured on). RRBS sets exist but are not
whole-genome (P is measured on whole files). No public set has them (searched 2026-10-09); controlled-access sets are not used (public data only, 2026-10-10).

**Another way, tested now.** The offset is a property of the laboratory and kit, not of the cell, if it comes from library chemistry. Then
healthy **other cell types read in the same run** can be the references, each read against its own healthy position P_cell.
GSE128731 (laboratory G) sequenced CD4 T cells of two further donors (Sample5, Sample8) on the same kits and sequencers.

**Run (Box Run 2 session 3).** (a) P_CD4 on the three whole Loyfer Blood-T-CD4 files (GSM5652279/80/81), the v2 rule: per donor
H(mean ε of the other two)/H(ε₀), averaged. (b) CD4 runs Swift/NovaSeq, Swift/HiSeqX, TruSeq/HiSeqX of Sample5 and Sample8 (QIAseq left out:
its neutrophils failed Q0 conversion), 25 M pairs each, the pinned pipeline, Stage Q0/Q. (c) pUC19 reads in the six readable neutrophil BAMs
(the index carries pUC19): if the libraries carry a CpG-methylated pUC19 spike, its unmethylated calls are a direct measure of the
technical error on methylated DNA, run by run.

**Checks (bars fixed now).**
1. Shared offset: CD4 A(Swift) ÷ A(TruSeq) (median over donors and sequencers) within 0.03 of the neutrophil ratio from session 2.
2. Cross-cell tare: each of the 6 readable neutrophil runs, tared against the same-kit readings of 3 other healthy people of laboratory G
   (CD4 Sample5, CD4 Sample8, the other neutrophil donor; each A on its own P), reads Normal (0.95–1.05): 6 of 6.
3. Kit gap removed: per neutrophil donor, |tared Swift − tared TruSeq| ≤ 0.02 (twice the measured repeatability 0.009).
4. Spike-in (recorded, no bar): pUC19 methylated-call fraction and technical error per run; whether Swift − TruSeq technical error
   accounts for the ε gap.
If 1 fails, the offset is cell-dependent and cross-cell references are not used; the route is public same-DNA data (EpiQC, CANDIDATE_EPIQC.md).

---
## Results (2026-10-09; nothing above the line changed)
P_CD4 = **1.167** (Loyfer Blood-T-CD4, 3 whole files; ε 0.0386 / 0.0398 / 0.0392; each donor reads 0.99 / 1.01 / 1.00 against the other two).
Laboratory-G CD4 (ε; A on P_CD4): Sample5 Swift/NovaSeq 0.0514, 1.226; Swift/HiSeqX 0.0517, 1.232; TruSeq/HiSeqX 0.0450, 1.110.
Sample8 0.0518, 1.233; 0.0521, 1.238; 0.0450, 1.110. (SRR9888326's conversion first failed on a missing genome link; redone from its BAM.)

| check | bar | result | met |
|---|---|---|---|
| 1. shared kit offset: CD4 Swift ÷ TruSeq vs neutrophils | within 0.03 | CD4 1.109 / 1.116; neutrophils 1.115 / 1.112 — **difference 0.001** | **yes** |
| 2. cross-cell tare: 6 neutrophil runs Normal | 6 / 6 | A_rel 0.939–0.948 — **0 / 6** | **no** |
| 3. kit gap removed after the tare | ≤ 0.02 | 0.004 and 0.002 | **yes** |
| 4. pUC19 spike | recorded | no methylated spike (0–29 reads, unmethylated) | – |

\measured The laboratory-and-kit offset is the same for two cell types to 0.001: it is a property of the library chemistry, not of the
cell, and a same-run tare removes it completely (gap 0.002–0.004). But a reference of another cell type does not carry the absolute level:
relative to Loyfer, laboratory G's CD4 sit ~6 % higher than its neutrophils, so neutrophils tared on CD4 read 0.94. Cross-cell references
need the CD4-to-neutrophil ratio measured in the same laboratory; same-cell references do not. Next: same-cell-type references
(DEV-IAMA-WBTARE-01, whole blood, running).
