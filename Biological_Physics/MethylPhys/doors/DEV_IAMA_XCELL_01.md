# DEV-IAMA-XCELL-01 — commissioning IAM-A without ≥ 3 same-cell donors per laboratory: cross-cell same-run tare and a spike-in standard
(development; written 2026-10-09 before any reading below)

**Problem.** IAM-A carries a laboratory-and-kit offset (DEV-IAMA-KIT-01: same neutrophils, Swift 1.16 vs TruSeq 1.05). The author chose the
same-run tare (≥ 3 healthy references read the same way). Search for open read-level neutrophil WGBS with ≥ 3 healthy donors per laboratory
and kit (2026-10-09; SRA, GEO, ENCODE): none besides Loyfer 2023 (3 granulocytes, the files P is measured on). RRBS sets exist but are not
whole-genome (P is measured on whole files). The only such set is controlled access: **BLUEPRINT EGAD00001001201, 6 mature neutrophils, one
laboratory (CNAG), DAC EGAC00001000135** — application drafted for the author.

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
If 1 fails, the offset is cell-dependent and cross-cell references are not used; the BLUEPRINT route remains.
