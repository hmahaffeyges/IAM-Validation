# Candidate: SEQC2 EpiQC — the same DNA across laboratories, kits and platforms (search only, 2026-10-10; nothing downloaded)

Foox et al. 2021, Genome Biology 22:327 (doi 10.1186/s13059-021-02529-2). Seven Genome in a Bottle reference cell lines (HG001–HG007, lymphoblastoid).
- **Sequencing:** SRA BioProject PRJNA646948, runs SRR13050956–SRR13051274 (paper's data statement): TruSeq, Accel-NGS MethylSeq (Swift), SPLAT,
  TrueMethyl, EM-seq, Methyl Capture EPIC, nanopore; two or more technical replicates per sample; NovaSeq 6000. bedGraphs in GEO GSE186383 (106).
- **Arrays:** GEO GSE230132 (the paper names GSE186383 for IDATs; they are deposited here), 30 EPIC arrays with IDATs: lab A all 7 lines × 2
  replicates, lab B all 7 × 1, lab C HG005/HG006/HG007 × 3.

**What it can test.** Not absolute readings (no healthy reference exists for these lines in either instrument). It tests what Box Run 2 could not:
IAM-A on identical DNA across kits (TruSeq against Swift is the offset left open in DEV-IAMA-WBTARE-01) and laboratories, before and after the
same-run tare; IAM-A repeat spread from technical replicates; Met-A's laboratory offset and repeat spread on identical DNA at three laboratories
(site rule of the commissioned floor, read relative within each cell line). Both instruments on the same DNA.
**Before any download:** check the IDAT revision (Stage 1 reads the 1,051,815-address EPIC; the 1,052,641 revision fails), the conversion of each
kit's libraries against the Q0 limit, and simulate the resolving power from the replicate counts above.
