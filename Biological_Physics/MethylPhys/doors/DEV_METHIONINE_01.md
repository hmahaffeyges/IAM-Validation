# DEV-METHIONINE-01 — methionine depletion as a second SAM lever (development; step 1 written 2026-10-10, nothing downloaded)

**Candidate.** Yokogami et al. 2022, BMC Cancer 22:1351 (doi 10.1186/s12885-022-10280-5); reads PRJDB12471 (DDBJ, public): three glioma-initiating
cell lines (MZGC1-3), complete vs methionine-deficient medium, RRBS 150 bp paired (HiSeq X), one library per condition. The paper reports
intracellular SAM lowered by depletion (Fig. 2B), depletion for 48-72 h, and markedly decreased proliferation with cell death.
Public search 2026-10-10 found no other methionine-restriction sequencing set (GEO, SRA); GSE225944 (homocysteine for methionine) is arrays only.

**Step 1, calculation (`development/sims/methionine_01.py`, output `methionine_01_output.txt`).** Copy error is written only when DNA is copied.
In 48-72 h only a share of the cells copy their DNA, and each copied duplex carries one new strand. Mapped through Stage Q's measured response on
RRBS molecules (wild-type mouse liver stand-in), IAM-A_rel is >= 1.08 in 6 of 27 cells of the grid, all needing a 10-fold SAM fall or most cells
copying in the window; with half the cells copying once and a 4-fold fall, IAM-A_rel is 1.03-1.05.

**Decision: parked.** With three pairs and one library each, a rise of 1.03-1.05 cannot be told from library-to-library spread (0.02-0.04).
The set is underpowered for the reason the physics gives: too few copies were made during the depletion. A decisive second lever needs restriction
long enough for many divisions (weeks in culture, or diet in vivo); none is public as of this date.
