# DEV-IAMA-WBTARE-01 — the IAM-A same-run tare on independent people: laboratory-G whole blood (written 2026-10-09, before reading)

**Why.** The tare reads each sample as a ratio against ≥ 3 healthy samples of the same laboratory, kit and pipeline, so the position P
cancels: the test needs healthy people of one kind, ≥ 3 per kit, not neutrophils. GSE128731 (laboratory G) sequenced whole blood of
**4 healthy donors on both the Swift and the TruSeq kit**, with repeat libraries (author: "find another way"; BLUEPRINT not pursued).

**Runs (Box Run 2 session 4).** Swift/HiSeqX and TruSeq/HiSeqX of Sample1-4 (8 runs; HiSeq X only, so the kit is the only difference),
plus the repeat libraries Swift/HiSeqX rep2 and TruSeq/HiSeqX rep2 of Sample2-4 (6 runs). 25 M pairs each; pinned pipeline; Q0; ε.
Whole blood is a mixture: ε is read as is (no cell sorting); the tare compares like with like (whole blood with whole blood).

**Checks (bars fixed now).** For each run: A_rel = H(ε) ÷ median H(ε) of the other donors' rep1 runs on the same kit (3 references).
1. Healthy reads Normal after the tare: 8 of 8 rep1 runs within 0.95–1.05.
2. Kit offset removed: per donor |A_rel(Swift) − A_rel(TruSeq)| ≤ 0.02, for 4 of 4 donors (untared gap recorded; neutrophils ~0.12).
3. Repeatability: per donor and kit |A_rel(rep2) − A_rel(rep1)| ≤ 0.02 (6 of 6).
4. A real change survives the tare: copy errors added in silico to one donor's molecules (extra rate δ giving IAM-A 1.10) read
   A_rel ≥ 1.08 against untouched references, on both kits.
