# PROC-G002-TRACE — how the eight original floors were made, and what IAM's law says about them (2026-09-30)

**How they were made (mphys_mcmc_g002.py, April 2026).** Each of 37 reference cells was given one number, β̄ = its genome-wide mean methylation
from a cited source. The class floor was set to H(β̄) of the most-methylated cell in the class. G-002 then floated the eight floors with a
likelihood Σ((H(β̄)/H_min − 1)/0.02)², i.e. it chose each floor so that every cell of the class reads A ≈ 1. The eight classes were inputs.

**Re-measured on single molecules (Loyfer 2023, 56 cell types, the same 798,000 CpGs), 15 of the 37 cells matched:**
median |β̄ measured − β̄ published| = 0.023, correlation 0.37. Close for 11 of 15; far for naive CD4 T (0.842 vs 0.730), NK (0.788 vs 0.735),
erythroid progenitor (0.554 vs 0.725), pancreatic acinar (0.685 vs 0.730). Table: g002_vs_measured.csv.

**What β̄ physically is.** Across the 56 cell types, the fraction of CpGs the cell keeps methylated explains 96.5 % of the variance in β̄
(with the partly methylated fraction, 98.3 %). The cell's copy error accounts for ~5 % of the spread directly. So H(β̄) measures the entropy of the
cell's PROGRAM — how much of its genome it keeps written as 1 — not the error on it. β̄ ≈ f_meth(1 − ε_m) + f_unmeth·ε_u + ½·f_mid (median
residual 0.026).

**Can IAM's law reproduce the eight floors?** Not from kT and M alone. IAM's law fixes the cost and error of HOLDING each bit (ε, 3.4 kT per
methylated site, φ = 0.16 of one ATP). How many bits a cell holds as 1 is set by its program, not by the physics. The original floors are
therefore program-composition floors; the IAM floor is an error floor. They answer different questions.

**Why the original A still read real signal.** Losing methylation across the genome lowers β̄ toward 0.5 and raises H(β̄): global
hypomethylation, a known feature of ageing and cancer, reads as A > 1 on the old gauge. That change is a shift in the program's composition
(plus error), and it is real — but the "1.00" was set by the class's reference cell, not by physics.

**Consequence for the instrument.** Two readings, both per cell, no cohort:
1. Error reading (IAM's law): A on the physics floor ε₀ = 1/(1+e^(φM)), per channel.
2. Program reading: β̄ or f_meth against the cell's own healthy value — the quantity the original floors captured.
