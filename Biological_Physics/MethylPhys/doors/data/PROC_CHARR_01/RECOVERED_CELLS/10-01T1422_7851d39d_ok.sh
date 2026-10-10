cd rimouski && cat > PROC_RIMOUSKI_01_PREREG.md <<'EOF'
# PROC-RIMOUSKI-01 — pre-registration (written 2026-10-01, before any read of this dataset is aligned)

**Question.** Does the copy-error reading of Atlantic salmon fin tissue differ between hatchery-stocked and wild-born adults, and is it set by the
fish or by the library? Second: in their offspring (parr from the four parental crosses), does the parents' origin show up?

**Data.** Rimouski River, Québec (PRJNA892473): 64 WGBS paired-end fin-clip libraries, one run each.
- F0 spawning adults: 16 wild, 16 stocked (8 female, 8 male each).
- F1 parr, 8 per cross: wild × wild, wild father × stocked mother, stocked father × wild mother, stocked × stocked.

**Processing (fixed now, same as PROC-CHARR-01 except the genome).** First 5 M read pairs per fish streamed from ENA; Trim Galore --paired; Bismark
(directional, paired) to Ssal_v3.1 (GCF_905237065.1); deduplicated; first and last 3 aligned bases ignored. A fish with < 90 % of the 5 M pairs
is a failed run (rule added after PROC-CHARR-01) and is re-fetched once; if it fails again, it is excluded and listed.

**Statistic.** Identical to PROC-CHARR-01 and PROC-SALMON-01: qualifying molecule ≥ 6 CpG calls, ≥ 80 % methylated; isolated error = unmethylated
interior CpG between two methylated neighbours; ε_corr = ε − sequencing-error rate; per-fish genotype mask (> 30 % of ≥ 5 molecules);
E = ln((1 − ε_corr)/ε_corr) kT. One run per fish, so the two halves are alternate read pairs.

**Predictions.**
- **P0 (instrument):** the two halves agree, ICC ≥ 0.80 over the fish.
- **P1 (the fish, not the library):** |Spearman ρ| between ε_corr and (a) conversion failure, (b) duplicate fraction, (c) masked fraction is < 0.30 for all three.
  If any |ρ| ≥ 0.30, P1 fails and no fish-level difference (P3, P4) is interpreted.
- **P2 (scale):** median F0 fish reads 3.6–4.3 kT. This is a fin tissue window, wider than the sperm window, because no salmonid fin has been read yet.
- **P3 (origin, F0):** ε_corr ~ origin + sex, linear model; origin term two-sided, α 0.05. No direction is predicted.
- **P4 (parents, F1):** ε_corr ~ father origin + mother origin; each term two-sided, α 0.05; descriptive if P1 fails.
- Descriptive: F0 vs F1 (adult vs juvenile; confounded with age and year, so not tested).

**Stated limits now.** Fin is a mixed tissue (epidermis, fibroblasts, blood). One river, one lab. 16 per group in F0. A difference would not show
that it affects fitness. The library duplicate confound seen in brook charr may recur; P1 is the guard.
EOF
shasum -a 256 PROC_RIMOUSKI_01_PREREG.md | cut -c1-16