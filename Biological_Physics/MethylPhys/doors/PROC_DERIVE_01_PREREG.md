# PROC-DERIVE-01 — pre-registration (written 2026-09-30, before the time-course files are summarised)

**Question.** Can the cell's steady copy error be derived from how fast it re-writes its marks after replication, measured independently?

**Data (independent of every dataset used so far).** Repli-BS, HUES64 human embryonic stem cells (Charlton et al. 2018, GSE82045): nascent DNA
after a 1-h BrdU pulse, chased 0, 1, 2, 4, 8, 16 h (replicate 3 series), and bulk DNA. Files hold methylated/total reads per CpG.

**Derived quantity.** On CpGs methylated in bulk (bulk β ≥ 0.8, ≥ 10 reads): restoration r(t) = β_nascent(t) / β_bulk, pooled over sites.
Per-cycle shortfall f = 1 − r(16 h) (16 h ≈ one hESC cycle). Energy of the kinetic shortfall E = kT·ln((1 − f)/f).

**Predictions.**
- P1 (derivation): f lies within a factor of 2 of the steady copy error measured on single molecules in HUES64 by ENCODE, instrument-corrected
  (0.0186; PROC-ENCODE-01) — i.e. 0.0093 ≤ f ≤ 0.037.
- P2 (enzyme link): E lies within the range set by DNMT1's published in-vitro selectivity, 1.9–4.4 kT.
Descriptive: r(t) curve; half-time of restoration.

**Stated limits now.** Different HUES64 cultures and labs; nascent-strand labelling dilutes by 16 h; bulk β itself carries the steady error, so
f is the shortfall relative to steady state; cycle length taken as 16 h, not measured here.
