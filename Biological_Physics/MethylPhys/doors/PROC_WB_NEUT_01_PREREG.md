# PROC-WB-NEUT-01 — pre-registration (written 2026-10-01, before any mixture is read on this reading)

**Question.** In a blood specimen, does Met-A of the neutrophil read Normal when healthy, and detect a known 2 % loss of the neutrophil's pattern,
if it is read against the healthy expectation for that specimen's own composition (no cohort)?

**Reading (fixed now).** Sites: neutrophil identity sites (Met-A v1.2 rule: SD ≤ 0.05; mean 0.75–0.95 or 0.05–0.25; ≤ 3,000 per channel).
Expected healthy β at each site for the specimen: e_i = Σ_c f_c μ_{c,i} (f = the specimen's fractions, μ = purified healthy profiles).
Met-A_blood = mean_i H(β_i) / mean_i H(e_i). Normal 0.95–1.05.
**Independence.** Sites and profiles for the Salas 2018 mixtures come only from Salas 2022 purified arrays, and vice versa.

**Data.** The 24 Salas DNA mixtures of purified healthy cells (GSE110554, GSE167998), known fractions (salas_mixture_truth.csv); our Stage 1.
Two fraction sources: (a) the known fractions, (b) the atlas v2 solver's fractions (deconv_v2).
**Damage.** The neutrophil share is blurred 2 % toward 0.5 in silico: β' = β + f_neu · 0.02 · (0.5 − μ_neu).

**Predictions** (on mixtures with neutrophil fraction ≥ 0.50, the blood-like range):
- W1: healthy, known fractions: all in Normal.
- W2: healthy, solver fractions: ≥ 80 % in Normal.
- W3: 2 % damage, known fractions: ≥ 80 % above 1.05.
Descriptive: every mixture vs its neutrophil fraction; the raw reading (no expectation) for comparison; repeat-reading agreement is tested
separately on real replicate pairs.
**Stated limits now.** DNA mixtures, not real blood; few mixtures with ≥ 50 % neutrophils; damage is simulated.
