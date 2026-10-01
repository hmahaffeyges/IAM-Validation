# PROC-PREDX-NEUT-01 — pre-registration (written 2026-10-01, before EPIC-Italy is read with the corrected rules)

PROC-PREDX-SEQUENCE-01 was uninformative: the reader used non-450K floors and printed A for minor cells below resolution. DIAG-450K-01 fixed both:
a 450K neutrophil floor from purified 450K neutrophils (GSE88824; H_min = 0.784057, file floors_450k_v1.json), and A printed only for the dominant
cell. Neutrophil A = H(mean of the separated neutrophil β over its identity loci) / 0.784057, separated = (β − Σ_{k≠neu} f_k μ_k)/f_neu.

**Data.** GSE51032 (EPIC-Italy), our Stage 1; evidence only from the 516 arrays not in GSE51057. Specimens with neutrophil fraction < 0.30 are
reported, not scored.

**Predictions.**
- P1 (healthy reads Normal): ≥ 80 % of controls read neutrophil A within 0.95–1.05.
- P2 (immune first, breast): breast cases > 8 years before diagnosis read outside Normal more often than controls (one-sided Fisher, p < 0.05),
  and the same holds among women only.
- P3 (immune first, colorectal): the same for colorectal cases > 8 years before diagnosis, overall and within each sex present.
Descriptive: neutrophil A by years-to-diagnosis bin (> 8, 5–8, 2–5, < 2) for breast and colorectal; age and sex association in controls.

**Stated limits now.** Floor from 8 donors on one 450K batch; EPIC-Italy is a different lab; arrays give the program reading only.
