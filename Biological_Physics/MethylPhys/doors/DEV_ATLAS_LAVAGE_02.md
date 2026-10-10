# DEV-ATLAS-LAVAGE-02 — the lavage reading on a second laboratory (written 2026-10-10, before download and before DEV-ATLAS-LAVAGE-01 is scored)
GSE206709 (National Jewish Health), 72 EPIC lavage arrays (chronic beryllium disease, sarcoidosis, controls). It records macrophage and lymphocyte
percentages, not neutrophils, so it tests the lymphocyte templates over a wide range. Method unchanged from DEV-ATLAS-LAVAGE-01
(`data/DEV_ATLAS_LAVAGE_02/score_lavage_02.py`). Simulation (`sim_lavage_lym_02.py`, lymphocytes 5–60 %, 400-cell count): mean |error| vs the
count 0.017, 98 % within 0.05, 100 % within 0.10. **Bars:** lymphocyte mean |error| ≤ 0.03; ≥ 90 % within 0.05; every sample within 0.10.
