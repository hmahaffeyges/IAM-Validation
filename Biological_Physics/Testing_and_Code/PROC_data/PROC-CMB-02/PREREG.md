# PREREG — PROC-CMB-02: the patient's sky, re-sealed after PROC-CMB-01 (C2 and C4 failed as sealed)

**Sealed 2026-09-21 after the CMB-01 outcome was written and before any array is scored under the new design.** Same 320 arrays, same seed-2028 panel/test split, same map.

**Changes from CMB-01, each tied to a measured failure.**
1. **Panel zero.** z_i = (r_i − m_i)/s_i, m_i = per-CpG mean residual across the laboratory's 40-array panel (CMB-01 C2: same-lab held-out median z ≈ −1.1; m_i median −0.036, |m|/s median 1.64).
2. **Centred pooled scale.** The β-binned pooled scale is computed on residuals centred by m_i (CMB-01 post-run check: uncentred pooling inflated s and gave 1.8 % tails).
3. **Measured presence floors.** presence_floor_c = max(0.02, p99 of f_c across the 160 PANEL arrays of the four labs), stored in `Runtime Matrices/Patient_CMB/presence_floors_v1.json`. Panels are never used to test themselves; C4 is judged on the 160 TEST arrays.

**Bars.**
- **C2′** per lab, 40 held-out: median frac |z| > 2 in [0.03, 0.08] and median z within ±0.15. Pass = ≥ 3/4 labs.
- **C3′** cross-lab reported (scale AND zero of lab A on held-out of lab B) — expected to fail C2′ bounds; documents that the zero is per-laboratory.
- **C4′** on the 160 TEST arrays: the gate follows its rule in 160/160, AND ≤ 2 % of test arrays (≤ 3) render any of stromal / cycling / secretory / terminal / stem_pluri; immune renders in 160/160.
- **C5, C6** as CMB-01 (mapping determinism; identical pixel arrays on regeneration).
- **Row 4.6 commissioned** if C2′ (≥ 3/4), C4′, C5, C6 pass.

---
**SEALED** sha256 `b940da55756caa7f7ec592ba21f70758320b04180e817e40925e070fa619d951` · 2026-09-21
