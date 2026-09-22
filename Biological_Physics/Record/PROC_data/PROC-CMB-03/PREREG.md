# PREREG — PROC-CMB-03: the patient's sky, third seal (CMB-02's presence-floor rule failed on its face)

**Sealed 2026-09-21 after CMB-02 ran to completion as sealed.** CMB-02's rule "floor_c = max(0.02, p99 of f_c over the 160 panel arrays) for every class" gives immune a floor of 0.973 and progenitor 0.369 — the healthy panel's own composition — so blood-lineage classes are masked in healthy blood. The rule confused "what healthy blood shows" (a noise floor for classes it does NOT carry) with "what healthy blood is made of".

**Only change:** presence_floor_c = max(0.02, p99 over the 160 PANEL arrays) for the five classes healthy blood does not carry (stromal, cycling, secretory, terminal, stem_pluri); presence_floor_c = 0.02 for the blood-lineage classes (immune, progenitor, stem_adult). Stored in `presence_floors_v1.json` with the rule written in.

**Bars unchanged from CMB-02:** C2′ (≥ 3/4 labs; median frac |z| > 2 in [0.03, 0.08], median z within ±0.15), C3′ reported, C4′ (rule 160/160 on TEST arrays; ≤ 3 test arrays render a non-blood class; immune 160/160), C5, C6. Row 4.6 commissioned if C2′, C4′, C5, C6 pass.

---
**SEALED** sha256 `67589d2128c4fcabf5cc371246a3bcd14d19cf22e9f97e36d60b4328f0b27c78` · 2026-09-21
