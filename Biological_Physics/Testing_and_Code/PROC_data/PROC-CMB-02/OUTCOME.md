# OUTCOME — PROC-CMB-02: FAILED AS SEALED (C2′ 0/4; C4′ by construction)

**Run 2026-09-21, as sealed.** Panel zero subtracted; centred pooled scale; presence floor = max(0.02, p99 of panel fraction) for EVERY class.

**The zero works.** Held-out median z moved from ≈ −1.1 (CMB-01) to −0.006 / −0.032 / −0.012 / −0.024.

| lab | median frac \|z\| > 2 [range] | median z | C2′ |
|---|---|---|---|
| GSE87571 | 0.018 [0.005–0.150] | -0.006 | FAIL |
| GSE42861 | 0.015 [0.006–0.089] | -0.032 | FAIL |
| GSE111629 | 0.019 [0.004–0.102] | -0.012 | FAIL |
| GSE125105 | 0.022 [0.006–0.120] | -0.024 | FAIL |

**C2′ fails the other way:** tails 1.5–2.2 % against a 3 % floor — the scale is inflated. **C4′ fails by construction:** the floor rule applied to blood-lineage classes gave immune a floor of 0.973 (its own p99), so immune rendered in 2/160 healthy arrays. The rule confused what healthy blood *shows* for absent classes with what it *is made of*. C1 0.607, C5 PASS, C6 PASS. Cross-lab (A's zero+scale on B): GSE87571->GSE42861 0.061 / -0.17; GSE87571->GSE111629 0.050 / +0.04; GSE87571->GSE125105 0.148 / -0.17; GSE42861->GSE87571 0.039 / +0.18; GSE42861->GSE111629 0.062 / +0.32; GSE42861->GSE125105 0.104 / -0.10; GSE111629->GSE87571 0.057 / -0.05; GSE111629->GSE42861 0.115 / -0.29; GSE111629->GSE125105 0.157 / -0.23; GSE125105->GSE87571 0.063 / +0.14; GSE125105->GSE42861 0.054 / +0.03; GSE125105->GSE111629 0.054 / +0.24.

---
**SEALED** sha256 `e9635518944f1c83239c0242b4860c33df9df9934cff501eabf775fdfa91577b` · 2026-09-21
