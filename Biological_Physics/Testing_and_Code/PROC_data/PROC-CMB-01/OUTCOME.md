# OUTCOME — PROC-CMB-01: the patient's sky, first sealed round — FAILED AS SEALED (C2, C4); C1, C5, C6 recorded

**Run 2026-09-21.** 320 healthy whole-blood arrays (four laboratories × 40 panel + 40 held-out, seed 2028), Stage 2 fractions from Walther, β on the `stage1_noob_450K` map.

| bar | result | verdict |
|---|---|---|
| C1 retired formula | median **60.7 %** of CpGs at \|z\| > 2 on 11 healthy arrays (expectation 5 %) | defect confirmed; the retired module's "built" label withdrawn |
| C2 held-out healthy, 4 labs | median z **−1.06 / −1.22 / −1.02 / −1.16**; frac \|z\| > 2 0.17–0.22 | **FAIL 0/4** |
| C3 cross-lab | 0.12–0.31 / −0.8 to −1.4 | reported |
| C4 gating | 153/160; the 7 are healthy arrays where Stage 2 reports terminal 2.1–2.9 % or stem_pluri 2.0 % and the gate rendered them as its rule says | **FAIL as sealed** — the bar's assumption failed, not the gate |
| C5 mapping | identical SHA-256 on rebuild; 0 unmapped atlas CpGs | PASS |
| C6 regeneration | identical pixel arrays | PASS |

**Why C2 failed — the analyst's design omission.** The PREREG said "the panel measures the zero AND the spread" and defined z = r/s — spread only. Held-out arrays from the *same* laboratory as the panel read median z ≈ −1.1, which cannot be a laboratory constant; it is the per-CpG mean residual m_i (median −0.036, IQR [−0.063, +0.033]; median |m|/s = 1.64) — the atlas-as-fifth-laboratory constant of PROC-SWITCH-01, resolved per CpG — left in the numerator. With m_i subtracted (measured after the run, GSE125105): median z −0.004, frac |z| > 2 = 0.018 — now *below* the 3 % floor, because the binned pooled scale was computed from uncentred residuals and absorbed m_i². Two errors, both in the analyst's design, both measured.

**Why C4 failed.** The presence floor (0.02) was asserted, not measured. Seven of 160 healthy arrays carry a non-blood class at 2.0–2.9 % under Stage 2. A presence floor must come from the healthy panels.

**Standing.** Row 4.6 NOT commissioned. The retired module's formula is closed (C1). Re-sealed as PROC-CMB-02 with the panel zero, centred pooling and measured presence floors.

---
**SEALED** sha256 `dd1c66cb799917a318e2f6e8576e8accd9437b460669b3f412e006004b9e3704` · 2026-09-21
