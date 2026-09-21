# PREREG — PROC-CMB-01: Stage 4.6, the patient's sky (CHAIN_COMMISSIONING row 4.6)

**Sealed 2026-09-21 before the module is written or any array is scored.**

**Defect found on reading (disclosed, to be measured as C1).** The retired `patient_brightness_comparison.py` computes z = (β_patient − μ_class)/σ_class per CpG with σ_class = the atlas **posterior SD of the class mean** (median 0.016 on immune) and μ_class = a **pure-class** mean. On one cached healthy whole-blood array this gives 40 % of CpGs at |z| > 2 (read before sealing). A healthy patient's sky under that formula is almost entirely anomalous. The row's "built" label is therefore withdrawn pending this procedure.

**Design (from PROC-SWITCH-02: patient β is the linear mixture of atlas class means).**
- Expectation per CpG: E_i = Σ_c f_c · μ_{c,i}, f = the sample's own Stage 2 (Walther) class fractions; μ from `IAMAtlasREBUILD.csv`; β on the mapped scale (`stage1_noob_450K` map).
- Residual r_i = β_i − E_i. Healthy scale s_i = per-CpG SD of r_i across the laboratory's 40-array healthy panel (the lab-zero panel, extended: the panel measures the zero AND the spread), floored at 0.005 and shrunk toward the β-binned pooled scale with weight n_panel/(n_panel+10).
- z_i = r_i / s_i. Sky = z projected to HEALPix NSIDE 128 in genomic order (manifest CHR × MAPINFO, sequential pixel assignment, mean per pixel), Mollweide, RdBu_r centred at 0, one panel per class.
- **Gating:** a class panel renders only if that sample's f_c ≥ 0.02 (Stage 2 presence); otherwise it is masked and labelled NOT ASSESSABLE. Within a rendered class panel the z is shown on that class's identity loci; a full-genome panel shows all loci.

**Bars.**
- **C1 defect measured** — retired formula on the 11 cached healthy whole-blood arrays: fraction |z| > 2 reported (expected ≫ 0.05).
- **C2 held-out healthy** — for each of the four laboratories (GSE87571, GSE42861, GSE111629, GSE125105): 40-array panel builds s_i; on 40 disjoint held-out healthy arrays of the same lab, the median over arrays of frac |z| > 2 lies in [0.03, 0.08] and the median z lies within ±0.15. Pass = ≥ 3 of 4 labs; the fourth reported.
- **C3 cross-lab** — panel of lab A applied to lab B's held-out: frac |z| > 2 and median z reported (expected to fail C2's bounds — this documents why the scale is per-laboratory, like the zero).
- **C4 gating** — on all 320 whole-blood arrays: stromal, cycling, secretory, terminal, stem_pluri panels masked (f_c < 0.02); immune rendered; progenitor/stem_adult rendered where f_c ≥ 0.02. 100 % conformity.
- **C5 mapping deterministic** — HEALPix mapping built twice from the manifest gives the same SHA-256; the number of atlas CpGs without a manifest position is reported.
- **C6 kit regeneration** — the plate for one held-out healthy array regenerates from the kit with an identical pixel array (numpy equality) on two runs.
- **Row 4.6 commissioned** if C2 (≥ 3/4), C4, C5, C6 pass. C1 and C3 are reported regardless.

**Panel/test selection:** per lab, seed 2028, 80 healthy arrays drawn without replacement from the cohort; first 40 = panel, last 40 = test. Ages as recorded by GEO.

---
**SEALED** sha256 `4182c400519831ec719e283f0fa8993e0309afe8337721936e848c564451d687` · 2026-09-21
