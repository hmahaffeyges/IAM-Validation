# CPG Patient Intake Questionnaire v1.0

**For the customer.** Your answers to these questions calibrate how the engine reads your sample. The engine uses your responses to subtract age, sex, and lifestyle drift from your cellular methylation pattern before scoring — so the result reflects your specific physiological state rather than the population average. **Honest answers produce better readings.** If you're unsure about a question, leave it blank or choose "prefer not to say"; the engine handles missing data conservatively rather than guessing.

**Privacy.** Your answers stay paired with your sample's anonymized ID. They are not shared with any third party. The engine reads them, the report references them when relevant, and the audit trail logs them for chain-of-custody integrity.

---

## Section A — Basic demographics

**1. What is your date of birth?**
*Engine use: chronological_age (Stage 4 immune age delta computation, Stage 6 cellular age reference).*

| | Date |
|---|---|
| Year | ☐☐☐☐ |
| Month | ☐☐ |
| Day | ☐☐ |

**2. What is your sex at birth?**
*Engine use: sex_at_birth (Stage 7 threshold stratification — immune-class effect magnitudes differ substantially by sex per CCL-002).*

☐ Female ☐ Male ☐ Intersex ☐ Prefer not to say

---

## Section B — Smoking history

**3. What is your current smoking status?**
*Engine use: smoking_status (Stage 7 threshold bin selection). Methylation carries a strong, durable tobacco signature — even years after quitting.*

☐ Never smoker (less than 100 cigarettes in lifetime)
☐ Former smoker (smoked previously, quit)
☐ Current smoker (any cigarettes in past 30 days)
☐ Prefer not to say

**4. If you're a former smoker, when did you quit?**

☐ Less than 1 year ago
☐ 1–5 years ago
☐ 5–15 years ago
☐ 15+ years ago
☐ Not applicable / current smoker / never smoker

**5. If you currently smoke or have smoked, approximately how many years total did you smoke?**

☐ Less than 5 years ☐ 5–15 years ☐ 15–30 years ☐ 30+ years ☐ Not applicable

---

## Section C — Recent illness and immune events

**6. Have you had any significant illness, fever, or infection in the past 3 months?**
*Engine use: recent_illness_within_3_months (report context — recent illness can elevate immune-class signal for weeks to months).*

☐ Yes ☐ No ☐ Prefer not to say

If yes, briefly describe: __________________________________________

**7. Have you received any vaccinations in the past 3 months?**
*Engine use: recent_vaccination_within_3_months (report context — vaccination produces a measurable immune-class signature for several weeks).*

☐ Yes ☐ No ☐ Prefer not to say

If yes, which vaccine(s) and approximately when? __________________________

---

## Section D — Hormonal status (all customers)

**8. Are you currently pregnant?**
*Engine use: current_pregnancy_with_trimester (Stage 7 CONTEXT_PREGNANCY mode — pregnancy produces well-documented immune compartment shifts).*

☐ No ☐ Yes — first trimester ☐ Yes — second trimester ☐ Yes — third trimester ☐ Prefer not to say

**9. Have you given birth in the past 6 months?**

☐ Yes ☐ No ☐ Not applicable

**10. (Female customers) When was your last menstrual period? What is your menopause status?**
*Engine use: menopause_status (Stage 7 threshold context — peri-menopause and menopause shift the immune baseline through estrogen withdrawal).*

☐ Pre-menopausal (regular cycles)
☐ Peri-menopausal (irregular cycles, hot flashes, other transition symptoms)
☐ Menopausal (no period for 12+ months, natural)
☐ Post-menopausal (surgical or chemical induction)
☐ Not applicable

Date of last menstrual period (if known): __________

**11. Are you currently on hormone replacement therapy (HRT or bHRT)?**
*Engine use: hrt_status (Stage 7 CONTEXT_HRT_BASELINE mode — first-of-its-kind HRT-stratified immune readout per CPG-VAL-018).*

☐ Yes — estrogen alone (oral or topical)
☐ Yes — estrogen + progesterone
☐ Yes — bio-identical (compounded)
☐ Yes — testosterone (low-dose for women)
☐ Considering or recently stopped
☐ No
☐ Prefer not to say

**12. (Male customers) Are you currently on testosterone replacement therapy (TRT)?**
*Engine use: trt_status (Stage 7 threshold context — TRT affects the male immune compartment indirectly through metabolic and inflammatory pathways).*

☐ Yes — currently on TRT
☐ Considering or recently stopped TRT
☐ No
☐ Not applicable
☐ Prefer not to say

---

## Section E — Weight, metabolic, GLP-1

**13. Are you currently taking a GLP-1 medication or related weight-loss medication?** (Semaglutide / Ozempic / Wegovy, Tirzepatide / Mounjaro / Zepbound, Liraglutide / Saxenda, Dulaglutide / Trulicity, or similar)
*Engine use: current_glp1_or_weight_loss_medication (Stage 7 CONTEXT_WEIGHT_LOSS_INTERVENTION mode — these are documented anti-inflammatory interventions per CPG-VAL-021).*

☐ Yes — currently taking ☐ Considering / recently started ☐ Previously took, no longer ☐ No ☐ Prefer not to say

If yes, which medication and approximately when did you start? __________________

**14. Have you had bariatric surgery in the past 18 months?**

☐ Yes ☐ No ☐ Not applicable

---

## Section F — Autoimmune, chronic inflammatory, immunosuppression

**15. Have you been diagnosed with an autoimmune condition?**
*Engine use: known_autoimmune_condition (Stage 7 TRAJECTORY_WATCH mode — autoimmune customers receive trajectory-mode reporting instead of single-timepoint tier scoring).*

☐ Yes ☐ No ☐ Prefer not to say

If yes, please specify (check all that apply):

☐ Systemic lupus erythematosus (SLE) ☐ Rheumatoid arthritis (RA) ☐ Multiple sclerosis (MS)
☐ Type 1 diabetes ☐ Hashimoto's thyroiditis ☐ Graves' disease ☐ Psoriatic arthritis
☐ Sjögren's syndrome ☐ Other: __________________

**16. Have you been diagnosed with a chronic inflammatory condition?**
*Engine use: known_chronic_inflammatory_disease (Stage 7 TRAJECTORY_WATCH mode).*

☐ Yes ☐ No ☐ Prefer not to say

If yes (check all that apply):

☐ Crohn's disease ☐ Ulcerative colitis ☐ IBD-unclassified
☐ Ankylosing spondylitis ☐ Other: __________________

**17. Are you currently on immunosuppressive medication?**
*Engine use: current_immunosuppression (Stage 7 EXPECTED_SUPPRESSION mode override).*

☐ Yes — biologic agent (e.g., adalimumab, infliximab, rituximab, etanercept, etc.)
☐ Yes — long-term oral corticosteroids
☐ Yes — methotrexate
☐ Yes — cyclosporine, tacrolimus, mycophenolate, or similar
☐ Yes — other immunosuppressant: __________________
☐ No
☐ Prefer not to say

---

## Section G — Cancer history and current treatment

**18. Are you currently receiving cancer treatment?**
*Engine use: current_cancer_in_treatment (Stage 7 TREATMENT_RESPONSE mode).*

☐ Yes — chemotherapy ☐ Yes — immunotherapy ☐ Yes — radiation
☐ Yes — recently completed treatment (within 12 months) ☐ Yes — recent surgery
☐ No ☐ Prefer not to say

If yes, briefly describe what cancer and what treatment: ________________________

**19. Have you had any prior cancer diagnoses?**
*Engine use: prior_cancer_history (Stage 7 threshold context — remission-baseline interpretation).*

☐ Yes ☐ No ☐ Prefer not to say

If yes, please list type(s) and approximate year of diagnosis: __________________

**20. Have you received chemotherapy in the past 5 years?**
*Engine use: prior_chemotherapy_history (Stage 7 threshold context — chemotherapy persistently alters methylation for years post-treatment).*

☐ Yes, within past 2 years ☐ Yes, 2–5 years ago ☐ No

**21. Have you received radiation therapy in the past 5 years?**
*Engine use: prior_radiation_history (Stage 7 threshold context — radiation persistently alters methylation in the irradiated field).*

☐ Yes — head/neck region ☐ Yes — chest/breast ☐ Yes — abdominal/pelvic
☐ Yes — bone-marrow-containing field (pelvis, sternum, vertebrae, ribs, proximal long bones)
☐ Yes — other: __________________
☐ No

---

## Section H — Transplant, HIV, infectious

**22. Have you had a solid-organ or stem-cell transplant in the past 24 months?**
*Engine use: transplant_status (Stage 7 EXPECTED_SUPPRESSION mode override).*

☐ Yes — solid-organ transplant ☐ Yes — stem-cell or bone-marrow transplant ☐ No

**23. Are you HIV-positive?**
*Engine use: hiv_status (Stage 7 TRAJECTORY_WATCH mode with shifted thresholds; HIV+ treated immune compartment runs at a known baseline shift).*

☐ Yes — on antiretroviral treatment with confirmed viral suppression
☐ Yes — not currently on treatment
☐ Yes — recently diagnosed, treatment status unclear
☐ No
☐ Prefer not to say

---

## Section I — Medications (optional but helpful)

**24. Please list any prescription medications you're currently taking** (other than what you've already mentioned above):
*Engine use: current_medications_systemic (report context — certain medications affect methylation; the engine surfaces unusual readings paired with declared medications).*

```
1. _________________________________
2. _________________________________
3. _________________________________
4. _________________________________
5. _________________________________
```

---

## Optional: Information you'd like your clinician to consider

If you have a specific concern that prompted you to take this test, you can note it here. The engine doesn't use this field — your clinician reads it during your follow-up conversation.

```
__________________________________________________
__________________________________________________
__________________________________________________
```

---

## How your answers are used in the chain

The engine uses your intake at five stages:

1. **Stage 0 — Intake.** Your `patient_manifest.json` is created and hashed for the audit trail.
2. **Stage 3 — Foreground subtraction.** Chronological age is subtracted from your per-CpG β values before A-score computation. Future: smoking-axis subtraction (v1.1) will subtract the tobacco-associated CpG drift before scoring.
3. **Stage 4 — A-score and cellular age.** Your chronological age sets the reference for computing the immune age delta (immune cellular age − chronological age = inflammaging quantum).
4. **Stage 7 — Tier breakpoint.** Smoking bin selects threshold table. Autoimmune / HIV / pregnancy / HRT / GLP-1 / chemotherapy contexts select the corresponding interpretation mode (TRAJECTORY_WATCH, TREATMENT_RESPONSE, EXPECTED_SUPPRESSION, CONTEXT_*).
5. **Stage 8 — Card matching.** Some covariates (e.g., HPV status for cervical card, prior chemotherapy for progenitor card) affect disease-card interpretation downstream.

You can update your intake before each retest. Trajectory monitoring across serial tests is the most useful lens on cellular health — your answers help the engine calibrate what's drift, what's context, and what warrants a clinician conversation.

---

*Version: v1.0 — 2026-06-06*
*Subject to revision as the engine's foreground-subtraction modules mature (v1.1) and as additional covariates become engine-relevant (v1.2+).*
