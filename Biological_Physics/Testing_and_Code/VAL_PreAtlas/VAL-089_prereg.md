# VAL-089 — glioma-epic Tumor Tissue arm — direct architecture A-score on brain tissue

**Status:** Pre-registration document; analysis fired with on-study healthy controls (no cross-platform reference confound).

---

## Cohort

- **Test cohort:** GSE60274 (Lai 2015, PMID 25622821 / 26482909 / 29299163 / 37491696). 77 samples on 450K (GPL13534):
  - 60 primary surgical GBM
  - 4 GBM with paired sphere line (treated as primary surgical for this VAL)
  - 4 recurrent GBM (originating from previously sampled primaries)
  - 4 cultured glioma spheres (LN-2207GS, LN-2540GS, LN-2669GS, LN-2683GS)
  - **5 non-tumor brain (NTB) controls** — from Lobectomy and Craniotomy for Epilepsy specimens. Age range 32-79, median 75.
- **Healthy reference:** the 5 NTB controls **on-study, internal**. CHK-3.2 cross-cohort baseline confound is eliminated for this VAL.

## Panel and normalization

- **Panel:** Xu-538 (SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`, n=538 CpGs).
- **Substrate-scope caveat (CHK-1.5):** Xu-538 was trained on whole-blood IMMUNE class. Applying it to brain tissue measures "what those CpGs read in this tissue" not "the tissue's intrinsic terminal-class architecture entropy." This is a **panel-on-non-native-specimen** analysis. Direction-of-effect is the primary inference; absolute magnitude depends on the H_min normalization choice and is reported with explicit substrate-scope caveat.
- **Two H_min normalizations reported:**
  - H_min(immune) = 0.838889 — for direct comparability with VAL-088 blood result
  - H_min(terminal) = 0.7728 — the brain-tissue-correct normalization per GAPE class assignment (brain is dominantly terminal class)

## Stratifications (declared before scoring)

1. NTB healthy controls (n=5) — reference
2. GBM primary surgical (n=64) — primary detection target
3. GBM recurrent (n=4) — disease-progression test
4. GBM cultured spheres (n=4) — pure neoplastic-cell-line homogeneity test (expected: A-score LOWER than mixed-cell tissue, not higher)

## Pre-locked decision criteria (CHK-2.1)

The framework hypothesis under test is: **GBM tumor tissue shows positive ΔA at v1 single-substrate methyl-only level, consistent in direction with Issue 002's 5-substrate cfDNA prediction (+0.217), but at smaller magnitude due to substrate-scope difference.**

Possible outcomes:
- **O1_PASS**: ΔA > 0 with d > +0.5 and 95% CI lower bound > 0. Framework hypothesis confirmed in direction at v1 scope.
- **O2_PARTIAL**: ΔA > 0 but |d| < 0.5 OR CI crosses zero. Direction consistent, magnitude weak.
- **O3_NULL**: |d| < 0.2 with CI tightly bracketing zero. No detectable shift.
- **O5_NEGATIVE**: ΔA < 0 with d < -0.5. Direction inverted from prediction.
- **O6_UNEXPECTED**: data integrity flagged or other diagnostic-pending status.

## Pre-locked supplementary outcomes

- Recurrent GBM expected to show similar-or-larger ΔA than primary (disease progression hypothesis).
- Cultured spheres expected to show LOWER A-score than mixed-cell brain tissue (homogeneity hypothesis — pure tumor-cell-line β distribution is less mixed than tissue containing neurons + glia + microglia + endothelium).

## CHK requirements

- **CHK-1.5 substrate-scope:** Issue 002 ΔA = +0.217 figure is 5-substrate cfDNA L2/L3 prediction; v1 single-substrate methyl-only readings are different scope. No direct magnitude comparison.
- **CHK-1.6 access tier:** Tier 1 (GEO public).
- **CHK-2.4 panel transferability:** Panel-on-non-native-specimen acknowledged. Direction-of-effect primary; absolute magnitude carries panel-scope caveat.
- **CHK-3.1 β-distribution:** Required >20% extremes <0.1 or >0.9, <40% in [0.4, 0.6]. Brain-tissue distributions may be flatter than blood; threshold accommodates this.
- **CHK-3.3 panel coverage:** Required mean Xu-538 coverage ≥400 of 538 per QC-passed sample; 450K platform expected ~100% coverage.
- **CHK-3.5 saturation:** Per-sample distance to A_ceiling reported under both normalizations.
- **CHK-3.2 healthy baseline:** ON-STUDY controls eliminate cross-cohort confound.

## Caveats declared

- NTB controls are surgical specimens from epilepsy / lobectomy patients — "non-tumor brain" but not pristine healthy brain. Some methylation drift from chronic seizure activity is possible.
- NTB controls are older (median age 75) than GBM cohort (median ~55). Age-adjustment not applied at v0.1; future work.
- Small healthy reference (n=5) produces wide CIs even when point estimates are meaningful.
- Per CCL-025, GBM tumor tissue contains genuine architecture disruption AND immune infiltrate (TAMs, microglia). The A-score reflects both contributions; clean separation requires Stage 3 deconvolution (GIMiCC / Salas 2024), deferred to v0.2.

## Files at lock

- `val_089_glioma_epic_tissue.py` — analysis script
- `GSE60274_manifest.json` — parsed metadata for all 77 samples

## Output

- `VAL-089_results.json` — full numerical output
- `VAL-089_distributions.png` — boxplot
- `VAL-089_outcome.md` — outcome interpretation
