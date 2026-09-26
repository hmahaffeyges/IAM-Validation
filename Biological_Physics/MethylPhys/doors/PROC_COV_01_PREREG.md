# PROC-COV-01 — is the reference's misfit against real blood a reproducible bias that can be measured and removed?

**Pre-registered 2026-09-26, before any residual was computed.**

## Why this replaces the shelf item as written

The ranked enhancement list says the full cell-type covariance is *"recoverable from the 165 MB per-class
MCMC archives."* **That is false, and it was checked rather than assumed.** The archives contain only
marginals — `cpg_id, mean, sd, ci_lo, ci_hi` — and the build script's own correctness note states that *each
CpG's posterior is independent in the model*, with the eight classes run as separate jobs. **No joint draws
exist, so no cross-class covariance at an address was ever estimated.** The list has been corrected.

What *is* available, and what actually governs the fit, is the **residual** covariance on real specimens:

    r_i = β_i − Σ_c f̂_c,i μ_c        for each healthy array i

This contains reference error, biological variation, technical noise **and model misspecification** — and it
is the last of these that closed PROC-PARTIAL-01, measured there as a **+0.067 β** systematic under-prediction
at secretory's identity loci, which 1/f amplification turned into 3.35 at f = 0.02. A posterior covariance
would not have contained that term at all.

## The question

Is that misfit a **reproducible, specimen-independent bias** — something the instrument could measure once
and subtract — or is it specimen-specific, in which case nothing can be removed and the limit is real?

## Data and construction

The **318 healthy whole-blood arrays** already calibrated and published in
`kit/results/PROC_BAND_01_arrays.json`, across four laboratories (GSE87571, GSE42861, GSE111629, GSE125105).
No new data. Fractions are the chain's own fitted `f̂`, not refitted.

**Everything is estimated leave-one-laboratory-out.** The bias vector is estimated on three laboratories and
applied to the fourth, four times over. A bias fitted and tested on the same arrays would reproduce itself by
construction, which is the whole trap this design avoids.

## The bars

| | bar | met when |
|---|---|---|
| **B1** | the bias is reproducible across laboratories | the held-out laboratory's median \|residual\| falls by **≥ 50 %** after subtracting a bias estimated without it, in all four folds |
| **B2** | it is a bias and not noise | the correlation between bias vectors estimated on disjoint laboratory pairs is **r ≥ 0.7** |
| **B3** | it fixes what it was meant to fix | with the held-out bias correction applied, PROC-PARTIAL-01's recovered mean β at **f = 0.20** lands inside **[0, 1]** and within **2σ (0.0418)** of the true 0.732 |
| **B4** | the covariance adds something the mean does not | a rank-k factor model on the residual reduces held-out residual variance by **≥ 20 %** beyond the bias vector alone |
| **B5** | the instrument does not move | immune A_mapped on the 318 published arrays is **unchanged to 1e-9** when no correction is applied — the correction is a new, optional path, not a redefinition |

**Direction fixed:** the residual bias is expected *positive* (the atlas under-predicts real blood), as
measured in PROC-PARTIAL-01. A negative bias of similar size would mean the two measurements disagree and
must be reported as such, not averaged.

## What each outcome licenses

- **B1–B3 met:** the misfit is an instrument constant, measurable once per laboratory and removable. This would reopen PROC-PARTIAL-01 as a new procedure with the same bars — not a rerun of the sealed one — and would be the first real improvement to the reference rather than a workaround.
- **B1 met, B3 not:** the bias is real and removable but too small to rescue fidelity recovery at low fractions. The correction still belongs in the chain as a reconstruction improvement; the PARTIAL-01 closure stands.
- **B1 fails:** the misfit is specimen-specific. That is the stronger negative and it closes the whole approach: no measurable constant exists, and improving the reference means new reference *data*, not better use of what is here.
- **B4 met but B1 not:** structure exists in the residual that is not a constant offset — worth reporting and worth a different procedure, but not a correction anyone can apply.

**No clinical claim and no change to any reported number follows from this procedure.** It is an instrument
measurement. Nothing in the chain changes unless a later, separate procedure commissions the correction.

## The ordinary check this does not replace

Per the caution recorded in [`ATLAS_READABILITY.md`](ATLAS_READABILITY.md) §5: before any residual structure
is called covariance, the residuals get looked at directly — per-locus outliers, contaminated control
addresses, and whether a **median** rather than a mean is the right summary. The defect that nearly cost two
specimens this week was three bad addresses and a mean where a median belonged, and no amount of factor
modelling would have found it.
