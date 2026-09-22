# GAPE Evidence — Access Restricted

**IAM-Genomics | US Provisional Patents 64/012,720 and 64/014,568**

The GAPE calibration artifacts previously published in this directory &mdash; the per-class
methylation posteriors, the non-methylation substrate posteriors, the per-cancer /
per-substrate validation matrices, the bootstrap-vs-MCMC cross-check tables, and the
reproducibility generator scripts &mdash; are part of the proprietary calibration layer and
are no longer distributed through this public repository.

## What the evidence package establishes

- **Cancer signal confirmation:** 27 of 28 TCGA cancer types produce A_tumor > A_normal
  at physics-derived thresholds, across 4,304 matched tumor-normal pairs. Full per-cancer
  detail is in the non-public evidence report.
- **Pre-cancer detection tier:** the window A = 1.01&ndash;1.05 is substrate-independent and
  reached in each of four independent substrates (methylation, nucleosome occupancy,
  fuzziness, WPS, fragment size).
- **Field effect:** adjacent-normal tissue elevation confirmed at p &lt; 10&#8315;&#185;&#8309;
  across 28 cancer types.
- **Aging trajectory:** normal aging does not reach the cancer threshold; the field effect
  signal in adjacent-normal tissue is decades ahead of chronological age.
- **Bootstrap cross-validation:** 10,000-resample non-parametric bootstrap agrees with
  G-003b MCMC posteriors at 0.168% mean relative difference; 24 of 32 posterior means
  within bootstrap 95% CI. **Calibration is method-independent.**

## What the A-score is

```
A_GAPE = H(beta) / H_min(class)

where H(beta) = -beta*log2(beta) - (1 - beta)*log2(1 - beta)   (Shannon binary entropy)
      beta    = mean CpG methylation from a standard array or WGBS
      H_min   = minimum entropy for the cell architecture class (proprietary)
```

The detection threshold A &gt; 1.05 is physics-derived, not fit from cancer data. It is the
point at which the three-component thermodynamic decomposition produces a significant
accessible gap &mdash; a departure from the architecture floor that the cell&rsquo;s
normal maintenance machinery cannot close.

## IP posture

- US Provisional Patent Application **64/012,720** &mdash; filed March 21, 2026
- US Provisional Patent Application **64/014,568** &mdash; filed March 23, 2026

The public disclosures across this repository &mdash; the VAL-XXX study descriptions, primary
data citations, the A-score formula, tier thresholds, and derivation framework &mdash; are
consistent with the scope of those filings. The numeric calibration layer, the MCMC chain
generators that produce the per-class floor values, and the engineering implementation
(including the scraper pipeline, the bootstrap cross-check infrastructure, and the
architecture-card builder) are not publicly disclosed.

## Access on request

Qualified research partners, clinical collaborators, journal referees, and commercial
licensees may request the full evidence package under mutual non-disclosure agreement.

- Research collaboration: [hmahaffeyges@gmail.com](mailto:hmahaffeyges@gmail.com?subject=GAPE%20Evidence%20Package%20%E2%80%94%20Access%20Request&body=Hello%20Heath%2C%0A%0AI%27m%20requesting%20access%20to%20the%20full%20GAPE%20evidence%20package.%0A%0AName%3A%0AAffiliation%3A%0APurpose%3A%0A%0AThank%20you.)
- Commercial / licensing: heath@iamperformance.net (through legal counsel)

---

*Heath W. Mahaffey &middot; Independent Researcher &middot; Entiat, Washington*
