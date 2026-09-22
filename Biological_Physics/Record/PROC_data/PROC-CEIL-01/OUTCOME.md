# OUTCOME - PROC-CEIL-01 (T-CEIL): the ceiling conformance guard. PASS, and it produced three findings the guard itself was not looking for.

**Run 2026-09-22** on every array held on this machine: the four laboratories' Stage 1 rebuild (318 healthy arrays: Uppsala 80, UCLA 80, Munich 80, Karolinska 78).
**492 readings checked** - the maximum over each (class x laboratory) on the identity gauge and each (atlas cell x laboratory) on the marker surface.

| bar | result |
|---|---|
| no reading exceeds its class-and-substrate ceiling 1/H_min | **0 violations of 492** - PASS |

The guard is arithmetically implied (H <= 1 bit, so A <= 1/H_min), which is exactly why it is worth having as a standing regression test: it fails only if a floor value, a
scale map, an entropy implementation or a beta range has been broken. It now guards all of those on every future change.

## Finding 1 - the presence floor is load-bearing, and this measures how much

Maximum A over 318 **healthy** arrays, class gauge, against each class's ceiling:

| class | max healthy A | ceiling 1/H_min | headroom used |
|---|---|---|---|
| stem_pluri | 1.0181 | 1.0181 | **100.0 %** |
| stromal | 1.1376 | 1.1587 | 98.2 % |
| terminal | 1.1455 | 1.2940 | 88.5 % |
| immune | 1.0660 | 1.1921 | 89.4 % |
| stem_adult | 1.0109 | 1.1446 | 88.3 % |
| progenitor | 1.0263 | 1.1734 | 87.5 % |
| cycling | 1.0100 | 1.1681 | 86.5 % |
| secretory | 1.0205 | 1.1858 | 86.1 % |

On healthy whole blood, **pluripotent stem reads exactly at its ceiling and terminal reads 1.1455 - above the breach line at 1.10.** Neither is a finding about stem cells or
neurons: those classes are absent from blood (Stage 2 fraction 0.000), so their identity loci carry whatever blood does at those addresses, which averages near a coin flip,
which is maximum entropy, which is the ceiling. **Without the measured presence floor, a perfectly healthy blood sample would report two classes past breach.** That is the
strongest justification the presence floor has had, and it is now measured rather than argued. Corollary for reading any report: a saturated or above-breach value on a class
below its presence floor is an artefact of absence, not a severity. The report already refuses those classes; this is why.

## Finding 2 - 1.465 % of mapped betas are unphysical, and the handling was never a declared decision

The Stage 1s pipeline map is `(beta - 0.0662) / 1.0127`. Any raw beta below the intercept maps to a **negative** value: on Uppsala, 146,897 of 10.0 M (1.465 %) go <= 0, and
none go >= 1 (every out-of-range value is a low-methylation locus). Measured effect on the reported class gauge, three ways, against a band width of 0.0524:

| class | A as-is | A clamped to (0,1) | A dropping them | as-is - clamped |
|---|---|---|---|---|
| stromal | 1.1210 | 1.1188 | 1.0439 | +0.0021 |
| terminal | 1.0920 | 1.0916 | 1.0753 | +0.0004 |
| secretory | 0.9630 | 0.9627 | 0.9504 | +0.0003 |
| cycling | 0.9534 | 0.9531 | 0.9403 | +0.0003 |
| immune | 0.9876 | 0.9876 | 0.9875 | +0.0000 |

**Keeping them and clamping them are equivalent** (worst case +0.0021, 4 % of the band). **Dropping them would have been the error** - it moves stromal by -0.077, larger than
the whole healthy band. The class gauge takes the entropy of the *mean*, so an out-of-range value contributes to an average and never reaches a logarithm; the per-cell surface
takes entropies per CpG but does so on **raw** betas, never mapped, so no unphysical value reaches an entropy there either. Conclusion: current behaviour is safe, the margin is
quantified, and the decision is now declared instead of incidental. `cpg_gauge_engine.H` returning 0.0 outside (0,1) is the correct guard for the per-CpG path.

## Finding 3 - a healthy sky is NOT spatially featureless, and the patch null must be spatial

From building the CMB comparison figure (`healthy_sky_vs_cmb.png`): beam-smoothing a healthy sky on the sphere (32 nearest pixels) leaves a smoothed spread of **0.171 against
0.131 +/- 0.001 for the same values spatially shuffled - 1.31x, 57 sigma**. The mottling is real spatial correlation: methylation is correlated along the genome and the sky
places pixels in genomic order, so neighbouring addresses carry correlated residuals. It is a property of healthy biology, not a departure.
**Consequence: any future test that looks for a patch, band or region in a patient's sky must be scored against a spatially-shuffled null, not a Gaussian one**, or this healthy
baseline will be read as a finding. The commissioned per-address test (fraction beyond |z| = 2) is unaffected, being per-pixel. Registered as an open item.

---
**SEALED** sha256 `d57be85e9b49d11d5b09063fcb58c689556c1a5e2bddba3181062d70371b7847` · 2026-09-22
