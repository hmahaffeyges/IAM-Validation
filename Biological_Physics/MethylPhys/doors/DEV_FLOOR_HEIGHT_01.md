# DEV-FLOOR-HEIGHT-01 — the height of the copying floor from the enzyme's own physics (development, 2026-10-09)

**Why.** The floor's form is Boltzmann: a bit held against thermal noise with an energy gap E is lost with ε = 1/(1 + e^(E/kT)). Its height
(3.41 kT, methylated channel; 4.37 kT, unmethylated channel) was MEASURED on Loyfer's healthy cells (PROC-CHANNEL-01). A height obtained
without the cells' copy error makes it a prediction.

## The derivation (DERIVED; standard transition-state physics, Hopfield 1974)
An enzyme choosing between a right and a wrong substrate, whose transition-state energies differ by ΔΔG‡, acts on them in the ratio
S = e^(ΔΔG‡/kT); the fraction of wrong events is ε = 1/(1 + S), so **E_hold = ΔΔG‡ = kT ln S**. For DNMT1, the right substrate is a
hemimethylated CpG (HM) and the wrong one an unmethylated CpG (UM). S = HM/UM specificity, measured in vitro with no cell copy error.
- **Unmethylated channel (de novo error: an isolated methylated call on an unmethylated molecule)** — the channel this maps onto
  directly. Adam et al. 2023 (NAR 51:6622, doi 10.1093/nar/gkad465), 256 flanking contexts: mean 87× on single sites (range 29 to > 300),
  145–180× on long patterned DNA. \calculated E = 4.47 kT (87×) to 5.0–5.2 kT (145–180×); ε = 0.011 to 0.0055.
- **Methylated channel (copy error: an isolated loss on a methylated molecule)** — no discrimination maps onto it. A maintenance miss is
  a failure to act on the right substrate in time (UHRF1 recruitment, processivity, the replication window), not a wrong choice.
  **Open:** no physics-only height for 3.41 kT exists yet.

## Prediction and test
\prediction The de novo error of healthy cells, after the instrument's own false-methylation rate is subtracted, lies at ε = 0.0055-0.011
(E = 4.5–5.2 kT). Uncorrected Loyfer value: 0.0125 (4.37 kT), 0.009-0.016 across 56 cell types; PROC-CHANNEL-01 found it "mostly instrument"
after an ENCODE correction — so the uncorrected agreement is not evidence. Test: libraries that carry an unmethylated spike-in (lambda)
give the instrument's false-methylation rate per run; subtract it from the de novo error of the same run. Above 0.011 or below 0.0055
after correction: the enzyme-physics height fails for that channel.
**Status.** Standard enzyme physics, not specific to IAM; what IAM adds is the claim that every held bit sits at its Boltzmann gap, with
the same reading at every scale. The methylated-channel height remains open.
