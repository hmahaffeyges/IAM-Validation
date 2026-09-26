# PROC-PARTIAL-01 — can a non-blood class's fidelity score be recovered from an ordinary blood draw?

**Pre-registered 2026-09-25, before any recovery was attempted.** Written on the author's standing point,
which is the right one: *finding secretory cells in blood is not the same as scoring them, and a composition
number without a fidelity score does not mean much.*

## The question, stated exactly

The chain will tell you that 4 % of a blood specimen is secretory. It will not tell you **how well that
secretory material is holding its state**, and the A-score is the whole point of the instrument. Today it
refuses, and the refusal is correct: the betas at secretory's identity loci in whole blood are ~95 % immune
material, so their entropy is the entropy of *blood measured at secretory's addresses*. The quantity is not
identified, which is a different and more serious problem than a missing band.

**Is it recoverable?** The chain already computes the machinery. Its sky subtracts the composition's
prediction from the observation; leave one class *in* that subtraction and what remains is that class's own
contribution:

&nbsp;&nbsp;&nbsp;&nbsp;`mu_hat_k = (beta_obs − Σ_{c≠k} f_c · mu_c) / f_k`

with the chain's **own fitted fractions**, not the true ones — because in a real specimen that is all you
have. Its entropy over class k's identity loci, divided by class k's floor, is a candidate fidelity score.

**And the arithmetic sets the price.** Dividing by `f_k` amplifies every error by `1/f_k`: 5× at f = 0.20,
20× at f = 0.05, 50× at f = 0.02, against a per-address between-person spread measured at 0.029
(PROC-DIFF, 2026-09-22). Per locus that is hopeless. Averaging entropy over tens of thousands of addresses
divides the noise by √n, which is the only reason this has any chance. **Whether it survives at 5 % or only
at 20 % is exactly what this measures.**

## What enters

| | |
|---|---|
| Specimens | the **5,088 mixtures** of PROC-FOREIGN-01 — 318 healthy hosts × stromal, secretory, terminal × f = 0.02…0.35 |
| Why these | the spiked material is the **atlas class mean**, so the true profile is known exactly, which makes recovery measurable rather than arguable |
| Estimator | partial residual as above, using the fractions the chain's own deconvolver fitted |
| Truth | `A_true = H(mu_atlas over class k's identity loci) / H_min_k` — the A-score of the material that was actually mixed in |
| Scale for "close enough" | σ = 0.02092, the immune band's own width, the only measured healthy spread available. No new tolerance is invented |

**The limitation that decides how a pass may be read.** Because the spiked material *is* the atlas mean,
this measures the estimator's **arithmetic recovery under ideal conditions** — no donor variation, no
tissue heterogeneity. Real secretory tissue differs from the atlas mean, so a pass here is an **upper bound
on performance**, not a clinical claim. A failure, by contrast, is conclusive: an estimator that cannot
recover the exact profile it was given will not recover a noisier one.

## The bars, fixed now

**B1 — the recovered score is accurate.** At some f ≤ 0.20, across the 318 hosts, the median
`|A_hat − A_true|` must be **≤ 0.0209 (1σ)** and the p10–p90 spread of `A_hat` must be **≤ 0.0418 (2σ)**.
Both conditions, or the fraction does not qualify.

**B2 — the score belongs to the material, not the host.** The same spiked material in 318 different hosts
must recover the same score: the interquartile range of `A_hat` at the qualifying f must be **≤ 0.0209**.
A reading that moves with whoever the material was found in is a reading about the host.

**B3 — the deconvolution is what buys it.** The naive score — entropy of the raw betas at class k's identity
loci, no subtraction — must be **further from `A_true`** than the partial-residual estimate, at every f that
qualifies. If the naive number is as good, the machinery adds nothing and should not be shipped.

**B4 — it refuses when it cannot see.** At f = 0 (the unspiked hosts) the estimator must produce **no
reading at all**, by a rule fixed now: no fidelity score is computed when the fitted fraction is below the
qualifying f from B1. A 1/f estimator with f near zero returns noise scaled by infinity, and the guard
against that is a refusal, not a wide interval.

**B5 — nothing in service moves.** Immune A″ on the 318 unspiked hosts must equal PROC-BAND-01's published
values, **max |ΔA″| = 0**, verified by recomputation and reported as a number.

## Decision rule

- **B1–B5 met at some f:** the smallest qualifying f becomes the **reporting floor for non-blood fidelity
  scores**. Above it the chain may report a class's A-score with its floor stated; below it the chain reports
  the fraction and refuses the score, as it does today. This changes what the instrument reports, so it also
  requires the two canonical documents to describe it before it ships.
- **No f ≤ 0.20 qualifies:** published as the measured amplification limit — non-blood fidelity scoring is
  not available from whole blood at the fractions blood actually contains, and the reason is arithmetic
  rather than a missing calibration. That is worth as much as a pass, because it closes a question that
  would otherwise be asked repeatedly.
- **B3 fails:** the estimator is not doing the work; the simpler quantity is reported instead, if anything is.

No patient specimen is involved and nothing here reports on disease.
