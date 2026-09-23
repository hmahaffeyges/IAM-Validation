# PROC-SMALL-01 — can the detection limit for a trace class in whole blood be brought below 5 %?

**Pre-registered 2026-09-23, before any configuration was run.** The measured limit is 5 % for secretory and
cycling ([`SMALL_CLASS_DETECTION_NOTE_2026-09-23.md`](../kit/SMALL_CLASS_DETECTION_NOTE_2026-09-23.md)), and
the same note shows why: two of three donors read *exactly* zero at every spike below 5 % because the
non-negativity constraint pins a small component at the boundary, while the one donor whose solution sits in
the interior responds monotonically to every 0.25 % step. The sensitivity exists in the data. This procedure
asks whether three changes recover it, and fixes the bars now so the answer cannot be chosen later.

## What will be changed, and what will not

Three modifications, each already reachable through the deconvolver's own parameters — no new solver:

1. **A detection statistic instead of a point estimate.** From the same class design matrix, refit without
   the candidate class and compare residuals (F-form: the drop in residual sum of squares for one degree of
   freedom, over the full model's mean square). A boundary point estimate is stuck at zero; a residual
   improvement is continuous and can respond below the boundary.
2. **Inverse-variance (generalised least squares) weighting.** `_solve_nnls` already accepts weights; the
   atlas carries a posterior SD for every class at every address, and the chain currently ignores them and
   treats all addresses as equally certain.
3. **Background-contrast markers.** `contrast_pairs=[(class, immune)]` selects addresses by the largest
   |class mean − immune mean| instead of the default between-class variance. For a trace component in blood,
   contrast against the *background* is what carries signal, not separation among the eight classes.

**Not changed:** the immune gauge. The commissioned band rests on 1,379 donors and none of this may enter
that path.

## The bars

| bar | what must hold |
|---|---|
| **B1 sensitivity** | On all three donors, the adopted configuration detects a **2 %** spike of secretory and of cycling — "detects" meaning its statistic reaches the 95th percentile of the same statistic measured on the 40 healthy donors for that class. Two per cent is chosen because it halves the present limit; anything that only reproduces 5 % is not an improvement. |
| **B2 specificity** | At zero spike, all three hosts read below that threshold for both classes. The threshold is computed on the 40 healthy donors, who play no part in choosing the configuration. |
| **B3 the commissioned gauge does not move** | Immune A″ on all 40 healthy donors is identical to the sealed path, to 1e-9. A configuration that shifts the immune reading is rejected outright, whatever it does for the small classes. |
| **B4 quantification (secondary)** | If a fraction is to be *reported* for a detected class, the 5 % spike must recover within ±1.5 percentage points on every donor (the present chain gives 2.0–4.0 %). Failing B4 does not block adoption of detection — it means presence may be reported and a number may not. |
| **B5 no double-dipping** | Configurations are compared on donors A (GSM2333901) and B (GSM2333905) only. Donor C (GSM1051533) and the 40 healthy donors are scored once, at the end. If any configuration is altered after donor C is seen, the comparison restarts on a donor not yet used. |

## Decision rule, fixed now

- **B1 and B2 and B3 met** → the configuration is adopted for detection of trace classes, the presence floor
  for those classes is set from the measured null rather than the typed 0.02, and PROC-SMALL-01 seals as
  COMMISSIONED FOR DETECTION. Reporting a *fraction* additionally requires B4.
- **B1 failed** → the 5 % limit stands, it is published as the measured ceiling for this substrate, and the
  note's conclusion holds: nothing about a trace class in blood is reported. That is a publishable result and
  it will be stated as one, not as a setback.
- **B3 failed by any configuration** → that configuration is discarded regardless of B1.

Scored on the identical 48 mixtures already sealed, plus the 40 healthy donors as the null. No threshold in
this document moves after results are visible.
