# R1: the virial share for driven systems (desk derivation, as it stands)

**A. Stationary virial with a drive term (Clausius 1870; standard; the external-pressure form is textbook astrophysics).** 2K = n|W| + S.
- check `S = 0, n = 1 gives K = |W|/2 (the book's half)`: PASS
**B. IAM's steps 1-3, isolated (E_i = 0, no work from outside):** Q = |W| - K = E_Landauer.
- check `isolated, 1/r: Q = K = |W|/2 (eq:vi_identity)`: PASS
**C. Driven or confined: the surroundings do work W_in during binding.** First law: Q = W_in + |W| - K. Written share s = Q/|W|.
  s = (-S + W + 2*W_in)/(2*W)  (n = 1)
- check `(a) W_in = 0: s = 1/2 - S/(2|W|)`: PASS
- check `(b) W_in = S: s = 1/2 + S/(2|W|)`: PASS
  (a) W_in = 0 (pressure confines, no net work): s = 1/2 - S/(2|W|), BELOW one half.
  (b) W_in = S (infall brings in work equal to the surface term): s = 1/2 + S/(2|W|), ABOVE one half.
  The two bookkeepings move the share in OPPOSITE directions; which holds is a question about the halo's actual energy flow (what the infall delivers and where the heat goes), so it is NOT decided here.
**D. With the book's measured halo virial ratios (2K/|W| = 1.1-1.3; S/|W| = 2K/|W| - 1):**
| 2K/|W| | S/|W| | share (a) | beta_m (a) = Omega_m s | share (b) | beta_m (b) |
|---|---|---|---|---|---|
| 1.1 | 0.1 | 0.450 | 0.1419 | 0.550 | 0.1734 |
| 1.2 | 0.2 | 0.400 | 0.1261 | 0.600 | 0.1892 |
| 1.3 | 0.3 | 0.350 | 0.1104 | 0.650 | 0.2049 |
For comparison only, NOT a target: the settled value 0.15765; the cell's phi 0.147 (its own drive, M = 20.94); the free fit 0.155 +/- 0.029.

**What this shows (derived):** the one-half is the stationary virial with no drive term; Clausius' theorem itself holds for driven stationary systems with a drive term S, and cosmic halos have a measured S of 10-30 % of |W|. So beta_m = Omega_m/2 is the S = 0 case, and a real halo population departs from it by an amount the book already has data for.
**What is open:** (1) how the drive enters IAM's step 1 (W_in), which sets the SIGN of the shift: the two bookkeepings disagree; (2) how halo-scale shares add up to the cosmic beta_m (mass-function weighting and epoch: the book's eq:vc_decompose with f_coll, eta_vir); (3) which halos' S (raw or surface-corrected; the book: 'a property of the halo sample chosen, not a constant').
**The cell (same ring thread):** the cell has no 1/r binding, so its share is set by first passage under the paid push (phi = E_hold/M, -> ln M/M for a strong push) rather than by a degree n. Common statement candidate: the kept share is the restoring law's own value (n/2 for a power law; first passage for a paid hold) shifted by the drive. NOT derived; the two are not shown to be one formula.
**Guard:** no number above is fitted; beta_m is not moved toward any value. The sign question (1) is decided from the halo energy flow before any value is compared with data.
