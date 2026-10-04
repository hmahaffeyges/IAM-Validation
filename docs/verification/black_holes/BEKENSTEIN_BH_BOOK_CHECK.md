# BEKENSTEIN_BH_BOOK_CHECK — additions found while carrying the three G6 papers into the book (2026-10-02)
Papers: Bekenstein_coefficient (596 lines), IAM_Black_Hole_Information_Paradox (453), IAM_BH_Thermodynamics (431), read in full in <=50-line chunks
(last Info-Paradox chunk 401-453). Script: `scripts/verify_bekenstein_bh.py` (44 of 44 checks pass; output in `_output.txt`).
Chapters: `docs/book/part3/p3_01_blackholes.tex`, `docs/book/part3/p3_01a_bekenstein.tex`, `docs/book/part3/p3_01b_bh_information.tex`.

## Confirmed again (existing rows)
- B1: sigma_SB A T^4 = hbar c^6/(15360 pi G^2 M^2) symbolically; numerical ratio 1.000000000000 at 1, 10, 1e6, 1e9 Msun (by construction).
- B2: S_tr/S_0 = 1 - (1 - t/tau)^(2/3); half at (1 - 2^-1.5) tau = 0.6464 tau (symbolic and quadrature); S_0(1 Msun) = 1.513e77 bits.
- B4: Clausius in SI gives hbar c eta/(2 pi) = c^4/(8 pi G), eta = c^3/(4 hbar G) = 1/(4 l_P^2); G = c^4/(4 hbar eta) carries an extra velocity;
  c^4/(4 hbar G) = c x 1/(4 l_P^2). Planck surface gravity c^2/l_P = 5.56e51 m s^-2 is the largest.
- B6: Mc^2 = k_B T ln2 S^(1/2) fails by ~39 orders at 1 Msun (1.9e8 J vs 1.8e47 J); Smarr 2 T S = Mc^2 to 1e-12.
- V17, V20: T_BH S_BH = Mc^2/2; S_BH = 4 pi G k_B M^2/(hbar c).

## New (for the author; applied in the chapters, minimal wording)
1. **Bekenstein §6.1 Eqs. 17-21 (per bit vs per nat; first law).** k_B T_H is the cost of one nat, not one bit; the area response is the first law
   delta(Mc^2) = (kappa c^2/8 pi G) delta A, so delta A = 8 pi G delta E/(kappa c^2). Eq. 18, (G/c^4) E 8 pi, is a length, not an area. With the first law
   delta A_min = 4 l_P^2 per nat for ANY kappa (kappa cancels); one bit = 4 ln2 l_P^2 = 2.77 l_P^2. Checked for Schwarzschild and Kerr (fixed J).
2. **Bekenstein §2.1** "each event produces I = log2 N bits": the record entropy is -sum |c_i|^2 log2 |c_i|^2 <= log2 N, equal only for equal weights.
3. **Bekenstein §3.1** KMS line "beta = 1/(k_B T) = 2 pi/(hbar kappa)": with kappa an acceleration, hbar/(k_B T) = 2 pi c/kappa.
4. **BH Thermodynamics §7** "each three-order-of-magnitude increase in information content produces a three-order-of-magnitude increase in seed mass":
   M ∝ sqrt(S), so six orders in S give three in M (the paper's own Table 3 shows this). Table 3 recomputed: 0.976, 976, 9.76e5, 9.76e8 Msun.
5. **BH Thermodynamics §4.3, §11, Table 4** "all real black holes radiate at the full Hawking rate throughout cosmic history": true against the
   cosmic-horizon bath ((T_GH/T_BH)^4 = 3.4e-90), but the CMB at 2.7255 K exceeds T_BH for every stellar and supermassive hole today (net absorbers;
   balance at M_CMB = 4.5e22 kg). Chapter text says so.
6. **BH Thermodynamics Table 2**: T_GH(z = 1) = 4.755e-30 K (printed 4.76e-30); other entries reproduce.
7. **BH Thermodynamics Table 4**: M_lens/M_dyn = 1/mu(0) = 1 + beta_m = 1.158 (15.8 %) with beta_m = Omega_m/2 = 0.15765 (printed 15.7 %); agrees with
   ch:threeway (R(0) = 1.158).
8. **Info Paradox §6**: the envelope min(S_tr, S_BH(t)) that bounds the fine-grained entropy turns over at the same 0.646 tau in the black-body
   accounting; real emission is irreversible (radiation coarse entropy > horizon entropy lost), so the Page time is earlier (Page 2013).
9. **Info Paradox abstract** "17 converged MCMC chains": count not reproduced here; the book refers to the chain record of ch:dual.

## Not carried (confirmed corrections)
- Bekenstein §6.4 (2 pi/8 pi linked to the CC 2/pi and Koide pi/2): BLACK_HOLES_CHECK #11; CC 2/pi not derived (errata C14).
- Info Paradox Eq. 8 and "particles and black holes are the same fixed-point class": B6.
- M–sigma passages (Info Paradox §8.1, the M–sigma sentence of §8.2; BH Thermodynamics §6.3 clause, §9.2): B3, M1.
