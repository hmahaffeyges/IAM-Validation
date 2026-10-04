# Cosmological constant derivation — recomputation (2026-10-01)

Source: the cosmological-constant paper. Parameters from the 18th chain
(Cosmological_Physics/mgcamb_validation/chains/iam_baryon_test.1.txt, BBN prior on Ω_b h² removed, 30 % burn-in, 14,957 weighted rows).

## The 18th chain (CMB only, no BBN prior)
| | posterior |
|---|---|
| Ω_b h² | 0.02232 ± 0.00014 |
| H0 | 67.04 ± 0.53 |
| Ω_m | 0.3197 ± 0.0075 |
| Ω_Λ | 0.6802 ± 0.0075 |
| Ω_b/Ω_m | 0.1554 ± 0.0019 |
| η = 273.9×10⁻¹⁰ Ω_b h² | (6.113 ± 0.037)×10⁻¹⁰ — observed 6.137 |

Both the baryon asymmetry and the CC relation are read from this one posterior; they share parameters and are not
independent tests.

**Settings, from the chain's own yaml (iam_baryon_test.updated.yaml, MGCAMB/Cobaya 3.6, CAMB 1.5.2):**
- Ω_b h²: flat prior 0.010–0.040 — **no BBN prior**. Data: Planck 2018 low-ℓ TT, low-ℓ EE, high-ℓ plik-lite TTTEEE, lensing. No
  BAO, no SNe, no H0 prior, no deuterium or helium abundance data.
- μ0 fixed at −0.13495 (IAM); H0, τ, ln A, n_s, Ω_c h² free; Rminus1_stop 0.01.
- **Helium:** YHe is not a sampled or fixed parameter, so CAMB sets it from Ω_b h² with its built-in BBN consistency relation
  (standard practice in Planck analyses). Nuclear physics therefore enters only through the helium fraction used to compute the
  CMB damping tail, not as a constraint on Ω_b h². To remove even that, re-run with YHe fixed (e.g. 0.245) or free.

## What holds
ρ_Λ/ρ_vac = (3Ω_Λ/8π)(l_P/l_H)² identically (critical density in Planck units). The paper's formula therefore reduces to
**Ω_b/Ω_m = (3/16)√Ω_Λ**. On the 18th chain: ratio 1.0049 ± 0.0068; difference 0.0008 ± 0.0011 (0.7σ). Holds at the present
epoch only (Ω_Λ(a) changes, Ω_b/Ω_m does not).

## What does not
1. The 10⁻¹²³ cancels on both sides; it is H0 in Planck units, an input.
2. 2/π: with A_eff = 2π l_H², l_P²/(A_eff/4π) = 2(l_P/l_H)², not (2/π)(l_P/l_H)².
3. History integral, eq. (integral) as written, evaluated from a_EW = 2.3×10⁻¹⁵: coefficient 3×10³⁰ against the required 0.523;
   the integrand grows as a⁻³ at early times, so the earliest epoch dominates.
4. Weighting the same quantity by the activation function's growth dℰ(a) gives O(1) coefficients that depend on the form
   chosen: ∫(H0/H)² dℰ = 0.470, ∫(H0/H) dℰ = 0.641, ∫dℰ = 1, ∫(H/H0) dℰ = 1.98, ∫(H/H0)² dℰ = 5.80. None is selected by a
   derivation; picking the closest (0.470, 10 % low) after seeing the target would be a fit.

## Status for the book
Present the relation Ω_b/Ω_m ≈ (3/16)√Ω_Λ as an observed present-epoch relation (0.5 %, 0.7σ on the CMB-only chain), with the
derivation open: the 2/π factor and the history-weighting both need a derivation fixed before comparison.

## The baryon paper (Baryon_Asymmetry_as_a_Derived_Quantity…pdf), read in full and checked
1. **The chain does not test IAM's constraint.** Eq. (3)/(5) is not in the chain; the chain measures Ω_b h² from the CMB acoustic
   peaks, as any Planck fit does. The ΛCDM baseline chains in the repo (no μ0, flat Ω_b h² prior 0.020–0.025) return the same
   value: lcdm_baseline 0.02234 ± 0.00014 → η = 6.120; planck_bao 6.139; planck_pantheon 6.119; planck_rsd 6.136 (×10⁻¹⁰).
   The 18th chain (IAM, prior 0.010–0.040) gives 0.02232 ± 0.00014 → 6.113. Widening the prior and fixing μ0 change nothing.
2. **The "standard configuration" had no BBN prior either.** N(0.02242, 0.00014) is Cobaya's `ref` (starting-point) distribution,
   not a prior; the prior in every Level 1 yaml is flat 0.020–0.025. Planck's own Ω_b h² is a CMB-only measurement.
3. **CMB–BBN agreement on η is established concordance** (WMAP, Planck), not new; the paper's Table 1 lists it (6.136 ± 0.038).
4. **The analytic numbers do not reproduce.** Inverting eq. (3) with Planck values gives Ω_b h² = 0.01839, η = 5.04×10⁻¹⁰
   (paper: 6.079). Inverting eq. (5) gives 0.02222, η = 6.09×10⁻¹⁰ (paper: 6.115).
5. **What survives:** eq. (5) inverted gives η = 6.09×10⁻¹⁰ against the CMB 6.11–6.14 (0.4–0.8 %). This is the same single
   relation, Ω_b/Ω_m ≈ (3/16)√Ω_Λ, read in a different variable. The baryon asymmetry and the cosmological constant are one
   observed relation, not two confirmations; its derivation is open (2/π, history weighting).

## Full reads with line ledgers (2026-10-02): CC paper 842 lines, Matter-Antimatter 438, Baryon 242, 18th-chain record 7
Frame (GRF essay, read in full 2026-10-02, 212 lines, 1–212, last line = page number 9; Theory paper): not modified gravity; Einstein equations untouched; the horizon entropy
functional gains S_info; off before structure, so the early universe is ΛCDM. Reproduction: `scripts/verify_cc_and_baryon.py` (output beside it).
Every number below is from that run.

### Reproduced
- ρ_vac = E_P⁴/(ħc)³ = 4.633 × 10¹¹³ J m⁻³; ρ_Λ = 5.250 × 10⁻¹⁰ J m⁻³; observed ratio 1.1332 × 10⁻¹²³ (paper 1.14).
- Baseline (2/π)(l_P/l_H)² Ω_b/Ω_m = 1.3804 × 10⁻¹²³ (×1.218); with √Ω_Λ 1.1421 × 10⁻¹²³ = **+0.79 %** (paper "0.07 %": both rounded to 1.14).
  Exponent that would close the baseline exactly: Ω_Λ^0.521 (paper 0.502).
- 18th chain: 14,957 rows after 30 % burn-in (22,400 accepted, R−1 0.009273 in the record); Ω_b h² 0.02232; η (6.113 ± 0.037) × 10⁻¹⁰;
  Ω_b/Ω_m 0.1554; ratio to (3/16)√Ω_Λ 1.0046 ± 0.0068. The record's 6.1155 uses the full chain (no 30 % cut on the record's own count).
- ΛCDM chains give the same: lcdm_baseline η 6.118, ratio 1.0058; Planck + BAO 6.137, 1.0106; Planck + Pantheon+ 6.117, 1.0056.
  Expected: the informational term is off at recombination (E → 0), so GR with the term is ΛCDM there and the CMB returns the same
  parameters. The (3/16)√Ω_Λ relation is a property of the measured universe; the chain is a consistency check that the early universe is intact.
- (Ω_dm + Ω_de)/Ω_b = 19.39 (MA eq. 5).
- De Sitter horizon: T_GH S_BH = M_H c² exactly for any H (the cosmic analogue of Smarr). So ρ_Λ = Ω_Λ × (horizon entropy × T_GH)/V_H
  identically — the 10⁻¹²³ is this identity, not a result.

### New corrections from the full reads
10. **CC §2.1:** N_max = A_H/l_P² → A_H/(4 l_P²) (Bekenstein–Hawking); §3.1 f_geo = l_P²/A_H = (1/4π)(l_P/l_H)², and eq. 7 then prints (2/π) with
    no step between them.
11. **CC §5.1 "the static patch subtends exactly 2π sr of the horizon":** the static patch of a de Sitter observer is bounded by the whole horizon
    sphere (4π sr); the observer receives signals from all of it. The Rindler analogy (half of Minkowski) concerns spacetime regions, not
    solid angle. With the paper's own A_eff = 2π l_H², eq. 26 gives 2(l_P/l_H)², not (2/π)(l_P/l_H)².
12. **CC §5.3 "holographic round trip 2/π × π/2 = 1" and the Koide link:** not a property of the holographic principle; not carried.
13. **CC §3.3 √Ω_Λ:** T_dS = T_H√Ω_Λ is the Gibbons–Hawking temperature of the pure de Sitter horizon set by Λ alone (correct); applying it as a
    factor on the baseline was introduced after the 1.22 gap was seen (formula timeline: ×Ω_b/Ω_m 16 Mar, ×2/π 16 Mar, ×√Ω_Λ 20 Mar;
    READERS_REVIEW G4 synthesis). "Not fitted" is not supported; say "introduced to close the 1.22 residual".
14. **CC §6.2** argues Ω_b/Ω_m was derived before use; the record shows it was added when the gap stood at 12 (to 1.9). Stated as the record shows;
    the physical argument (only baryons decohere) is given as the paper's reason.
15. **CC §6.3 "the probability of recovering 1.14 × 10⁻¹²³ within a factor of 2 by accident is vanishingly small":** the 10⁻¹²³ is the identity in
    the reproduced list, so any O(1) prefactor lands within a factor of a few. The meaningful count is for O(1) combinations of Ω_b/Ω_m, Ω_Λ, π
    and small integers: 2 of 540 within 1 % (READERS_REVIEW look-elsewhere count).
16. **CC §7 "w > −1, consistent with DESI DR2":** an accumulating ρ_Λ grows with time, so ρ_de increases: that is w < −1 (phantom), not w > −1.
    DESI DR2 prefers w₀ > −1 today with w_a < 0. Sign corrected; comparison with DESI is open.
17. **CC §8 / MA eq. 2 history integral** (as in items 3–4): diverges as a⁻³ from a_EW; normalisation never fixed.
18. **CC §2.1, MA §2 "β_m confirmed by 17 chains at 0.2σ without fitting":** β_m is fixed in every chain (VIRIAL_CHECK #1). **MA §2 "virial ratio
    Ω_m/[β_m E(a)] asymptotes to exactly 2 at a = 1":** equal to 2 at a = 1 by the definition β_m = Ω_m/2, E(1) = 1.
19. **MA §7 QCD-epoch numbers:** l_H ≈ 15 km = 5 × 10⁻¹³ pc (paper 10⁻³ pc); horizon bits 2.9 × 10⁷⁸ (paper 10⁴⁰); a_QCD ≈ 1.0 × 10⁻¹²
    with entropy conservation (paper 1.6 × 10⁻¹²).
20. **MA §5 "the 10⁹ annihilation partners constitute the dark sector":** annihilation energy is in the CMB photons (standard, measured); dark
    matter is present at its measured density at recombination (z ≈ 1100), before structure; 19.4 is not related to 10⁹. Not carried in Part 2.
21. **MA §6 weak force as "force of becoming", EW breaking as "origin of duration", MA §5 "95 % memory":** philosophy (ruled out); not carried.
22. **MA §8 and BA §3:** same as C4–C9 (no BBN prior existed; six times wider; sampled parameters; no IAM constraint in the chain).
    MA §8.3 "IAM posteriors at < 0.1σ" is a stray line. MA §10 repeats a paragraph.
23. **"Prediction stated in advance" (MA ref. 2026g, BA ref. 2026c):** the advance statement was that a CMB chain with Ω_b h² free would return
    η ≈ 6 × 10⁻¹⁰. Every CMB fit does (ΛCDM chains above), so the test could not have failed for IAM-specific reasons. (Resolves C12.)
24. **BA eq. 6 / §2:** the "Python pre-test 6.079" is eq. 5 (with √Ω_Λ) at Planck values, 6.080 here; eq. 3 gives 5.03. The paper labels them
    the other way round (C3).
25. **References:** MGCAMB is Wang et al. 2023 JCAP 08, 038 (as P16), not "JCAP 2023, 022"; CC ref. list labels 2026d/2026e swapped against the text.

### The accumulation test (planned 2026-10-01; run 2026-10-02, no free factors)
Sheth–Tormen halos (M > 10⁸ M☉, Eisenstein–Hu P(k), σ8 0.811), virial kinetic energy per unit mass (3/10)GM/r_vir, summed over all baryons in
halos today = heat radiated by baryonic virialisation (virial theorem: radiated = stored K).
- Baryon mass in halos 0.58; σ_eff 213 km s⁻¹; **accumulated virial heat / ρ_Λc² = 3.2 × 10⁻⁸.**
- Priced at the horizon (bits = Q/(k_BT_gas ln 2), cost k_BT_GH ln 2 each): 2.6 × 10⁻⁴⁴.
- Upper bound, all baryon rest mass: Ω_b/Ω_Λ = 0.072.
**Result:** heat radiated by baryons is 3 × 10⁻⁸ of ρ_Λc², so radiated heat is not the measure of the accumulated cost. The informational
term's size is set by the virial partition of the matter itself (β_m = Ω_m/2, activated by E(a)), a horizon-entropy term in the first law.
The history integral (items 3–4) must be written in that variable. Frame (author, 2026-10-02): Λ is the vacuum baseline before structure and
accumulates as structure forms and decoherence accelerates; the GRF essay's "baseline" and the CC paper's "accumulated" are the same picture at
two epochs. Approved wording kept: "Within GR with the informational term, Λ is the accumulated Landauer cost of baryonic decoherence…"

## Excluded from the book (author, 2026-10-02: "keep anything out that is wild speculation")
- MA §5, §10: the 10⁹ annihilation partners as the dark sector; "95 % memory, 5 % present".
- MA §6: the weak force as the "force of becoming"; electroweak breaking as "the origin of duration"; the CKM phase fixed by the loop.
