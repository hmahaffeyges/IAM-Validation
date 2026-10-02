# Missing Satellites (March 2026) — read in full myself (confirmed 2026-10-02 in 50-line chunks, PDF text 574 lines, ledger complete); equations recomputed; σcrit tested against the satellite census (2026-10-01)

**Eq. 11 recomputed** (Ω_m 0.3153, H0 67.36): M_min(σ = 4 km/s) = 10^8.44 M☉. Eq. 12 with M_min = 10^8.4 gives σ_crit = 3.9 km/s.
So Eq. 11 as printed already lands on the observed floor. §4.4's "raw prediction gives 10^6.4, ~100× below" does not follow from Eq. 11;
the stated 100× offset is not in the paper's own equation (to be traced in the dated prediction script, tests/iam_small_scale_structure_prediction.py).
Eq. 10 taken directly (t_dyn = 1/H0) gives √(6/π) σ³/(G H0) = 10^8.48 at 4 km/s; the 4Ω_m prefactor (1.26 vs 1.38) is not derived in the text.
**Citation:** the 10^8.4 M☉ occupation floor is attributed to Kim & Peter (2021), a paper on SIDM signatures in cluster mergers. The source needs
correcting; the correct reference for a ~10^8.4 M☉ satellite-occupation floor is still to be found and verified.

**σ_crit tested** (Local Volume Database, Pace et al., dwarf_mw.csv; 68 Milky Way satellites, 54 with a dispersion or upper limit):
25 satellites have σ or its upper limit below 4 km/s; 24 of them are flagged confirmed galaxies; 26 have σ ≥ 4 km/s.
19 have a resolved dispersion below 4 km/s (lower error bound > 0), e.g. Crater II 2.34 (+0.42/−0.30), Hydrus I 2.70, Bootes II 1.92, Hercules 2.25;
6 more have only upper limits (Tucana III < 1.5, Grus II < 2.0, Segue 2 < 2.06, Draco II < 2.6, Triangulum II < 3.5, Aquarius III < 3.5).
Many carry spectroscopic metallicity spreads (0.2–0.6 dex), the standard signature of a dark-matter-hosted galaxy.

**Result, by the paper's own falsification criterion (§6: "If a significant population of satellites below σcrit is discovered, the dispersal
condition is inconsistent with observations and Mechanism B requires revision"):** the population already exists in published data — 46 % of
satellites with kinematics sit below 4 km/s. Mechanism B as stated (no halo virializes below σ_crit ≈ 4 km/s) is rejected.
Caveats a referee would accept: tidal stripping lowers present-day σ (Crater II, Tucana III), so σ today is not σ at formation; a test against
σ at infall (or peak circular velocity) is the fair version, and Eq. 11's σ is not defined as either.
**Mechanism A** (§3): ΔD/D(z=0) = −7.4 % does not reproduce under any implementation. At fixed parameters: −0.78 % with G_eff = µG (Level 1), −0.67 % with the matter friction (Level 2), −1.87 % with the whole growth equation on H_IAM; in the chains σ8 falls 1.6 % (L1) and 1.1 % (L2). **Other:** "all 17 converged R−1 < 0.01": at the paper's date 14 of 17 had (today all 18 are ≤ 0.010); "Euclid DR1 October 2026" → mid-2027; β_m posterior 0.1583 is the derived Ω_m/2 (β_m fixed in the chains), not a recovery.
**Status:** Mechanism B as stated is rejected by the census; the infall-σ version is the open re-test. Mechanism A: direction stands, amplitude ~1 %.
Book inclusion: the author's decision (the paper is on the 2026-10-02 reading list, G5 #24).

## Added on the confirmation read (2026-10-02)
- §2 "the potential half deposits into spacetime curvature; the kinetic half is the Landauer cost": the book's statement is IAM's Law with the virial
  partition setting how much bound matter writes (Part 1); this paper's two-channel wording is not carried.
- §2, §3.3, Table 1, §6 "β_m = 0.1583 ± 0.0033 confirmed at 0.2σ": β_m is fixed in every chain. "17 chains" → 18. "Euclid DR1 October 2026" → complete
  DR1 mid-2027; σ(µ0) ≈ 0.04 is the full-survey forecast. "5.4σ with DESI Y5" is from the Survey Predictions paper (SP5: unsourced).
- §4.3, §7 σ³ / σ² / σ⁴ family: M–σ is abandoned (author) and cusp-core is a side test; the "coherent family" is not carried.
- §4.1, §7 "the informational half must be written to a local black-hole horizon": the cosmic horizon is always available in IAM; the paper's
  own reason for needing a local surface (t_dyn ≪ 1/H) is not derived, and it fails the census below.

## For the book (author rule 2026-10-02: only what is accurate or still a live prediction; failed side predictions stay out)
- Carried: none as a chapter. Mechanism A's amplitude is the growth result already in the S8 chapter (ΔD/D −0.78 % today, exact coupling).
- Not carried: Mechanism B as stated (rejected by the Milky Way census by the paper's own §6 criterion).
- Predictions appendix (future test): σ_crit against σ at infall / peak circular velocity, once the dispersal condition is derived.
