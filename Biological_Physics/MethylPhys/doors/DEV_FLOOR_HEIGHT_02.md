# DEV-FLOOR-HEIGHT-02 — the copy channel as a driven steady state: what physics sets its height (derivation, 2026-10-09)

**Status labels.** DERIVED (two-state Markov steady state and its thermodynamics; standard non-equilibrium physics, Hill/Schnakenberg);
numbers CALCULATED from published in-vivo rates; nothing fitted.

**1. The steady state.** A CpG in a methylated domain is lost with probability f per division (maintenance miss) and regained with
probability g per division (de novo / repair). At steady state the unmethylated fraction is ε = f/(f + g), so
**E_hold = kT ln((1 − ε)/ε) = kT ln(g/f)** — exactly, for any rates. The Boltzmann form of the floor is this identity: the holding energy
in kT is the logarithm of the cell's restore-to-loss ratio.

**2. What thermodynamics allows.** The two states have (nearly) equal free energy, so without driving g/f = 1 and ε = 1/2. Holding g/f > 1
needs a driven step; for a two-state system the steady-state ratio is bounded by the driving free energy per renewal,
**E_hold ≤ Δμ** (Δμ = free energy released by one driven methylation: SAM methyl transfer, with SAM made at the cost of one ATP
to PPi + Pi). With Δμ of order one ATP or more (M = 20.94 kT per ATP), the bound sits far above the measured 3.41 kT.
\derived The copy-channel height is **not set by thermodynamics**: the bound is not binding, and cells hold the bit at ≈ 0.16 of the
free energy available per event (φ, PROC-CHANNEL-01), an efficiency. The height is set by kinetics: ln(g/f).

**3. Is the measured height consistent with measured kinetics?** In-vivo per-division rates from double-strand (hairpin) data in human
cells: maintenance ≈ 0.95–0.99 (f ≈ 0.01–0.05) and de novo in methylated islands ≈ 0.05–0.17 (Laird 2004; Fu 2012 Table 2;
Riggs E_m > 0.99, E_d 0.05). \calculated g/f from these spans ≈ 1 to 17, E_hold 0–2.8 kT, ε ≈ 0.06–0.5 at the hairpin loci.
These loci are islands and X-linked regions, not the genome-wide methylated domains our ε reads; the per-locus numbers do not test 3.41 kT.

**4. The test that would.** Genome-wide per-site rates in one cell type: Repli-BS in human ES cells (remethylation rate constants for
> 10 M CpGs; Busto-Moner 2020, PLoS Comput Biol) and whole-genome single-molecule data of the same cells. Prediction: the single-molecule
isolated-error rate of each region equals f/(f + g) from that region's rates (no free parameter). This tests that IAM-A measures
ln(restore/loss) — the instrument's physical meaning — rather than an IAM-specific constant.

**What this means for IAM.** The cell's methylation bit is a driven, far-from-equilibrium bit, like a refreshed memory cell, not a passive
bit sitting at a thermal floor. The holding energy E_hold = kT ln(g/f) is a real, measurable physical quantity and IAM-A reads it; the
"0.2043-bit floor" is the reference height of healthy cells, not a thermal limit. An IAM-specific prediction would have to fix φ (the
efficiency) from first principles; none exists yet.

## 5. The half-law conjecture and the Mahaffey number (author's question, 2026-10-09)
\conjecture If the copying machine runs at its maximum-rate point, it holds half of its driving free energy per renewal:
E_hold = Δμ/2, i.e. Δμ = 2 E_hold = 6.8 kT per site per division (0.33 of one ATP, M = ΔG_ATP/kT = 20.94 at 310 K). M sets the ceiling
(E_hold ≤ M kT at one ATP per bit, ε ≥ 8 × 10⁻¹⁰), the efficiency φ = E_hold/M = 0.163, and the temperature dependence (E_hold = φM ∝ 1/T).
**First check (calculated, order of magnitude).** Every maintained site takes one methyl group per division from one SAM. Methyl transfer
potential from a sulfonium donor at pH 7: −7.5 kcal/mol (amine acceptor, Wolfenden et al. bioRxiv 2021.03.17.435195; the C5-cytosine
value is not sourced) → 12.2 kT; with SAM/SAH = 4–10 in cells, **Δμ ≈ 13.6–14.5 kT per transfer**: E_hold/Δμ = 0.24–0.25, not 0.5.
Counting the full cost of making each SAM (≥ one ATP), E_hold/cost = 0.163. **Not met** at face value: the cell spends about 2× (per
transfer) to 3× (per SAM) what the half-law needs. Open: the C5 methyl-transfer free energy, and whether part of each transfer's free
energy is not available to the discriminating step.
