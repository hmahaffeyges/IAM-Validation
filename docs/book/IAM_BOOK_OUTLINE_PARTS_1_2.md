# Law and Order: The Thermodynamics of Informational Actualization

Master outline, Parts 1 and 2 (draft for the author's approval, 2026-10-01)

Source: every paper in `docs/papers/` (47 PDFs). Dates are those printed in each paper. 'Stated predictions' counts entries in
`CANON/predictions_register.csv` that cite the paper (as stated by the source, not re-verified). Roles: **core** = carries a result the
chapter needs; **supporting**; **interpretation** = printed with that label, kept apart from derivations; **record** = history of testing;
duplicates are merged. All constants come from `CANON/iam_canon.json`. No chapter cites anything outside physics.

Parts (author's titles, 2026-10-01):
1. Introduction to IAM's Law and the Virial Theorem
2. Informational Actualization of GR: The Cosmological Dynamic
3. The Quantum Order of Informational Actualization (qubits and semiconductors)
4. Cellular Physics: Thermodynamics of the Methylome (human cells, salmonids, methods from the sky)
5. 37 Orders of Magnitude and the Web that is IAM


## Already written: *IAM's Law and the Quantum Order* (the quantum/semiconductor book)

That book already covers the law and much of the cosmology, with every number recomputed. Its chapters move into the omnibus
as written and are **expanded, not rewritten**. Its recomputed values (Appendix B, Ch. 22) are the values used in every chapter below.

| existing chapter | omnibus place | what Parts 1–2 add |
|---|---|---|
| 1 From the horizon to the laboratory | 1.1 | — |
| 2 The statement of the law (postulate, encoding temperatures, A, M, virial face, two ledgers) | 1.1, 1.3, 1.4 | Variational derivation; 37-orders table in full |
| 3 The Bekenstein coefficient | 1.2 | Jacobson / Cai–Kim background; Saridakis bridge |
| 4 Encoding surfaces across sixty decades | 1.3 | — |
| 5 The cosmological constant | 2.8 | w(z) far future |
| 6 The dual-sector model (MCMC record, two H0) | 2.1–2.3 | β_m and E(a) derivations (holographic, decoherence bridge); CAMB and Planck methods; S8, DESI, lensing, clusters, satellites, forecasts — **not in the existing book** |
| 7 The baryon asymmetry | 2.9 | — |
| 8 Duration, the arrow of time … | 2.11 | — (retitle per decision 1) |
| 9 Electron rest mass; 10 Koide; 11 Higgs | 2.12 | — |
| 12 Measurement; 13 Gravitational decoherence; 14 Nonlocality | 2.11 | GRF essay; E_q(t) predictions |
| 15–21 Qubits, x_qp, A for processors, thermal n, coherence walls, CMOS, saturation events | **Part 3** | — |
| 22 Status table; 23 Open problems and falsification | **Part 5** (merged across all parts) | cell and salmon rows when commissioned |
| A Constants; C Reproduction | shared appendices | constants come from CANON; the recomputed values of Appendix B go straight into the chapters |

**Consequences for Part 2.** The genuinely new cosmology chapters are 2.2–2.7 (μ–Σ, CAMB/Planck, H0/S8/DESI, lensing and clusters,
small scales, survey forecasts) and 2.10 (black holes). Everything else exists and is extended. Every headline number is re-run from the
repo's chains and only the re-run value is printed (current Level 2 result: Δχ² = +0.54 against ΛCDM, consistent with the data).

## Part 1 — Introduction to IAM's Law and the Virial Theorem

### 1.1 What IAM measures

The thermal floor as a measured reference, in every domain; what IAM adds to Landauer; scope (physics only).

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| The Informational Actualization Model   A Technical Reference for Physicists | September 2026 | 28 | core (newest statement, Sept 2026) | 14 |
| IAM Law | March 2026 | 29 | core (law statement) | 42 |
| IAM Overview Companion | February 25, 2026 | 7 | background | 9 |

**Recompute before printing:** law statement and every constant from the canon (M = 20.94).

### 1.2 Gravity from thermodynamics

Jacobson 1995, Cai–Kim 2005, Bekenstein–Hawking entropy: the established physics IAM builds on, cited to the originals.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM Master Preprint | March 2026 | 25 | core | 52 |
| A Note on Entropic Gravity  Saridakis  | March 2026 | 10 | core (relation to Barrow/Tsallis) | 12 |
| Bekenstein coefficient | April 2026 | 11 | core | 3 |
| iam bekenstein coefficient | April 2026 | 11 | same paper as above; one copy used | 1 |

**Recompute before printing:** the Bekenstein coefficient derivation step by step.

### 1.3 The virial theorem and IAM's thermodynamic step
**Drafted (2026-10-02):** `docs/book/part1_drafts/p1_virial_law.tex` — theorem and why ½, atom-to-cluster table (no N-body row), thermodynamic identity with its scope (systems that release the binding energy; hydrogen 13.6 eV). Placement confirmed by the author.

2⟨K⟩+⟨V⟩=0 (Clausius 1870); IAM's added identification ⟨K⟩ = Q = TΔS = E_Landauer, labelled as interpretation until tested; the 37-orders table labelled as a test of the virial theorem itself.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| PRL Version Thermodynamic Identity Governing Virial Theorem | March 18, 2026 | 3 | core | 6 |
| Virial Partitian Across Wide Domains | February 25, 2026 | 13 | core (37-orders table) | 25 |
| Virial Efficiency and Effective Nonlinear Exponent | February 25, 2026 | 7 | supporting | 8 |

**Recompute before printing:** every row of the 37-orders table from its cited measurement.

### 1.4 The Mahaffey number and the IAM floor

M = E_drive/k_BT; the IAM floor; the informational fidelity ratio A; one gauge in every domain. Sets up Parts 2–4.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM Law | March 2026 | 29 | core (§14.7, corrected) | 42 |
| The Informational Actualization Model   A Technical Reference for Physicists | September 2026 | 28 | core | 14 |

**Recompute before printing:** all constants from CANON/iam_canon.json only.

### 1.5 How the work was tested

Record of what was tested, when, and what failed; the scorecard as history, not evidence.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM Official Score Card | March 5, 2026 | 8 | record | 6 |
| IAM Test Validation Compendium | February 2026 | 22 | record | 9 |
| Supplementary Methods Reproducibility Guide | February 2026 | 24 | methods (reproduction steps) | 21 |

**Recompute before printing:** every score-card entry re-run from tests/ before it is quoted.

## Part 2 — Informational Actualization of GR: The Cosmological Dynamic

### 2.1 From the virial theorem to β_m = Ω_m/2 and E(a)
**Drafted (2026-10-02):** `docs/book/part2_drafts/p2_virial.tex` — coupling, E(a) and the exponent n = 7/2, µ(z), σ8/S8, two H0, Ω_m growth vs geometry, E_G, R(a), tests. DM/DE as halves, coincidence reading and arrow of time → Part 5 (held for discussion).

The single derived coupling and the activation function; the derivation chain written once.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM Theory Paper | April 14, 2026 | 34 | core (newest derivation, 14 Apr) | 38 |
| Dark Matter and Dark Energy as Virial Partners | March 2026 | 12 | core | 14 |
| Virial Efficiency and Effective Nonlinear Exponent | February 25, 2026 | 7 | supporting (n_eff) | 8 |

**Recompute before printing:** β_m: canon 0.1575 (Ω_m 0.315); Dual-Sector CAMB paper prints 0.15765 (Ω_m 0.3153) — state one value and its Ω_m.

### 2.2 The dual-sector model

μ < 1 for matter, Σ = 1 for light; why the data demanded two sectors.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM M Sigma Paper | February 25, 2026 | 7 | core | 12 |
| Late Time Growth Suppression in the mu Sigma Framework  Confrontation with Planck and Large Scale Structure | February 19, 2026 | 13 | core | 13 |
| IAM Dual Sector Note | March 2026 | 9 | supporting (informal reasoning) | 14 |
| Dual Sector Validation Paper | February 23, 2026 | 12 | core (critical test) | 13 |

**Recompute before printing:** μ₀ = −0.135 mapping from β_m.

### 2.3 Boltzmann solver and the Planck likelihood

MGCAMB and dual-sector CAMB implementations; MCMC against Planck 2018.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM CAMB Technical Note | February 19, 2026 | 28 | core | 46 |
| Dual Sector Perturbation Cosmology CAMB | February 28, 2026 | 17 | core | 15 |

**Recompute before printing:** '17 converged chains' and the χ²/σ improvement re-run from the repo's chains (R−1, priors, data versions stated).

### 2.4 Hubble tension, growth and S8

Photon-sector vs matter-sector H0; the S8 redshift trend; DESI dynamical dark energy as a sector artifact.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| The Redshift Dependent S 8 Trend in the Context of IAM | March 2026 | 8 | core | 16 |
| Dark Energy or Sector Tension | March 2026 | 20 | core | 28 |

**Recompute before printing:** '5.5σ improvement over ΛCDM' recomputed with current data releases (DESI DR2 if used).

### 2.5 Lensing and cluster masses

Σ = 1 means lensing is unchanged; three-way mass comparison in merging clusters.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM Lensing Dynamics Paper | February 25, 2026 | 9 | core | 17 |
| 3Way Mass Discrepancy in Galaxy Clusters | February 25, 2026 | 8 | core | 10 |

**Recompute before printing:** cluster list and mass sources.

### 2.6 Small scales: missing satellites and cusp–core

Both derived scalings and the shared ~10² missing prefactor — the main open problem in the gravity sector.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| Missing Satellites | March 2026 | 10 | core | 17 |

**Recompute before printing:** locate the cusp–core derivation (not a separate repo paper); state the prefactor as open.

### 2.7 Forecasts for coming surveys

Euclid Fisher forecast, ISW, binned μ reconstruction — dated predictions that can fail.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM Survey Predictions Paper | February 25, 2026 | 11 | core | 4 |

**Recompute before printing:** forecast figures in tests/ regenerated.

### 2.8 Vacuum energy and the far future

The cosmological-constant discrepancy; w(z) and the far future.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| The Cosmological Constant as Actualized Vacuum Energy | March 2026 | 18 | core (retitle in physics wording) | 11 |
| wz far future | February 2026 | 19 | core | 16 |

**Recompute before printing:** 10¹²² comparison and the w(z) curve.

### 2.9 Baryon asymmetry

η ≈ 6.1×10⁻¹⁰ as a derived quantity; CMB evidence without the BBN prior.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| Matter Antimatter Asymmetry and the Information Writing Constraint | March 2026 | 12 | core | 8 |
| Baryon Asymmetry as a Derived Quantity CMB Evidence Without BBN Prior | March 2026 | 6 | core | 4 |

**Recompute before printing:** η derivation; the 18th-chain run (18thChainBaryonAsymmetry.rtf) re-read as the record.

### 2.10 Horizons and black holes

Horizon thermodynamics with decoherence entropy; the information paradox reframed.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM BH Thermodynamics | February 25, 2026 | 9 | core | 9 |
| IAM Black Hole Information Paradox | February 2026 | 14 | interpretation (label) | 4 |

**Recompute before printing:** P_SB = P_Hawking identity.

### 2.11 Decoherence: from horizon to laboratory

Gravitational decoherence in open-system language; quantum Darwinism at cosmological scale; measurement and non-locality; the direction of time.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| Gravitational Decoherence Quantum Level | February 2026 | 22 | core | 15 |
| Quantum Darwinism at Cosmological Scales | March 2026 | 15 | core | 18 |
| IAM Measurement Problem Quantum | February 2026 | 21 | interpretation (label) | 15 |
| Non Locality and the Boundary of Reality | March 2026 | 10 | interpretation (label) | 9 |
| The Boundary Between Potential and Actual | March 2026 | 12 | interpretation (label; retitle) | 12 |
| The Two Faces of Time | March 2026 | 7 | interpretation (label) | 3 |

**Recompute before printing:** decoherence rates; every 'interpretation' claim kept separate from derivations.

### 2.12 Particle masses

Electron rest mass from boundary encoding; the Koide relation; the Higgs and the weak force.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| Electron Rest Mass from  IAM | February 2026 | 10 | core (assumptions stated) | 7 |
| Koide Mahaffey | April 22, 2026 | 7 | core | 6 |
| The Higgs Boson and the Origin of Duration | March 2026 | 15 | interpretation (label; retitle) | 10 |

**Recompute before printing:** numerical agreement values and their precision.

### 2.13 Bridge to the qubit

The transmon quasiparticle density x_qp ~ 10⁻⁷ — the cosmology-to-device link that opens Part 3.

| source paper | dated | pages | role | stated predictions |
|---|---|---|---|---|
| IAM Xqp Mahaffey | April 14, 2026 | 8 | core (opens Part 3) | 13 |

**Recompute before printing:** x_qp prediction vs published measurements.

## Not in Parts 1–2

| paper | placement |
|---|---|
| Physics of Methylation — Landauer Metrology | Part 4 |
| IAMPerformance the quantum-processor report Issue 002, the semiconductor report Issue 002 | Part 3 |
| Quantum Platform Demo.html, Semiconductor Platform Demo.html | Part 3 (companion material) |
| Gravitational Propulsion and IAM; IAM Gravitational Engineering Exploration | **Out of the book.** Both call themselves exploratory notes; same header (likely two versions). If kept, an 'Open directions' appendix marked speculative. |
| one excluded archive paper (author ruling) | its virial-equilibrium data go into 1.3 as 'the virial theorem across scales'; the paper is never cited |


## Additional sources from the LaTeX archive (Archive.zip, 2026-10-01)

These are not in `docs/papers/` (except where noted). LaTeX sources are used for equations and figures; every number is still re-run first.

| chapter | source (archive file) | role |
|---|---|---|
| 1.2 | Variational_Derivation_of_IAM.tex — Jacobson–Cai-Kim derivation of the dual-sector field equations | core (not in repo) |
| 1.2 | IAM_Saridakis_Bridge.tex | LaTeX of the repo's 'A Note on Entropic Gravity' |
| 1.3 | iam_thermo_prl.tex, iam_thermodynamic_identity.tex, IAM_Virial_37_Orders_Paper.tex + generate_figures.py, Virial tests/*.py | LaTeX + figure code for the repo papers |
| 2.1 | Holographic_Derivation_of_IAM.tex — E(a) from horizon thermodynamics | core (not in repo) |
| 2.1 | iam_decoherence_bridge_paper.tex — decoherence to the dual-sector coupling | core (not in repo) |
| 2.1 | Master_Master_Paper.pdf (33 pp, 'A Zero-Parameter Derivation of μ < 1, Σ = 1 with Full Planck 2018 Validation') | core (not in repo); its 24-pp arXiv version is the repo's IAM_Master_Preprint |
| 2.4 | IAM Sector Phantom: iam_sector_phantom.py + figure | figure code for the DESI chapter |
| 2.7 | Dated predictions with no paper (script + figure each, in `tests/` and `figures/`): cosmic age excess (1.378 Gyr at z = 3.5), CMB hemispherical asymmetry (A = 0.0628), cusp–core σ² scaling, small-scale structure (13.6 % suppression). Forecasts in `mgcamb_validation/forecasts/`: Fisher (Euclid/DESI σ(μ0)), ISW (A_ISW = 1.134), binned μ(z), transition zone; `tests/forecast_detection_threshold.py` | **lead the predictions chapter**; all re-run 2026-10-01 |
| 2.10 | IAM_BH_Cosmology_Paper_B.tex ('The Cessation of Projection') | LaTeX of the repo's IAM_Black_Hole_Information_Paradox |
| 2.10 | work in progress/IAM_Black_Holes_Working_Document.md | working draft (not in repo) |
| 2.11 | GRF_Essay_Final.tex ('Gravitational Decoherence on Timelike Worldlines') | core (not in repo) |
| 2.11 | Gravitational_Decoherence.tex (E_q(t)) | LaTeX of the repo's Gravitational_Decoherence_Quantum_Level |
| 2.11 | Paper9_Measurement_Problem.tex, zurek_paper.tex, iam_smolin_paper.tex | LaTeX of the repo's measurement-problem, quantum-Darwinism and boundary papers; interpretation (label) |
| 2.12 | Koide_Paper_revised.tex, electron_mass_referee_safe.tex | LaTeX of the repo papers |
| Part 3 | Quantum Decoherence_Quantum Computing/Quantum computing forecasts (phonon heating, detection threshold, timeline) | Part 3 forecasts |
| Part 4 | floor_breach_derivation.pdf (same file as the 30 Sept upload) | Part 4 |

**Excluded from the book:** several archive files by the author's ruling (non-physics, confidential or retired); the list is kept outside the repository.

| file | reason |
|---|---|
| IAM_Gravitational_Engineering_Exploration.tex | exploratory note (see above) |

## Decisions for the author

1. **Vocabulary.** Several titles use *potential / actual / actualized / duration*. These are also terms of classical philosophy, the risk you want to avoid. Proposed book wording: superposition → classical record (decoherence); 'Actualized vacuum energy' → 'Vacuum energy after decoherence'; 'Origin of Duration' → 'The weak force and irreversibility'. The model name itself stays and is defined by its physics.
2. **Interpretation chapters.** Measurement problem, non-locality, the two faces of time, the information paradox: keep in 2.10–2.11 with an 'interpretation' label, or move to Part 5. They carry few testable predictions.
3. **Duplicates and versions.** Two Bekenstein-coefficient PDFs; two exploratory gravity notes. I diff them before drafting.
5. **Archive papers into the repo.** Five physics papers exist only in the archive: Variational_Derivation_of_IAM, Holographic_Derivation_of_IAM, iam_decoherence_bridge_paper, GRF_Essay_Final and Master_Master_Paper (33 pp), plus one working draft (IAM_Black_Holes_Working_Document.md). Push these, together with the LaTeX sources of the repo papers and their figure code, into `docs/papers/latex/` so every chapter source is in the repo? Excluded files would not be pushed.
4. **Re-verification.** Every headline cosmology number (5.5σ, 17 chains, H0 pair, μ₀, η, x_qp) is re-run from the repo's own code before it is printed, as for the cell work.

## Book writing rule (author, 2026-10-01)
Every chapter states only the current, correct value or definition. No 'as printed', 'corrected', 'superseded' or 'earlier versions'
wording appears anywhere in the book; correction history stays in the development records and git.
