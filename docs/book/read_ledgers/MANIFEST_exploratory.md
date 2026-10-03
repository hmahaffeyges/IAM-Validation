# MANIFEST: Part 5, Chapter "Exploratory: what the law permits for gravitational engineering"

Clone: `e37aabb` (sparse: docs/book, docs/verification, docs/papers/IAM_Gravitational_Engineering_Exploration.pdf, CANON). Nothing pushed.

## Files
| file | status |
|---|---|
| `docs/book/part5/p5_02_exploratory.tex` | rewritten (98 -> 361 lines; long-line source), in the author's own words from coverage/wave2/47_Gravitational_Engineering_Exploration.tex. main.tex placement unchanged: line 97 `\input{part5/p5_02_exploratory}` |
| `docs/book/figscripts/fig_p5_exploratory.py` | new, on `_bookstyle.py` |
| `docs/book/figures/part5/fig_exploratory_{transit,recession,twin,steering}.{pdf,png}` | new |
| `docs/verification/scripts/verify_exploratory.py` + `_output.txt` | new (sympy + numerics, every step) |
| `docs/book/bib_exploratory.bib` | no new entries; records the CrossRef check of the 12 iam.bib keys used |

## For the lead / author: decisions and follow-ups
1. **Scope conflict to confirm.** Memory holds an author exclusion of 2026-10-01 ("gravitational engineering and gravitational propulsion notes are NOT in the omnibus book"). The later ONE BOOK order (2026-10-02) lists "exploratory (propulsion, engineering)" as wanted, main.tex already inputs this chapter, and this task assigns it. I followed the later order and the task. If the 2026-10-01 exclusion still stands, drop line 97 of main.tex and the refs listed in item 4.
2. **Proposed new errata rows** (not in PAPER_ERRATA.md; please add):
   - **GE6** | Gravitational Engineering note | I.3, Eq. (8) | hover: grad Phi_IAM = -g_ambient | sign: with g = -grad Phi, cancelling the ambient field needs grad Phi_eng = +g_amb (the printed form doubles the field) | confirmed (verify_exploratory.py s.4) | p5_02 Eq. ge_hover
   - **GE7** | Gravitational Engineering note | II.4 | three non-coplanar sources are the minimum to steer one focal node through a 3D volume | for an isotropic kernel n sources are mirror-symmetric about any plane containing them: three sources steer a single node only in their plane (off-plane foci have an equal mirror twin); a volume needs >= 4 non-coplanar sources or an anisotropic kernel; also a static Laplace kernel has no isolated focus (maximum principle), so the drive must oscillate | confirmed (verify s.9; fig_exploratory_steering) | p5_02 sec:ge_steer
3. **Wording changes, not errata** (the 3 stand-alone and 11 rule FLAGs of the coverage file, all resolved): physics-terms rule applied ("actualization potential/field" -> effective potential Phi and the record-writing rate; "potential/actual boundary" -> boundary between coherent superposition and classical record; "not-yet-actualized" -> held in coherent superposition, not yet written as a classical record). Premise reframed to the book's frame (IAM leaves GR as it is) in place of "gravity not a fundamental force". D_H now given for H0 = 67.16 and 72.26 (paper used 70; kept as the round value).
4. **Other files that reference this chapter (not mine; owners should update):** `appendices/app_I_provenance.tex` rows 172-173 still point `fig:transit`/`fig:recession` to `fig_transit.pdf`/`fig_recession.pdf` from `fig_p5_extra.py`/`fig_p5.py` (the latter used H0 = 67.36). The labels are kept; the chapter now includes `fig_exploratory_transit.pdf`/`fig_exploratory_recession.pdf`. Old figures and scripts are untouched. New labels fig:twin_exploratory, fig:steering need provenance rows. `app_B_errata_physics.tex` rows 30-31 and `p5_11_status_all.tex` row 105 are consistent with the chapter.
5. GE5 (two repo PDFs = one paper) stands; only `IAM_Gravitational_Engineering_Exploration.pdf` is present at e37aabb.

6. **Own-words measure** (normalised sentence match, math stripped, same method for both versions): 45 of 124 source sentences of 6+ words are now verbatim in the chapter (first version: 12). The remainder are (a) sentences containing math, which are carried as numbered equations; (b) sentences flagged by the coverage file and reworded under the physics-terms and stand-alone rules (actualization/potential/actual, every first-person 'we'/'us' including the closing 'The engineering says we cannot yet show how to go' -> 'The engineering cannot yet show how to go', 'this note/document', author line); (c) sentences withdrawn or corrected by GE1-GE3 and proposed GE6-GE7 (tilt signature, E -> e ceiling, 'no new physics beyond its existence', hover sign, three-source minimum). The coverage header's 'Errata: (none listed)' is out of date: PAPER_ERRATA.md has GE1-GE5 for this paper.

## Read ledger
pypdfium2 text of `docs/papers/IAM_Gravitational_Engineering_Exploration.pdf`: 6 pages, **268 lines**, 2,580 words. PAPER_LINE_COUNTS.md row 47 says 274 (difference 6 lines, extraction differences; within a few lines). No LaTeX source exists under docs/papers/latex/.
Chunks read in full (text shown, none truncated): 1-50, 51-100, 101-150, 151-200, 201-240, 241-268. Ledger complete 1..268.
Coverage source `docs/book/coverage/wave2/47_Gravitational_Engineering_Exploration.tex`: 96 lines, read in full (1-50, 51-96); its 17 EQUATION comments retyped from the PDF; every FLAG resolved.

## Errata applied
GE1 (tilt signature), GE2 (E -> e speed limit), GE3 (displaced region needs negative energy), GE4 (state the equivalence-principle bound), GE5 (duplicate file). Proposed GE6, GE7 applied in the chapter.

## Carriage table (paper line | content | book p5_02_exploratory.tex line | verdict | status label)
| paper | content | book | verdict | label |
|---|---|---|---|---|
| 1-8 | title, author block, timestamp | 1 (chapter title) | CARRIED (stand-alone: no author/paper naming; title per FLAG) | - |
| 9-18 | status of document: two zones, Zone I derivable, Zone II conjectural, boundary never blurred, separation is the point | 3-10 | CARRIED (own words; 'note/document' -> 'chapter') | - |
| 19-21 | premise: decoherence on timelike worldlines paid at Landauer cost; 'gravity not a fundamental force' | 13-15 | CARRIED-CORRECTED (book frame: GR unchanged) | - |
| 22-25 | Eq. (1) Delta E = k_B T_H ln2, T_H = hbar H/2 pi k_B; Gibbons-Hawking | 16-22 (eq:ge_landauer) + today's values 20-22 | CARRIED (+ values at 67.16/72.26) | \calc |
| 26-29 | virial halves: geometric/potential half local, kinetic/informational half global (dark-energy term) | 24-26 | CARRIED | \interp |
| 29-30 | 1/2 fixed by equilibrium, beta_m = Omega_m/2 | 26-27 | CARRIED (0.15765) | \prediction |
| 30-36 | Eq. (2) E(a) = exp(1-1/a), E(1)=1, E(inf)=e | 27-31 (eq:ge_Ea) | CARRIED (+ rate peak a = 1) | \derived |
| 37-42 | Eq. (3) H^2_eff,m = H^2_LCDM + beta_m E H0^2 | 32-35 (eq:ge_Hm) | CARRIED | \derived |
| 44-46 | not new; tested elsewhere; single question | 36-37 | CARRIED (-> Part 2) | - |
| 47-50 | Zone I; I.1 gravity as a steerable effective potential | 39-43 | CARRIED | - |
| 51 | Eq. (4) g = -grad Phi_IAM | 44-46 (eq:ge_g) | CARRIED | \derived |
| 52-56 | definitional; Newton; local gradient only; only ingredient; engineering is Zone II | 47-50 | CARRIED ('We' -> impersonal) | \derived |
| 57-59 | I.2 geodesic; Eq. (5) | 52-57 (eq:ge_eom) | CARRIED | \derived |
| 60-64 | equivalence principle since 1907; Eq. (6) f_felt = 0 | 58-63 (eq:ge_felt) | CARRIED | \derived |
| 64-67 | uniform over L; Eq. (7) f_tidal ~ m L d2Phi; vanishes | 64-72 (eq:ge_tidal) | CARRIED (+ 2GML/r^3; 3.1e-6 g, 3.1e-5 g; fig:transit b) | \derived, \calc |
| 68-72 | Consequence (rigorous): no felt acceleration, 90-deg turn, dissolves inertial objection, no new physics | 74-78 | CARRIED | \derived (conditional) |
| 73-75 | I.3 hover; Eq. (8) grad Phi_IAM = -g_ambient | 80-86 (eq:ge_hover) | CARRIED-CORRECTED (proposed GE6, sign) | \derived |
| 76-79 | local null; departure; same operation; Newton + EP | 86-90 | CARRIED | \derived (conditional) |
| 81-83 | I.4 rigid body suspended orients like a plumb bob | 92-95 | CARRIED (own words) | \derived |
| 84-90 | Eq. (9) theta_tilt = arctan(|grad|_h/|grad|_v); monotonic in acceleration | 95-100 (eq:ge_plumb) | CARRIED-CORRECTED (GE1: hang angle of a suspended body only) | \derived |
| 91-96 | falsifiable tilt signature; no banking; discriminator vs lift-based aircraft | 101-114 (zero torque; eq:ge_ggtorque) | EXCLUDED as a signature (GE1: contradicts I.2); replaced by the correct orientation result | \derived |
| 97-107 | I.5 Eq. (10) v_rec = H0 D; Eq. (11) D_H = c/H0 ~ 1.4e10 ly at 70 | 116-127 (eq:ge_vrec, eq:ge_DH); fig:recession 135-139 | CARRIED-CORRECTED (1.46/1.35e10 ly at 67.16/72.26; 70 kept) | \calc |
| 108-112 | nothing moves faster than c; space carries them; standard; global half; mechanism; 'speed limit' not c | 127-131 | CARRIED | \interp |
| 112-113 | set by actualization imbalance, bounded above by E -> e | 131-133 | CARRIED-CORRECTED (GE2: open, not derived) | \openprob |
| 114-119 | I.6 locally inertial bubble; Eq. (12) d tau_ship = dt_bubble | 142-157 (eq:ge_alc added, eq:ge_tau) | CARRIED (+ Alcubierre metric, centre worldline) | \derived |
| 120-134 | no SR dilation; Eq. (13) ~ Delta Phi/c^2 << 1 | 157-165 (eq:ge_dtau) | CARRIED (+ weak-field clock rate; 'second-order' dropped: the term is first order in Phi/c^2) | \derived |
| 135-140 | Consequence: no twin penalty; Alcubierre-class; 'no new physics beyond existence' | 167-171; obstacles 173-184 (eq:ge_rho) | CARRIED-CORRECTED (GE3: WEC, Pfenning-Ford, Everett-Roman) | \derived (conditional); \derived (published) |
| - | twin comparison figure | fig:twin_exploratory 186-189 | NEW | \calc |
| 141-153 | I.7 Pleiades 136 pc = 444 ly; xi; Eq. (14) | 191-199 (eq:ge_tone) | CARRIED | \calc |
| 154-160 | 7 days = 0.0192 yr; Eq. (15) xi ~ 2.3e4 | 200-205 (eq:ge_xi); fig:transit a 212-216 | CARRIED (2.31e4 recomputed) | \calc |
| 161-166 | 2e4 c-equivalent; not forbidden; 'far below the E -> e ceiling'; negligible fraction of D_H; not the obstacle; Zone II | 206-210 | CARRIED-CORRECTED (GE2 ceiling clause removed; 3e-8; GE3 obstacle) | \calc |
| 167-177 | transition to Zone II | 219-225 | CARRIED (own words; 'We' -> impersonal; 'actualization field' -> source of Phi_IAM) | - |
| 179-195 | II.1 Conjecture C1; Eq. (16) | 227-235 (eq:ge_kernel) | CARRIED | \conjecture |
| 196-198 | G unknown; sourced by rate; driven not observed; central open problem | 236-239 | CARRIED ('actualization' -> record writing) | \conjecture, \openprob |
| 199-204 | II.2 Premise P1 knife-edge; Z ~ 114-126 | 241-244 | CARRIED (physics terms) | \conjecture |
| 205-207 | Premise P2 coherence | 246-248 | CARRIED | \conjecture |
| 208-213 | Tension; magic numbers; unproven; no derivation past this point | 250-255 (+ Oganessian2015) | CARRIED | \conjecture, \observed |
| 214-220 | II.3 conditional prediction: weighs less; weigh it | 258-262 | CARRIED ('not-yet-actualized' -> not yet written as a classical record) | \prediction (conditional) |
| - | equivalence-principle baseline (MICROSCOPE, LLR, Podkletnov + replications) | 264-271 | CARRIED-CORRECTED (GE4) | \observed |
| 221-232 | II.4 steering; Eq. (17) | 273-280 (eq:ge_drive) | CARRIED (sum over i) | \derived (conditional on G) |
| 234-236 | three is the minimum; two axial; monopole none | 281-290; fig:steering 297-304 | CARRIED-CORRECTED (proposed GE7; Laplace no-focus added) | \derived (conditional) |
| 236-240 | P2 gives stable phase; three elements jointly necessary | 291-295 | CARRIED ('three sources' -> 'sources') | \conjecture |
| 241-248 | II.5 what would close the gap; honest status | 306-313 | CARRIED-CORRECTED (GE3 negative energy added) | \openprob, \interp |
| 249-257 | Summary Zone I | 315-321 | CARRIED-CORRECTED (GE1 tilt removed; GE3) | - |
| 258-262 | Summary Zone II | 323-326 | CARRIED (+ 1e-15 baseline) | - |
| 263-267 | closing; Zone I derivable / Zone II conjectural | 328-332 | CARRIED-CORRECTED (GR negative-energy sentence added; 'we'/'us' removed per stand-alone FLAG) | \interp |
| 267-268 | 'Timestamped in repository'; page number | - | EXCLUDED (stand-alone; no repository narrative) | - |
| - | status table | 333-361 (tab:ge_status) | NEW | - |

Equations carried: 17 of 17 (Eq. 8 sign-corrected; Eq. 9 restricted to a suspended body); chapter has 20 numbered equations. Added derivations: tidal 2GML/r^3, plumb-line angle, zero torque in a uniform field, gravity-gradient torque, Alcubierre metric, centre proper time, weak-field clock, Eulerian energy density, Laplace no-focus, mirror symmetry.
Exclusions: tilt-toward-acceleration and no-banking as a falsifiable signature (GE1); 'bounded above by E -> e' and 'far below the E -> e ceiling' (GE2); 'requires no new physics beyond the existence of such a region' (GE3); 'Timestamped in repository' footer.

## Verification (docs/verification/scripts/verify_exploratory_output.txt)
T_H 2.6459e-30 K / 2.8468e-30 K; kT ln2 2.5321e-53 / 2.7244e-53 J; E(1)=1, E->e, rate peak a=1; f_felt = 0; tidal 2GML/r^3, 3.14e-6 g (10 m), 3.14e-5 g (100 m); hover grad Phi_eng = +g_amb; dumbbell torque = -(3GM/2r^3)(I_perp-I_par) sin2theta (residual 0); D_H 1.4559e10 (67.16), 1.3968e10 (70), 1.3532e10 (72.26) ly; Pleiades 443.6 ly = 3.05e-8 / 3.28e-8 of D_H; centre worldline ds^2 = -c^2 dt^2; Eulerian rho minus Alcubierre's form = 0; xi = 2.3145e4 (2.3125e4 with the rounded 444/0.0192); SR v = 0.99c: 448.1 / 63.2 yr; Laplacian(1/r) = 0; 3 coplanar mirror/target 1.000, 4 non-coplanar 0.448; MICROSCOPE combined 2.75e-15.

## Static checks
Braces balanced; environments matched; 36 labels unique across docs/book (RETIRED excluded); every \ref/\eqref resolves; 12 \cite keys all in iam.bib and CrossRef-checked; 4 figures exist; floats [htbp]; no paper/author/the quantum-processor report/the semiconductor report/the methylation report/potential-actual wording. Not compiled (no TeX in this sandbox).
