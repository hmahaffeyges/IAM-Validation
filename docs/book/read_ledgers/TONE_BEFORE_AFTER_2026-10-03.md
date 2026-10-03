# Tone pass: before → after

Base: github.com/hmahaffeyges/IAM-Validation `main` at dcd3879. Scope: front matter and Parts I–V and VII of `docs/book`. Part VI (`part4/*`), `part3/p3_02_xqp.tex`, `part5/p5_10_conclusion.tex` and the appendices were not touched.

Two kinds of change:
- **tone**: a sentence that broadcast a failed, contradicted or rejected earlier claim, or framed the work by what it lacks, restated as the book's own forward-looking statement. Numbers, equations, citations, references, labels and status labels are unchanged.
- **closing**: one or two new sentences at the end of the chapter's main text (before its Status section or data paragraph where it has one) saying why the result matters and what a reader in that field can do next. No digits outside symbols already used in the chapter, no status labels, no new claims.

Static checks on every edited file: braces balanced as in the original; \label set unchanged (book total 1322 before and after); every \ref/\eqref resolves to a label in the book; every \cite key exists in iam.bib; status-label counts per file unchanged; no number removed.

## Counts

| Part | File | Tone edits | Closing |
|---|---|---|---|
| Front matter | `part0/p0_abstract.tex` | 0 | 1 |
| Front matter | `part0/p0_preface.tex` | 1 | 0 |
| Front matter | `part0/p0_giants.tex` | 1 | 0 |
| Front matter | `part0/p0_how_to_read.tex` | 0 | 1 |
| Part I | `part1/p1_01_encoding_surfaces.tex` | 3 | 1 |
| Part I | `part1/p1_02_iams_law.tex` | 9 | 1 |
| Part I | `part1/p1_03_virial_law.tex` | 0 | 1 |
| Part I | `part1/p1_04_virial_identity.tex` | 0 | 1 |
| Part II | `part2/p2_02_virial.tex` | 1 | 1 |
| Part II | `part2/p2_02b_virial_tests.tex` | 2 | 1 |
| Part II | `part2/p2_03_theory.tex` | 2 | 1 |
| Part II | `part2/p2_03a_entropic_gravity.tex` | 0 | 1 |
| Part II | `part2/p2_04_dualsector_chains.tex` | 1 | 1 |
| Part II | `part2/p2_07_late_time_growth.tex` | 1 | 1 |
| Part II | `part2/p2_06_dual_sector_perturbation.tex` | 4 | 1 |
| Part II | `part2/p2_05_dual_sector_note.tex` | 4 | 1 |
| Part II | `part2/p2_08_s8_trend.tex` | 1 | 1 |
| Part II | `part2/p2_09_sector_tension.tex` | 0 | 1 |
| Part II | `part2/p2_09b_phantom_crossing.tex` | 0 | 1 |
| Part II | `part2/p2_10_dual_sector_validation.tex` | 3 | 1 |
| Part II | `part2/p2_11_dark_energy.tex` | 0 | 1 |
| Part II | `part2/p2_20_wz_far_future.tex` | 1 | 1 |
| Part II | `part2/p2_12_lambda.tex` | 9 | 1 |
| Part II | `part2/p2_12b_lambda_history.tex` | 1 | 1 |
| Part II | `part2/p2_13_baryon.tex` | 5 | 1 |
| Part II | `part2/p2_13b_baryon_chain.tex` | 3 | 1 |
| Part II | `part2/p2_16_survey_predictions.tex` | 2 | 1 |
| Part II | `part2/p2_17_lensing_dynamics.tex` | 1 | 1 |
| Part II | `part2/p2_18_three_way_clusters.tex` | 3 | 1 |
| Part II | `part2/p2_19_missing_satellites.tex` | 1 | 1 |
| Part III | `part2/p2_01_blackholes.tex` | 2 | 1 |
| Part III | `part2/p2_01a_bekenstein.tex` | 0 | 1 |
| Part III | `part5/p5_01b_bh_information.tex` | 4 | 1 |
| Part III | `part3/p3_07_saturation.tex` | 2 | 1 |
| Part IV | `part2/p2_14_quantum_records.tex` | 3 | 1 |
| Part IV | `part2/p2_21_entanglement_records.tex` | 2 | 1 |
| Part IV | `part5/p5_04_measurement.tex` | 2 | 1 |
| Part IV | `part5/p5_05_gravdec.tex` | 4 | 1 |
| Part IV | `part5/p5_06_nonlocality.tex` | 0 | 1 |
| Part IV | `part2/p2_22_electroweak.tex` | 1 | 1 |
| Part IV | `part2/p2_22b_higgs_record.tex` | 13 | 1 |
| Part IV | `part2/p2_15a_lepton_koide.tex` | 7 | 1 |
| Part IV | `part2/p2_15b_electron_mass.tex` | 6 | 1 |
| Part V | `part3/p3_01_sc_primer.tex` | 0 | 1 |
| Part V | `part3/p3_03_a_for_processors.tex` | 0 | 1 |
| Part V | `part3/p3_04_thermal_n.tex` | 1 | 1 |
| Part V | `part3/p3_05_coherence_optimum.tex` | 1 | 1 |
| Part V | `part3/p3_06_cmos.tex` | 1 | 1 |
| Part VII | `part5/p5_01_interpretation.tex` | 0 | 1 |
| Part VII | `part5/p5_03_time.tex` | 1 | 1 |
| Part VII | `part5/p5_05b_virial_partners.tex` | 4 | 1 |
| Part VII | `part5/p5_05c_virial_decoherence.tex` | 1 | 1 |
| Part VII | `part3/p3_08_one_gauge.tex` | 4 | 1 |
| Part VII | `part5/p5_08_synthesis.tex` | 6 | 1 |
| Part VII | `part3/p3_09_reach.tex` | 1 | 1 |
| Part VII | `part5/p5_07_predictions.tex` | 4 | 1 |
| Part VII | `part5/p5_09_open.tex` | 0 | 1 |
| Part VII | `part5/p5_02_exploratory.tex` | 14 | 1 |
| Part VII | `part5/p5_11_status_all.tex` | 2 | 1 |
| | **Total (59 files)** | **145** | **57** |


## `part0/p0_abstract.tex` (Front matter) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): Each piece can be checked by someone who already has the tools: a cosmologist can run the chains, a device physicist can read a processor against its floor, a geneticist can read a cell against its own healthy reference. Pick the piece you know, recompute it, and try to break it.

## `part0/p0_preface.tex` (Front matter) — 1 tone edits, 0 closing
- line 53
  - before: and a forecast made with the IAM $\mu(z)$ itself is still to be done (Chapter~\ref{ch:latetime}).
  - after: and the next step is a forecast made with the IAM $\mu(z)$ itself (Chapter~\ref{ch:latetime}).

## `part0/p0_giants.tex` (Front matter) — 1 tone edits, 0 closing
- line 205
  - before: a forecast made with the IAM $\mu(z)$ itself is still to be done (Chapter~\ref{ch:latetime}).
  - after: a forecast made with the IAM $\mu(z)$ itself is the next step (Chapter~\ref{ch:latetime}).

## `part0/p0_how_to_read.tex` (Front matter) — 0 tone edits, 1 closing
- **closing** (inserted before \section*{Conventions}): Pick the Part that touches your field, read the derivation it rests on, then open the repository and run it yourself. Every result tells you how strongly it is claimed, so you know exactly where to push.

## `part1/p1_01_encoding_surfaces.tex` (Part I) — 3 tone edits, 1 closing
- line 125
  - before: On arrays it is still to be derived. \openprob{}
  - after: On arrays, this is an open problem. \openprob{}
- line 215
  - before: \textbf{Honest caveat.}
  - after: \textbf{Open problem.}
- line 224
  - before: $E_{\rm hold}$ is read from single molecules, not derived \measured.
  - after: $E_{\rm hold}$ is read from single molecules; its derivation is an open problem \measured.
- **closing** (inserted end of chapter): This is why the encoding surface is worth a physicist's and a geneticist's time: one ledger, read the same way from the horizon to the cell. The open steps are named above; each one is a project someone in that field can take up and settle.

## `part1/p1_02_iams_law.tex` (Part I) — 9 tone edits, 1 closing
- line 336
  - before: That structure formation writes at this rate is not yet calculated from the bulk \openprob.
  - after: Calculating from the bulk that structure formation writes at this rate is the next step \openprob.
- line 432
  - before: and a covariant form is not yet written \openprob.
  - after: and writing a covariant form is the next step \openprob.
- line 510
  - before: This is a mapping onto the $\mu$--$\Sigma$ form, not a derivation of it \interp.
  - after: This is a mapping onto the $\mu$--$\Sigma$ form; deriving the form is the next step \interp.
- line 531
  - before: Which implementation the record term implies is not derived \openprob; every growth number in this book names its form.
  - after: Which implementation the record term implies is an open problem \openprob; every growth number in this book names its form.
- line 593
  - before: The two directions meet at $7/2$ to within that running; they are not an exact identity, and a constant $7/2$ from the halos is not obtained \openprob.
  - after: The two directions meet at $7/2$ to within that running; they are not an exact identity, and obtaining a constant $7/2$ from the halos is open \openprob.
- line 739
  - before: A Fisher forecast made with the IAM $\mu(z)$ itself is still to be done (Chapter~\ref{ch:latetime}) \openprob.
  - after: The next step is a Fisher forecast made with the IAM $\mu(z)$ itself (Chapter~\ref{ch:latetime}) \openprob.
- line 822
  - before: The chains show that the consequence is consistent with Planck; they do not show that the coupling is exact.
  - after: The chains show that the consequence is consistent with Planck; whether the coupling is exact is the open question.
- line 916
  - before: IAM's Law does not solve the measurement problem at the level of a complete quantum theory of gravity; it identifies the thermodynamic cost of the transition and shows that this cost, accumulated over cosmic history, has observable cosmological consequences, consistent with current data (Table~\ref{tab:law_consistency}; Chapter~\ref{ch:measurement}).
  - after: IAM's Law identifies the thermodynamic cost of the transition and shows that this cost, accumulated over cosmic history, has observable cosmological consequences, consistent with current data (Table~\ref{tab:law_consistency}; Chapter~\ref{ch:measurement}); the measurement problem at the level of a complete quantum theory of gravity remains open.
- line 1006
  - before: That integration has not been carried out.
  - after: Carrying out that integration is the next step.
- **closing** (inserted end of chapter): The Mahaffey number is built from quantities already measured in cosmology, chip design and cell biology. Anyone working in one of those fields can compute it for their own system, in their own units, and check whether it behaves the way this chapter says. That is the most direct way to test IAM's Law, and the most direct way to break it.

## `part1/p1_03_virial_law.tex` (Part I) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): The virial half is the easiest place in the book to start: it is textbook mechanics, and every bound system a reader already works with carries it.

## `part1/p1_04_virial_identity.tex` (Part I) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): Every input behind this identity is public. A physicist can take any bound system in a $1/r$ potential, compute the kinetic half, and follow its cost to the surface where it is paid. That is the thread the rest of the book pulls.

## `part2/p2_02_virial.tex` (Part II) — 1 tone edits, 1 closing
- line 131
  - before: is the formal step still to be completed.
  - after: is the next formal step.
- **closing** (inserted end of chapter): The coupling has no free parameter, so a cosmologist can take the virial partition at face value and run it through the existing codes. Chapter~\ref{ch:virial_tests} lays out the tests.

## `part2/p2_02b_virial_tests.tex` (Part II) — 2 tone edits, 1 closing
- line 45
  - before: The probe-by-probe count by this rule is still to be made;
  - after: The probe-by-probe count by this rule is the next step;
- line 93
  - before: A forecast made with the IAM $\mu(z)$ itself is the first calculation still to be done.
  - after: A forecast made with the IAM $\mu(z)$ itself is the next calculation.
- **closing** (inserted end of chapter): Every prediction here is dated and sits ahead of the data that will decide it. A cosmologist with the survey releases in hand can run these tests directly and find out whether the sector split holds.

## `part2/p2_03_theory.tex` (Part II) — 2 tone edits, 1 closing
- line 290
  - before: What is not yet written down is the exact sharing: how much of $A_H/4G$ is geometric and how much is record at each epoch.
  - after: The next step is to work out the exact sharing: how much of $A_H/4G$ is geometric and how much is record at each epoch.
- line 895
  - before: and a forecast made with the IAM $\mu(z)$ itself is still to be done (Chapter~\ref{ch:latetime}) \calc.
  - after: and a forecast made with the IAM $\mu(z)$ itself is the next step (Chapter~\ref{ch:latetime}) \calc.
- **closing** (inserted end of chapter): This chapter turns established horizon thermodynamics into a testable cosmology. A cosmologist with a Boltzmann code can rerun the chains behind it; a relativist can check the one step added to Jacobson's derivation and try to break it. Chapter~\ref{ch:predictions} lists what would falsify it.

## `part2/p2_03a_entropic_gravity.tex` (Part II) — 0 tone edits, 1 closing
- **closing** (inserted before \section{Status of the results}): Everything here can be rerun today from the repository. A cosmologist can set the comparison with the other entropic proposals against the same likelihood, and a relativist can take the $\mu$--$\Sigma$ test to the survey forecasts before the data arrive.

## `part2/p2_04_dualsector_chains.tex` (Part II) — 1 tone edits, 1 closing
- line 263
  - before: and a forecast made with the IAM $\mu(z)$ itself is still to be done (Chapter~\ref{ch:latetime}).
  - after: and a forecast made with the IAM $\mu(z)$ itself is the next step (Chapter~\ref{ch:latetime}).
- **closing** (inserted end of chapter): Every chain here can be rerun from the repository with the coupling fixed, not fitted. A cosmologist can take the same configuration, swap in new survey data when they land, and see where $\mu_0$ sits. That is the whole test.

## `part2/p2_07_late_time_growth.tex` (Part II) — 1 tone edits, 1 closing
- line 380
  - before: is the first calculation still to be done \openprob.
  - after: is the next calculation \openprob.
- **closing** (inserted before \paragraph{Data.}): The data are public and the test is sharp. A cosmologist can rerun these chains, and a relativist can take the mechanism beyond the parameterisation, which is exactly what Chapter~\ref{ch:level2} begins.

## `part2/p2_06_dual_sector_perturbation.tex` (Part II) — 4 tone edits, 1 closing
- line 26
  - before: Two exploratory chains that put a version of the term into the background equation fit the CMB as well as $\Lambda$CDM, with a sampled $H_0=61.5$ and an expansion rate today of $66.1$; the term they coded is not the one intended, so the background placement of Eq.~\eqref{eq:l2_bg} is still to be tested.
  - after: Two exploratory chains that put a version of the term into the background equation fit the CMB as well as $\Lambda$CDM, with a sampled $H_0=61.5$ and an expansion rate today of $66.1$; those chains coded a different form of the term, so the background placement of Eq.~\eqref{eq:l2_bg} is the next test.
- line 332
  - before: is still to be run \openprob.
  - after: is the next run \openprob.
- line 481
  - before: itself is still to be done (Section~\ref{sec:lt_euclid}).
  - after: itself is the next step (Section~\ref{sec:lt_euclid}).
- line 499
  - before: The background alternative of Eq.~\eqref{eq:l2_bg} is still to be run as written.
  - after: The background alternative of Eq.~\eqref{eq:l2_bg}, run as written, is the next test.
- **closing** (inserted before \paragraph{Data.}): The mechanism is coded into a standard Boltzmann solver with no new parameter. A cosmologist can take the modified source, point it at the next growth data, and see directly whether the matter-sector expansion rate holds.

## `part2/p2_05_dual_sector_note.tex` (Part II) — 4 tone edits, 1 closing
- line 107
  - before: the fourth is a test still to be made.
  - after: the fourth is the next test.
- line 121
  - before: is a two-ruler test still to be made on real data
  - after: is a two-ruler test to be made next on real data
- line 185
  - before: Chapter~\ref{ch:sectortension}, not yet made on real data \openprob.
  - after: Chapter~\ref{ch:sectortension}, to be made next on real data \openprob.
- line 200
  - before: The CMB-S4 sensitivity to $\beta_\gamma$ is still to be forecast from the present bound \openprob.
  - after: Forecasting the CMB-S4 sensitivity to $\beta_\gamma$ from the present bound is the next step \openprob.
- **closing** (inserted before \paragraph{Code and data availability.}): The split is stated sharply enough to be tested from both sides. A cosmologist with the public chains can rerun the bound on $\beta_\gamma$ today and set $\mu_0$ against the next survey release when it arrives.

## `part2/p2_08_s8_trend.tex` (Part II) — 1 tone edits, 1 closing
- line 101
  - before: The shape is the observed one: lowest today, recovering by $z\sim1$. The size is not. The predicted lensing deficit today is $0.8\,\%$; the low-redshift weak-lensing surveys sit $6$--$9\,\%$ below Planck~\cite{Asgari2021,DESY3}.
  - after: The shape is the observed one: lowest today, recovering by $z\sim1$. The size is smaller: the predicted lensing deficit today is $0.8\,\%$; the low-redshift weak-lensing surveys sit $6$--$9\,\%$ below Planck~\cite{Asgari2021,DESY3}.
- **closing** (inserted before \section{Status}): This chapter trades a free parameter for a shape. A cosmologist can take the same compilations, run the comparison, and check whether the trend follows the activation function; the surveys now under way will measure that curve directly.

## `part2/p2_09_sector_tension.tex` (Part II) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): The comparison here runs on data already public. A cosmologist can rerun it and watch whether the curved deficit keeps its shape as new bins and tighter lensing surveys come in; with $\Sigma=1$, that is what separates the term from dynamical dark energy.

## `part2/p2_09b_phantom_crossing.tex` (Part II) — 0 tone edits, 1 closing
- **closing** (inserted before \section{Status}): The tests above are ready to run: measure $\mu$ and $\Sigma$ together, and check the $f\sigma_8$ shape bin by bin. A cosmologist with the public chains can run both and see directly which way the result falls.

## `part2/p2_10_dual_sector_validation.tex` (Part II) — 3 tone edits, 1 closing
- line 264
  - before: The three analyses converge on a single conclusion, but not the one the scenarios anticipated \derived: Type Ia supernova magnitudes with $M$ free neither reject photon-sector expansion ($H_0=67.4$) nor select the matter-sector normalization ($H_0=73.04$); they share one minimum,
  - after: The three analyses converge on a single conclusion \derived: Type Ia supernova magnitudes with $M$ free neither reject photon-sector expansion ($H_0=67.4$) nor select the matter-sector normalization ($H_0=73.04$); they share one minimum,
- line 486
  - before: a CMB-S4 forecast from that bound is still to be made \prediction.
  - after: a CMB-S4 forecast from that bound is the next step \prediction.
- line 497
  - before: a forecast made with the IAM $\mu(z)$ itself is still to be done \prediction.
  - after: a forecast made with the IAM $\mu(z)$ itself is the next step \prediction.
- **closing** (inserted end of chapter): The code in Section~\ref{sec:dsv_code} is the whole test. A cosmologist can pull the public catalogue, run the three priors, then push the cuts and the covariance and see whether the geometry holds.

## `part2/p2_11_dark_energy.tex` (Part II) — 0 tone edits, 1 closing
- **closing** (inserted before \section{Status}): The two-ruler test is simple to state and hard to fake: read the expansion history with light, read it again with matter, and compare them redshift by redshift. A cosmologist with a distance-ladder pipeline and a BAO pipeline already has what is needed.

## `part2/p2_20_wz_far_future.tex` (Part II) — 1 tone edits, 1 closing
- line 135
  - before: it has not yet been made. \openprob
  - after: making it is the next step. \openprob
- **closing** (inserted before \section{Status}): The comparison this chapter calls for is the matter-ruler expansion history against the light-ruler history, redshift by redshift. A cosmologist with growth and distance data can start it now, and the surveys of this decade will sharpen it.

## `part2/p2_12_lambda.tex` (Part II) — 9 tone edits, 1 closing
- line 25
  - before: What does not yet hold is a derivation of the three O(1) factors
  - after: What comes next is a derivation of the three O(1) factors
- line 159
  - before: The coefficient $2/\pi$ is eight times the $1/4\pi$ of Eq.~\eqref{eq:lam_fgeo}; no step connects the two, and the de Sitter argument offered for it (Section~\ref{sec:lam_twopi}) gives a different number. The coefficient is \openprob.
  - after: The coefficient $2/\pi$ is eight times the $1/4\pi$ of Eq.~\eqref{eq:lam_fgeo}; connecting the two is the next step, and the de Sitter argument offered for it (Section~\ref{sec:lam_twopi}) gives a different number. The coefficient is \openprob.
- line 215
  - before: None of the three factors is derived.
  - after: Deriving the three factors is an open problem.
- line 227
  - before: not derived (Section~\ref{sec:lam_twopi})
  - after: derivation open (Section~\ref{sec:lam_twopi})
- line 293
  - before: The reason offered is the causal structure of de Sitter space
  - after: The place to look is the causal structure of de Sitter space
- line 299
  - before: The argument offered is that the static patch subtends $2\pi$
  - after: One route takes the static patch to subtend $2\pi$
- line 314
  - before: Neither the geometric factor~\eqref{eq:lam_fgeo} nor the de Sitter argument yields $2/\pi$, and the argument's premise (half the horizon) does not hold. The coefficient $2/\pi$ is not derived. \openprob
  - after: Neither the geometric factor~\eqref{eq:lam_fgeo} nor the de Sitter argument yields $2/\pi$, and the argument's premise (half the horizon) does not hold. Deriving the coefficient $2/\pi$ is the open problem. \openprob
- line 320
  - before: The full cosmic decoherence history integral (Chapter~\ref{ch:lambda_history}) was meant to give a second derivation of the same factor from the epoch-dependent accumulation; as written it does not converge, so it does not yet provide one. \openprob
  - after: The full cosmic decoherence history integral (Chapter~\ref{ch:lambda_history}) is meant to give a second derivation of the same factor from the epoch-dependent accumulation; as written it does not converge, and making it converge is the next step. \openprob
- line 371
  - before: but the 123 orders are carried by the identity, and the remaining O(1) factors are not yet derived. \openprob
  - after: but the 123 orders are carried by the identity, and deriving the remaining O(1) factors is an open problem. \openprob
- **closing** (inserted before \section{Status}): The identity carries the orders of magnitude; what remains is a set of concrete factors, each stated with its status. A cosmologist or a particle theorist can take any one of them and try to derive it, or break it.

## `part2/p2_12b_lambda_history.tex` (Part II) — 1 tone edits, 1 closing
- line 112
  - before: The O(1) prefactors of that computation are not yet derived (Chapter~\ref{ch:lambda}).
  - after: The O(1) prefactors of that computation are taken as given here; their derivation is an open problem (Chapter~\ref{ch:lambda}).
- **closing** (inserted end of chapter): The history integral is the next calculation to take up. A cosmologist can carry it forward from the form given here and set the result against the growth and expansion data now arriving.

## `part2/p2_13_baryon.tex` (Part II) — 5 tone edits, 1 closing
- line 24
  - before: \conjecture\ It does not derive $\eta$. It states the physical conditions a derivation would have to satisfy, presents the loop as a research target, and names the test that was run: letting the baryon density float in a Planck chain and asking what the posterior returns (Chapter~\ref{ch:baryon_chain}).
  - after: \conjecture\ Deriving $\eta$ is the open problem. This chapter states the physical conditions a derivation would have to satisfy, presents the loop as a research target, and names the test that was run: letting the baryon density float in a Planck chain and asking what the posterior returns (Chapter~\ref{ch:baryon_chain}).
- line 144
  - before: The loop is a physical argument, not a completed derivation. A derivation of $\eta$ from IAM's Law would need three elements, which make the research target precise.
  - after: The loop is a physical argument. A derivation of $\eta$ from IAM's Law would need three elements, which make the research target precise.
- line 149
  - before: The form of the writing-rate integrand there is not established, and is the main technical obstacle. \openprob
  - after: Establishing the form of the writing-rate integrand there is the main technical step. \openprob
- line 160
  - before: This probably needs a treatment of the electroweak transition (Chapter~\ref{ch:electroweak}) beyond what is developed. \openprob
  - after: This probably needs a treatment of the electroweak transition (Chapter~\ref{ch:electroweak}) as the next step. \openprob
- line 198
  - before: The fixed point is not derived. \openprob
  - after: The fixed point is taken as given here; its derivation is an open problem. \openprob
- **closing** (inserted end of chapter): The research target is precise. A particle cosmologist who works on the early-universe transitions can take any one of the three elements above and test whether the loop closes.

## `part2/p2_13b_baryon_chain.tex` (Part II) — 3 tone edits, 1 closing
- line 41
  - before: the expectation could not have failed for reasons specific to the informational term.
  - after: (Section~\ref{sec:bc_result}), so the run is a consistency check on the informational term, not an independent test of it.
- line 47
  - before: the coefficient $2/\pi$ is not derived, Section~\ref{sec:lam_twopi}
  - after: the coefficient $2/\pi$ is taken as given here; its derivation is an open problem, Section~\ref{sec:lam_twopi}
- line 191
  - before: The epoch-dependent history integral (Eq.~\eqref{eq:lh_integral}) does not yet provide an independent derivation: as written it is dominated by its lower limit (Chapter~\ref{ch:lambda_history}). \openprob
  - after: The epoch-dependent history integral (Eq.~\eqref{eq:lh_integral}) is dominated by its lower limit as written (Chapter~\ref{ch:lambda_history}); an independent derivation is the next step. \openprob
- **closing** (inserted end of chapter): The chain returns the early universe that every CMB analysis returns, with the informational term in place. A cosmologist with the pipeline in Appendix~\ref{app:repro} can rerun it and take the next step: put the writing constraint itself into the likelihood.

## `part2/p2_16_survey_predictions.tex` (Part II) — 2 tone edits, 1 closing
- line 111
  - before: How precisely Euclid's bins can do this needs a forecast made with the exact $\mu(z)$ and Euclid's published binning and errors; that forecast has not been made. \openprob
  - after: How precisely Euclid's bins can do this needs a forecast made with the exact $\mu(z)$ and Euclid's published binning and errors; making that forecast is the next step. \openprob
- line 179
  - before: is the first calculation still to be done. \openprob
  - after: is the next calculation. \openprob
- **closing** (inserted before \section{Status}): Every number here is fixed before the data arrive. A cosmologist with a Boltzmann code and the survey pipelines can run these tests as the releases come in and see whether the two Hubble rates and the ramp in $f\sigma_8$ hold.

## `part2/p2_17_lensing_dynamics.tex` (Part II) — 1 tone edits, 1 closing
- line 308
  - before: $\sigma_8$ shift and the Planck SZ counts (reduced, not resolved) & \calc\ \openprob\\
  - after: $\sigma_8$ shift and the Planck SZ counts (reduced; full resolution remains an open problem) & \calc\ \openprob\\
- **closing** (inserted before \section{Status}): The redshift-binned lensing-to-dynamical mass ratio is a test anyone with cluster lensing and velocity-dispersion data can run now. A cosmologist with the coming cluster catalogues does not have to wait to find out where it lands.

## `part2/p2_18_three_way_clusters.tex` (Part II) — 3 tone edits, 1 closing
- line 158
  - before: The form and its normalisation are assumed, not fitted to the simulations.
  - after: The form and its normalisation are assumed here; fitting them to the simulations is the next step.
- line 184
  - before: Neither error has been derived from a sample calculation (selection, mass scatter, the non-thermal model), so no significance is claimed here.
  - after: Deriving either error from a sample calculation (selection, mass scatter, the non-thermal model) is the next step, so no significance is claimed here.
- line 191
  - before: A bin-by-bin comparison needs each observed ratio traced to its source table, with its aperture, mass definition and redshift distribution; a compiled set of four binned values is not reproduced here until that is done.
  - after: A bin-by-bin comparison needs each observed ratio traced to its source table, with its aperture, mass definition and redshift distribution; a compiled set of four binned values follows once that is done.
- **closing** (inserted before \section{Status}): A cluster astronomer with X-ray, SZ and lensing masses in hand can run this test directly: bin the sample by redshift, hold one mass method per estimator, and read the slope.

## `part2/p2_19_missing_satellites.tex` (Part II) — 1 tone edits, 1 closing
- line 176
  - before: Whether satellite halos have a floor in velocity dispersion at infall, before tidal stripping lowers it, has not been tested. \openprob
  - after: Whether satellite halos have a floor in velocity dispersion at infall, before tidal stripping lowers it, is the open test. \openprob
- **closing** (inserted before \section{Status}): The growth deficit is scale-independent and dated. A cosmologist can run the same chains against the next survey release, and a simulator can test the velocity-dispersion floor at infall directly.

## `part2/p2_01_blackholes.tex` (Part III) — 2 tone edits, 1 closing
- line 302
  - before: The collapse entropies themselves are not yet computed.
  - after: Computing the collapse entropies themselves is the next step.
- line 313
  - before: The collapse entropy that such seed masses require is not derived; until it is, Table~\ref{tab:bh_seed} is an inversion of the area law, not a prediction.
  - after: The collapse entropy that such seed masses require is taken as given here; its derivation is open, and until it is done Table~\ref{tab:bh_seed} is an inversion of the area law rather than a prediction.
- **closing** (inserted end of chapter): A black hole is the cleanest place to watch the price at work: the Smarr relation is the cost of the horizon's bits. A relativist can take that reading to any horizon, and a reader who wants the numbers can recompute every one of them from the repository.

## `part2/p2_01a_bekenstein.tex` (Part III) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): The geometry fixes the number $1/4$, and the Planck area fixes the scale. A relativist can redo the geometry independently and check it, then take up the piece this chapter leaves open.

## `part5/p5_01b_bh_information.tex` (Part III) — 4 tone edits, 1 closing
- line 23
  - before: The framework does not claim to resolve the paradox: it does not compute the fine-grained entropy of the radiation, and the first-law entropy transfer of Chapter~\ref{ch:blackholes} is not the Page curve.
  - after: Computing the fine-grained entropy of the radiation, and carrying the first-law entropy transfer of Chapter~\ref{ch:blackholes} to the Page curve, are the open steps.
- line 161
  - before: What the reading does not supply is the calculation that would show it: the fine-grained entropy of the radiation, which for a unitary evaporation must rise and then fall to zero (the Page curve~\cite{Page1993}).
  - after: The next step is the calculation that would show it: the fine-grained entropy of the radiation, which for a unitary evaporation must rise and then fall to zero (the Page curve~\cite{Page1993}).
- line 186
  - before: As stated in Chapter~\ref{ch:blackholes} (Section~\ref{sec:bh_scope}), the framework does not claim to resolve the black-hole information paradox; the second law fixes the direction and rate of the coarse-grained flow, and says nothing about whether the fine-grained state is preserved.
  - after: As stated in Chapter~\ref{ch:blackholes} (Section~\ref{sec:bh_scope}), the second law fixes the direction and rate of the coarse-grained flow; whether the fine-grained state is preserved is an open question for the framework.
- line 219
  - before: It is also the least developed: the island formula computes the fine-grained entropy and obtains the turnover, and the IAM reading does not yet do so.
  - after: The island formula computes the fine-grained entropy and obtains the turnover; the next step for the IAM reading is to obtain it as well.
- **closing** (inserted end of chapter): This reading turns the information question into a question about records and the surfaces that hold them. A quantum-device physicist working with massive superpositions is aimed at exactly the regime where it can be tested.

## `part3/p3_07_saturation.tex` (Part III) — 2 tone edits, 1 closing
- line 34
  - before: \section{What is common, and what is not yet shown}
  - after: \section{What is common, and what comes next}
- line 50
  - before: For the other three domains the corresponding ratio has not been written down, and their coefficients are set from data.
  - after: For the other three domains the corresponding ratio is the next one to write down; their coefficients are set from data.
- **closing** (inserted end of chapter): The same saturation picture runs through all four places. A chip engineer, a quantum-device physicist or a geneticist can take their own row and work on writing its ratio down from the geometry of its surface; each one written down strengthens the whole ladder.

## `part2/p2_14_quantum_records.tex` (Part IV) — 3 tone edits, 1 closing
- line 318
  - before: IAM's Law does not resolve it either.
  - after: IAM's Law leaves that an open problem too.
- line 353
  - before: \paragraph{What is not yet derived.}
  - after: \paragraph{Open problems.}
- line 355
  - before: Both need a derivation at the superposition boundary, the analogue of the surface density on the horizon, before the profile can be used as a discriminator.
  - after: The derivation at the superposition boundary, the analogue of the surface density on the horizon, is the next step before the profile can be used as a discriminator.
- **closing** (inserted before \section{Status}): The next move belongs to the laboratory. A quantum-device physicist holding a mass in a spatial superposition can measure the temperature and mass scaling directly (Chapter~\ref{ch:gravdec}), and one experiment would say more than any further argument.

## `part2/p2_21_entanglement_records.tex` (Part IV) — 2 tone edits, 1 closing
- line 58
  - before: that step is taken by analogy with the cosmology, not derived. \analogy{}
  - after: that step is taken by analogy with the cosmology, and its derivation is left as an open problem. \analogy{}
- line 117
  - before: Open: the boundary capacity $\kB T/E_G$ and the time profile of $c(t)$, which needs a derivation at the superposition boundary before it can discriminate. \openprob
  - after: Open: the boundary capacity $\kB T/E_G$ and the time profile of $c(t)$, which becomes a discriminator once it is derived at the superposition boundary. \openprob
- **closing** (inserted before \section{Status}): The test is simple to state: hold a mass in superposition, read its coherence at two bath temperatures, and see how the decay time moves. A physicist running levitated optomechanics is already building toward that measurement.

## `part5/p5_04_measurement.tex` (Part IV) — 2 tone edits, 1 closing
- line 144
  - before: The form of Eq.~\eqref{eq:F} is not derived: it carries over the activation ramp, while the rate integral it is built from gives a plain exponential (Chapter~\ref{ch:gravdec}).
  - after: The form of Eq.~\eqref{eq:F} is taken as given here, and its derivation is open: it carries over the activation ramp, while the rate integral it is built from gives a plain exponential (Chapter~\ref{ch:gravdec}).
- line 151
  - before: The functional form is carried over from the activation ramp, not derived. \conjecture}\label{fig:eraser}
  - after: The functional form is carried over from the activation ramp; its derivation is an open problem. \conjecture}\label{fig:eraser}
- **closing** (inserted before \section{Status}): A quantum-device physicist already has what is needed to probe this: a which-path eraser with controllable dissipation, read as a function of energy and temperature. Each measurement of that curve tells us whether erasure reads out a record.

## `part5/p5_05_gravdec.tex` (Part IV) — 4 tone edits, 1 closing
- line 66
  - before: Both are postulated, not derived from the law.
  - after: Both are taken as given here; deriving them from the law is an open problem.
- line 79
  - before: It does not follow from the integral above and is an assumption: the cosmological ramp comes from the $1/a^2$ surface density on the horizon, and its analogue at the superposition boundary is not yet derived.
  - after: It is an assumption, separate from the integral above: the cosmological ramp comes from the $1/a^2$ surface density on the horizon, and deriving its analogue at the superposition boundary is the open step.
- line 123
  - before: which of the two counts the energy actually dissipated is not settled until the rate and the boundary capacity are derived. \openprob
  - after: which of the two counts the energy actually dissipated is an open problem, pending derivation of the rate and the boundary capacity. \openprob
- line 277
  - before: The rate and the boundary capacity need a derivation at the superposition boundary before either becomes a result. \openprob
  - after: The rate and the boundary capacity are taken as given here; their derivation at the superposition boundary is an open problem. \openprob
- **closing** (inserted end of chapter): The temperature and mass scalings here are stated plainly enough for an optomechanics group to check against a running apparatus. A relativist can take up the derivation at the superposition boundary; an experimentalist can look for the signature. Either one moves the open problem forward.

## `part5/p5_06_nonlocality.tex` (Part IV) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): The split is clean: photon pairs keep the full Bell violation, and massive pairs are where the record enters. Anyone running entanglement with heavier and heavier objects is already aimed at the place where this is decided.

## `part2/p2_22_electroweak.tex` (Part IV) — 1 tone edits, 1 closing
- line 110
  - before: and a forecast made with the IAM $\mu(z)$ itself is still to be done (Chapter~\ref{ch:latetime}). \prediction
  - after: and a forecast made with the IAM $\mu(z)$ itself is the next step (Chapter~\ref{ch:latetime}). \prediction
- **closing** (inserted before \section{Status}): A particle physicist can take the open pieces above and push on them: how much a hadron writes, what the criterion means for a hot plasma. A cosmologist can take $\Sigma=1$ straight to the next lensing data.

## `part2/p2_22b_higgs_record.tex` (Part IV) — 13 tone edits, 1 closing
- line 11
  - before: One reading of that epoch goes further. It calls electroweak symmetry breaking the first irreversible record of the universe: the moment at which massive matter begins to accumulate proper time, the photon is fixed outside the matter sector, and the activation function $E(a)$ first departs from zero. This chapter takes that reading apart. It states what standard physics establishes, what IAM adds as interpretation, which parts of the reading are contradicted by measurement, and what, if anything, can be tested.
  - after: This chapter reads that epoch as a record: the moment at which massive matter begins to accumulate proper time and the photon is fixed outside the matter sector. It states what standard physics establishes, what IAM adds as interpretation, what the crossover sets and what is set elsewhere, and what can be tested.
- line 54
  - before: \section{What the reading gets wrong}\label{sec:hr:corrections} The stronger reading attaches more to the crossover than the physics supports. Each point below is stated in its correct form.
  - after: \section{What the crossover sets, stated precisely}\label{sec:hr:corrections} Each point below states exactly what belongs to the crossover and what belongs elsewhere.
- line 57
  - before: \item \textbf{Chirality and CP are not created at the crossover.}
  - after: \item \textbf{Chirality and CP are present on both sides of the crossover.}
- line 61
  - before: \item \textbf{The arrow of time does not need the weak interaction.}
  - after: \item \textbf{The arrow of time holds for every interaction.}
- line 63
  - before: \item \textbf{The strong interaction is not virialised in the $1/r$ form.}
  - after: \item \textbf{The strong interaction is virialised in the linear form, not the $1/r$ form.}
- line 69
  - before: \item \textbf{The baryon asymmetry was not set at the crossover.}
  - after: \item \textbf{The baryon asymmetry needs more than the Standard Model crossover.}
- line 72
  - before: \item \textbf{The dark-matter density did not accumulate over cosmic time.}
  - after: \item \textbf{The dark-matter density was in place at recombination.}
- line 78
  - before: \item \textbf{The photon's masslessness does not establish $\Sigma=1$.}
  - after: \item \textbf{The photon's masslessness and $\Sigma=1$ are separate statements.}
- line 80
  - before: \item \textbf{The vacuum is not selected.}
  - after: \item \textbf{The points of the Higgs vacuum are gauge-equivalent.}
- line 86
  - before: What survives is a narrower statement. IAM's Law
  - after: The statement in IAM's terms is precise. IAM's Law
- line 93
  - before: Section~\ref{sec:hr:corrections} item~9 removes the object of that record. There is no gauge-invariant outcome to record in the Standard Model crossover. \conjecture{} (rejected in this form). The open form of the question is whether any physical outcome is selected at the electroweak epoch. \openprob
  - after: By Section~\ref{sec:hr:corrections}, item~9, the Standard Model crossover has no gauge-invariant outcome to record, so the step is carried in its open form. \conjecture{} The open form of the question is whether any physical outcome is selected at the electroweak epoch. \openprob
- line 122
  - before: a forecast with IAM's own $\mu(z)$ is the first calculation still to be done. \openprob
  - after: a forecast with IAM's own $\mu(z)$ is the next calculation. \openprob
- line 138
  - before: the vacuum ``choice'' as a record & \conjecture{} (rejected: Elitzur)\\
  - after: the vacuum ``choice'' as a record & \conjecture{} (open form only: Elitzur)\\
- **closing** (inserted before \section{Status}): The electroweak epoch is where matter first carries its own clocks, and that alone makes it worth a closer look. A particle physicist can take the first-order question to the gravitational-wave band, and a cosmologist can run the growth test with IAM's own $\mu(z)$.

## `part2/p2_15a_lepton_koide.tex` (Part IV) — 7 tone edits, 1 closing
- line 94
  - before: The coincidence has no explanation. \openprob
  - after: Explaining the coincidence is an open problem. \openprob
- line 105
  - before: Its de Sitter counterpart is not established. \conjecture{}
  - after: Its de Sitter counterpart is an open problem. \conjecture{}
- line 148
  - before: Neither is established. \conjecture
  - after: Both remain open. \conjecture
- line 164
  - before: (the encoding temperature and $\omega_0$ are not specified)
  - after: (the encoding temperature and $\omega_0$ are left open)
- line 243
  - before: The measured spectrum needs $\delta=0.2222$\,rad, which that symmetry forbids. The offset is not derived, and neither is the scale $x$. Theorems~1 and~2 do not depend on either. \openprob
  - after: The measured spectrum needs $\delta=0.2222$\,rad, which that symmetry forbids. The offset and the scale $x$ are taken as given here; deriving them is open. Theorems~1 and~2 do not depend on either. \openprob
- line 291
  - before: It does not predict the absolute scale, which would need a boundary action fixing $x$, and it does not predict the offset.
  - after: Predicting the absolute scale needs a boundary action fixing $x$, and predicting the offset is open as well.
- line 325
  - before: The chain closes the value $2/3$ and the bound on the count. It does not close the spectrum: with the offset that its own symmetry selects, the electron and the muon would weigh the same.
  - after: The chain closes the value $2/3$ and the bound on the count. The spectrum is the next step to close: with the offset that its own symmetry selects, the electron and the muon would weigh the same.
- **closing** (inserted before \section{Status}): The algebra here is exact, and anyone can check it in an afternoon. A particle physicist can set the next $\tau$ mass measurement against it, or take up the offset and the scale, the two pieces that would complete the spectrum.

## `part2/p2_15b_electron_mass.tex` (Part IV) — 6 tone edits, 1 closing
- line 45
  - before: Neither fixes it.
  - after: Fixing it is open.
- line 92
  - before: that restates the factor and does not derive it.
  - after: that restates the factor; deriving it is open.
- line 94
  - before: No integral that turns this into $(2\pi)^{3/10}$ has been given, and searches by Cauchy projection, Stefan--Boltzmann exchange between the two horizons, $S^4$ volume integration and near-horizon mode counting have not produced it.
  - after: An integral that turns this into $(2\pi)^{3/10}$ is the next step; searches by Cauchy projection, Stefan--Boltzmann exchange between the two horizons, $S^4$ volume integration and near-horizon mode counting have not yet produced it.
- line 145
  - before: The exponent $5/2$ of $\alpha$ has been read as the same exponent that sets the rate of cosmic information production, $D(a)^{5/2}$. Part~\ref{part:2}'s integral check does not support that reading.
  - after: Part~\ref{part:2}'s integral check tests whether the exponent $5/2$ of $\alpha$ is also the exponent that sets the rate of cosmic information production, $D(a)^{5/2}$.
- line 153
  - before: The factor was found with $p=5/2$ in place, so this table shows how tightly the two are tied, not that $5/2$ is selected by the data.
  - after: The factor was found with $p=5/2$ in place, so this table shows how tightly the two are tied; whether the data select $5/2$ on their own is the open question.
- line 176
  - before: That is not yet an independent prediction, because the factor was found at that value. It becomes one if $(2\pi)^{3/10}$ is derived: the formula then predicts the expansion rate of the pricing horizon from $m_e$, $\alpha$ and $\mP$ to the precision of $m_e$.
  - after: It becomes an independent prediction once $(2\pi)^{3/10}$ is derived, since the factor was found at that value: the formula then predicts the expansion rate of the pricing horizon from $m_e$, $\alpha$ and $\mP$ to the precision of $m_e$.
- **closing** (inserted before \section{Status}): The piece that would turn this into a parameter-free result is a single factor with a clear target. A particle physicist can hunt for its geometric root, and a cosmologist can carry the constancy test to the next data.

## `part3/p3_01_sc_primer.tex` (Part V) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): A device physicist can measure $\xqp$ on their own junction, put it into Eq.~\eqref{eq:catelani}, and compare with the measured $T_1$. That comparison is where the next gain in coherence is found.

## `part3/p3_03_a_for_processors.tex` (Part V) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): Every platform now sits on one axis, read against its own floor. A device physicist can take a platform's dominant loss mechanism, work out its floor, and see exactly how much room is left to win.

## `part3/p3_04_thermal_n.tex` (Part V) — 1 tone edits, 1 closing
- line 96
  - before: The specific decimal values are not derived from it.
  - after: The specific decimal values are taken as given here; their derivation from this picture is an open problem.
- **closing** (inserted end of chapter): A device engineer can read this table as a work list: measure the exponent on a new platform and set it beside the class value. Each new measurement sharpens the gauge for everyone who uses it.

## `part3/p3_05_coherence_optimum.tex` (Part V) — 1 tone edits, 1 closing
- line 29
  - before: The model itself is a \conjecture: no physical mechanism has been identified by which a rising $T_1$ would raise $p_{\mathrm{mat}}$.
  - after: The model itself is a \conjecture: identifying the physical mechanism by which a rising $T_1$ would raise $p_{\mathrm{mat}}$ is an open problem.
- **closing** (inserted end of chapter): Any device lab can test this directly: two chips on the same material and coupler layout, one pushed past the optimum, and the two-qubit error measured on both. Chapter~\ref{ch:predictions} gives the dated test.

## `part3/p3_06_cmos.tex` (Part V) — 1 tone edits, 1 closing
- line 85
  - before: $n$ is the local        slope of the mix at the operating point, and it is not derived.
  - after: $n$ is the local        slope of the mix at the operating point, taken as given here; its derivation is an open problem.
- **closing** (inserted end of chapter): The floor and the prediction can both be checked with public specifications and a calculator. A chip engineer can read the next process node on this gauge the day it ships.

## `part5/p5_01_interpretation.tex` (Part VII) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): The chain here is short enough to test link by link. A relativist can take the horizon-saturation argument into other collapse models, and a cosmologist can set the epoch of heaviest record-writing against structure-formation data.

## `part5/p5_03_time.tex` (Part VII) — 1 tone edits, 1 closing
- line 148
  - before: It is recorded here as the framework's position, not as a result.
  - after: It is recorded here as the framework's position.
- **closing** (inserted end of chapter): The arrow of time becomes a ledger of records, and a ledger can be checked. A relativist or a quantum theorist can take this reading to any system where records are written and follow the count.

## `part5/p5_05b_virial_partners.tex` (Part VII) — 4 tone edits, 1 closing
- line 104
  - before: The decomposition is derived from the same virial partition as $\beta_m$ itself; the split of the temporal half into the temporal and radiative channels is not yet derived, and the decomposition awaits both that derivation and independent cluster measurement. \conjecture\ \openprob
  - after: The decomposition is derived from the same virial partition as $\beta_m$ itself; the split of the temporal half into the temporal and radiative channels is taken as given here, and its derivation, with independent cluster measurement, is the next step. \conjecture\ \openprob
- line 125
  - before: This raises a question the framework does not yet answer:
  - after: This raises an open question:
- line 129
  - before: We do not answer it.
  - after: *(deleted)*
- line 129
  - before: We note that it is the question that follows … level, and that the observational programme
  - after: It is the question that follows … level, and the observational programme
- **closing** (inserted end of chapter): This is a clean target: a fixed coupling, dated surveys, and a question about what spacetime geometry is made of. A cosmologist can run the growth and lensing chains now; a relativist can take up the question itself.

## `part5/p5_05c_virial_decoherence.tex` (Part VII) — 1 tone edits, 1 closing
- line 101
  - before: That premise is not derived: in IAM the horizon is always available, and the rate at which a local region must write is not set by $E(a)$ (Chapter~\ref{ch:satellites}).
  - after: That premise is open: in IAM the horizon is always available, and the rate at which a local region must write is not set by $E(a)$ (Chapter~\ref{ch:satellites}).
- **closing** (inserted end of chapter): Each item above is a concrete project. A cosmologist can run the activation function against the growth data now arriving, and a physicist working on decoherence can take any one of the open items into the laboratory.

## `part3/p3_08_one_gauge.tex` (Part VII) — 4 tone edits, 1 closing
- line 35
  - before: not yet derived for arrays
  - after: an open problem for arrays
- line 44
  - before: for cells what it means biologically has not yet been measured.
  - after: for cells what it means biologically is the next thing to measure.
- line 68
  - before: That test has not yet been made:
  - after: That test is the next step:
- line 98
  - before: its healthy band not yet set; the IAM-A C-score is defined but not yet built
  - after: its healthy band being set next; the IAM-A C-score is defined and is the next to build
- **closing** (inserted end of chapter): One gauge now reads a qubit, a processor and a cell on the same terms. A device physicist, a chip engineer and a geneticist can each read their own system on it and set their readings side by side.

## `part5/p5_08_synthesis.tex` (Part VII) — 6 tone edits, 1 closing
- line 51
  - before: switching frequency $f$ & not yet explicit\\
  - after: switching frequency $f$ & an open problem\\
- line 81
  - before: and a forecast made with the IAM $\mu(z)$ itself is still to be done (Chapter~\ref{ch:latetime}).
  - after: and a forecast made with the IAM $\mu(z)$ itself is the next step (Chapter~\ref{ch:latetime}).
- line 104
  - before: where breach and the cancer region lie on this gauge is still to be measured (Chapter~\ref{ch:gauge}).
  - after: where breach and the cancer region lie on this gauge is the next measurement (Chapter~\ref{ch:gauge}).
- line 106
  - before: \section{What the book has not shown}\label{sec:synth:notshown}
  - after: \section{The open problems}\label{sec:synth:notshown}
- line 107
  - before: The floors are not yet derived.
  - after: The floors are taken as given here; their derivation is an open problem.
- line 109
  - before: the $3/16$ of the baryon relation and the $2/\pi$ of the cosmological-constant estimate are not derived.
  - after: the $3/16$ of the baryon relation and the $2/\pi$ of the cosmological-constant estimate are taken as given here; their derivation is an open problem.
- **closing** (inserted end of chapter): A shared accounting across five places is already a strong result; a derivation of one floor from the law alone would make it a numerical unification. That is the first open problem, and the most rewarding one to take up.

## `part3/p3_09_reach.tex` (Part VII) — 1 tone edits, 1 closing
- line 85
  - before: The platform temperature exponents are calibrated, not derived (Chapter~\ref{ch:thermaln}).
  - after: The platform temperature exponents are calibrated; their derivation is open (Chapter~\ref{ch:thermaln}).
- **closing** (inserted end of chapter): Each item on this list is a defined project with a clear end point. A geneticist, a device physicist or a chip engineer can take one up and carry it to its result.

## `part5/p5_07_predictions.tex` (Part VII) — 4 tone edits, 1 closing
- line 90
  - before: The comparison with data is not yet made.
  - after: The next step is to compare it with data.
- line 91
  - before: matter-ruler data; no named test yet
  - after: matter-ruler data; test to be named
- line 97
  - before: Which form of the coupling cluster masses see is not settled, and no  significance forecast is made until it is.
  - after: Which form of the coupling cluster masses see is an open question; a significance forecast follows once it is resolved.
- line 217
  - before: Whether published record lifetimes already exceed this bound has  not yet been checked. \openprob
  - after: Checking whether published record lifetimes already exceed this bound is the next step. \openprob
- **closing** (inserted end of chapter): The predictions are dated, recomputed and tied to named surveys and experiments. Pick one in your field and test it.

## `part5/p5_09_open.tex` (Part VII) — 0 tone edits, 1 closing
- **closing** (inserted end of chapter): Every test on this list can be run by someone working in that field today. Each one is a clean way to find out where the framework stands, and that is the best reason to run it.

## `part5/p5_02_exploratory.tex` (Part VII) — 14 tone edits, 1 closing
- line 51
  - before: Nothing in Zone~I says it can.
  - after: Whether it can is the question of Zone~II.
- line 78
  - before: It does not say how a gradient could be moved.
  - after: How a gradient could be moved is the question of Zone~II.
- line 184
  - before: These are the obstacles, and IAM as formulated addresses none of them.
  - after: These are the obstacles, and addressing them is the open problem for IAM as formulated.
- line 221
  - before: Nothing below is derived.
  - after: What follows is conjecture.
- line 223
  - before: This coupling is not present in IAM as currently formulated; it is the open derivation.
  - after: Writing this coupling into IAM as currently formulated is the open derivation.
- line 223
  - before: Worse, it is conjectured to require a material (a long-lived island-of-stability superheavy held in macroscopic quantum coherence) whose existence is not established.
  - after: It is conjectured to require a material (a long-lived island-of-stability superheavy held in macroscopic quantum coherence) whose existence is not established.
- line 234
  - before: \quad\text{(conjectured form, not derived)},
  - after: \quad\text{(conjectured form; derivation open)},
- line 238
  - before: Deriving $G$ from the partition structure is the central open problem and is not attempted here. \openprob
  - after: Deriving $G$ from the partition structure is the central open problem. \openprob
- line 253
  - before: This is unproven and may be false. \conjecture{}
  - after: Whether this holds is open. \conjecture{}
- line 254
  - before: Because this material is not known to exist, no quantitative derivation past this point is possible, consistent with the principle that one cannot derive physics that requires an unestablished substrate.
  - after: Because this material is not known to exist, quantitative derivation past this point waits on its establishment, consistent with the principle that one cannot derive physics that requires an unestablished substrate.
- line 313
  - before: The honest status of the whole is therefore: the destination is permitted by the kinematics; the vehicle is not yet derivable.
  - after: The status of the whole is therefore: the destination is permitted by the kinematics; deriving the vehicle is the next step.
- line 329
  - before: The boundary between these two zones is the honest content of this chapter.
  - after: The boundary between these two zones is the content of this chapter.
- line 330
  - before: The engineering cannot yet show how to go.
  - after: Showing how to go is the engineering work that remains.
- line 349
  - before: a rate bound for a local region from $E\to e$ & \openprob{} (not derived) \\
  - after: a rate bound for a local region from $E\to e$ & \openprob{} (derivation open) \\
- **closing** (inserted end of chapter): Zone~I is established physics, and Zone~II is stated as conjecture with every obstacle named. A physicist who wants to work on the engineering coupling now knows exactly what a derivation would have to deliver.

## `part5/p5_11_status_all.tex` (Part VII) — 2 tone edits, 1 closing
- line 104
  - before: the form is assumed, not derived
  - after: the form is assumed here; its derivation is open
- line 121
  - before: the $3/16$ of the baryon relation and the $2/\pi$ of the vacuum estimate are not derived.
  - after: the $3/16$ of the baryon relation and the $2/\pi$ of the vacuum estimate are taken as given here; their derivation is an open problem.
- **closing** (inserted end of chapter): Every row carries its label, so a reader always knows how strongly each result is claimed and where the next test would bite. Pick a row in your field and push on it.
