# MANIFEST -- bibliography audit and glossary regeneration (2026-10-03)

Base: work started on HEAD 5f5997b; every insertion anchor, glossary pointer, number trace and canon check was re-run on HEAD **3290e60** (author rulings: IAM never expanded; Level 2b author wording restored; x_qp chapter restored). Not compiled (no TeX); static checks only.

## Files in this zip

| repo path | what | ownership |
|---|---|---|
| docs/book/appendices/app_F_glossary.tex | regenerated glossary, 677 entries | OWNED (replace whole file) |
| docs/book/bib_additions.bib | 5 new entries for references carried in the book (B-blocks below) | new file: append to iam.bib or add to \bibliography |
| docs/book/corrected_entries.bib | 126 iam.bib entries with the missing DOI added (each replaces the entry of the same key; only `doi` added) | new file |
| docs/book/bib_insertions.json | the 32 citation insertion blocks in machine-readable form (anchor, old, new, full replacement line) | new file |
| docs/verification/scripts/verify_bib_glossary.py | checks: DOIs on CrossRef/arXiv, anchors unique, keys present, glossary refs/cites/numbers/retired terms | new file |
| docs/verification/scripts/verify_bib_glossary_output.txt | output: 131/131 DOIs OK (online), 32/32 blocks OK, glossary 0 FAIL (offline at 3290e60) | new file |
| MANIFEST.md | this file | -- |

## 1. Line counts read

**Source papers (bibliographies and every \cite context):** 861 `\bibitem` in 38 .tex files (34 under docs/papers/latex; 4 outside the sparse set, fetched from git: MethylPhys papers and docs/RETIRED_2026-10, shown with prefix extra/) + 140 entries in 5 .bib files (3 under docs/papers/latex; MethylPhys refs.bib 37; refs_vertebrate.bib 26). 1,350 citation contexts extracted. Files were parsed in full by script, not read in 50-line chunks by eye.

| source file | lines |
|---|---|
| docs/papers/latex/Dual_Sector_Validation_Paper/Dual_Sector_Validation_Paper.tex | 1004 |
| docs/papers/latex/Evidence_Baryon/Evidence_Baryon.tex | 383 |
| docs/papers/latex/GRF_Essay_Final/GRF_Essay_Final.tex | 360 |
| docs/papers/latex/Gravitational_Decoherence/Gravitational_Decoherence.tex | 772 |
| docs/papers/latex/Holographic_Derivation_of_IAM/Holographic_Derivation_of_IAM.tex | 585 |
| docs/papers/latex/IAM_BH_Cosmology_Paper_B/IAM_BH_Cosmology_Paper_B.tex | 675 |
| docs/papers/latex/IAM_BH_Thermodynamics_Paper/IAM_BH_Thermodynamics_Paper.tex | 532 |
| docs/papers/latex/IAM_CAMB_Technical_Note/IAM_CAMB_Technical_Note.tex | 1053 |
| docs/papers/latex/IAM_Lensing_Dynamics_Paper/IAM_Lensing_Dynamics_Paper.tex | 498 |
| docs/papers/latex/IAM_Saridakis_Bridge/IAM_Saridakis_Bridge.tex | 559 |
| docs/papers/latex/IAM_Survey_Predictions_Paper/IAM_Survey_Predictions_Paper.tex | 515 |
| docs/papers/latex/IAM_ThreeWay_Cluster_Paper/IAM_ThreeWay_Cluster_Paper.tex | 499 |
| docs/papers/latex/IAM_Virial_37_Orders_Paper/IAM_Virial_37_Orders_Paper.tex | 794 |
| docs/papers/latex/IAM_Virial_Efficiency_neff_Paper/IAM_Virial_Efficiency_neff_Paper.tex | 549 |
| docs/papers/latex/IAM_wz_FarFuture_Paper/IAM_wz_FarFuture_Paper.tex | 499 |
| docs/papers/latex/Koide_Paper/Koide_Paper.tex | 360 |
| docs/papers/latex/Paper2_Dual-Sector_Cosmology_and_Hubble_Tension/Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex | 981 |
| docs/papers/latex/Paper9_Measurement_Problem/Paper9_Measurement_Problem.tex | 668 |
| docs/papers/latex/Variational_Derivation_of_IAM/Variational_Derivation_of_IAM.tex | 949 |
| docs/papers/latex/arXiv_Master_file/arXiv_Master_file.tex | 2497 |
| docs/papers/latex/electron_mass/electron_mass.tex | 487 |
| docs/papers/latex/iam_baryon_asymmetry/iam_baryon_asymmetry.tex | 704 |
| docs/papers/latex/iam_bekenstein_coefficient/iam_bekenstein_coefficient.tex | 764 |
| docs/papers/latex/iam_cosmological_constant/iam_cosmological_constant.tex | 1034 |
| docs/papers/latex/iam_decoherence_bridge_paper/iam_decoherence_bridge_paper.tex | 748 |
| docs/papers/latex/iam_decoherence_virial_partition/iam_decoherence_virial_partition.tex | 582 |
| docs/papers/latex/iam_desi_paper/iam_desi_paper.tex | 1300 |
| docs/papers/latex/iam_electroweak_matter_sector/iam_electroweak_matter_sector.tex | 281 |
| docs/papers/latex/iam_entanglement_decoherence/iam_entanglement_decoherence.tex | 301 |
| docs/papers/latex/iam_missing_satellites/iam_missing_satellites.tex | 534 |
| docs/papers/latex/iam_mu_sigma_paper/iam_mu_sigma_paper.tex | 592 |
| docs/papers/latex/iam_ocolgain_note/iam_ocolgain_note.tex | 418 |
| docs/papers/latex/iam_theory_paper/iam_theory_paper.tex | 2418 |
| docs/papers/latex/iam_thermodynamic_identity/iam_thermodynamic_identity.tex | 790 |
| docs/papers/latex/iam_two_faces_of_time/iam_two_faces_of_time.tex | 362 |
| docs/papers/latex/iam_virial_dark_sector/iam_virial_dark_sector.tex | 664 |
| docs/papers/latex/zurek_paper/zurek_paper.tex | 902 |
| extra/Biological_Physics_MethylPhys_papers_IAM_Hubble2Methyl_Alpha_Omega_5.tex | 7066 |
| extra/Biological_Physics_MethylPhys_papers_IAM_for_physicists_IAM_for_physicists.tex | 1082 |
| extra/Biological_Physics_MethylPhys_papers_Landauer_Metrology_of_the_Methylome.tex | 192 |
| extra/Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex | 1924 |
| extra/Biological_Physics_RETIRED_2026-10_MethylPhys_papers_iam_vertebrate_lifespan.tex | 741 |
| extra/docs_RETIRED_2026-10_top_level_development_IAM_Manuscript.tex | 717 |
| **total** | **39335** |

**Book chapters (glossary source), main.tex order at 3290e60:**

| file | lines |
|---|---|
| docs/book/part0/p0_preface.tex | 52 |
| docs/book/part0/p0_giants.tex | 208 |
| docs/book/part0/p0_how_to_read.tex | 59 |
| docs/book/part1/p1_01_encoding_surfaces.tex | 216 |
| docs/book/part1/p1_02_iams_law.tex | 1020 |
| docs/book/part1/p1_03_virial_law.tex | 177 |
| docs/book/part1/p1_04_virial_identity.tex | 162 |
| docs/book/part2/p2_01_blackholes.tex | 410 |
| docs/book/part2/p2_01a_bekenstein.tex | 417 |
| docs/book/part2/p2_02_virial.tex | 345 |
| docs/book/part2/p2_02b_virial_tests.tex | 155 |
| docs/book/part2/p2_03_theory.tex | 1094 |
| docs/book/part2/p2_03a_entropic_gravity.tex | 321 |
| docs/book/part2/p2_04_dualsector_chains.tex | 265 |
| docs/book/part2/p2_07_late_time_growth.tex | 425 |
| docs/book/part2/p2_06_dual_sector_perturbation.tex | 508 |
| docs/book/part2/p2_05_dual_sector_note.tex | 243 |
| docs/book/part2/p2_08_s8_trend.tex | 208 |
| docs/book/part2/p2_09_sector_tension.tex | 306 |
| docs/book/part2/p2_09b_phantom_crossing.tex | 163 |
| docs/book/part2/p2_10_dual_sector_validation.tex | 606 |
| docs/book/part2/p2_11_dark_energy.tex | 270 |
| docs/book/part2/p2_12_lambda.tex | 400 |
| docs/book/part2/p2_12b_lambda_history.tex | 162 |
| docs/book/part2/p2_13_baryon.tex | 212 |
| docs/book/part2/p2_13b_baryon_chain.tex | 207 |
| docs/book/part2/p2_14_quantum_records.tex | 376 |
| docs/book/part2/p2_15a_lepton_koide.tex | 327 |
| docs/book/part2/p2_15b_electron_mass.tex | 199 |
| docs/book/part2/p2_16_survey_predictions.tex | 276 |
| docs/book/part2/p2_17_lensing_dynamics.tex | 312 |
| docs/book/part2/p2_18_three_way_clusters.tex | 266 |
| docs/book/part2/p2_19_missing_satellites.tex | 192 |
| docs/book/part2/p2_20_wz_far_future.tex | 230 |
| docs/book/part2/p2_21_entanglement_records.tex | 120 |
| docs/book/part2/p2_22_electroweak.tex | 120 |
| docs/book/part2/p2_22b_higgs_record.tex | 142 |
| docs/book/part3/p3_01_sc_primer.tex | 117 |
| docs/book/part3/p3_02_xqp.tex | 100 |
| docs/book/part3/p3_03_a_for_processors.tex | 83 |
| docs/book/part3/p3_04_thermal_n.tex | 112 |
| docs/book/part3/p3_05_coherence_optimum.tex | 79 |
| docs/book/part3/p3_06_cmos.tex | 111 |
| docs/book/part3/p3_07_saturation.tex | 70 |
| docs/book/part4/p4_01_bridge.tex | 182 |
| docs/book/part4/p4_02_landauer.tex | 272 |
| docs/book/part4/p4_03_surface.tex | 128 |
| docs/book/part4/p4_04_ledgers.tex | 135 |
| docs/book/part4/p4_05_floorbreach.tex | 107 |
| docs/book/part4/p4_06_gauge.tex | 139 |
| docs/book/part4/p4_07_meta.tex | 117 |
| docs/book/part4/p4_08_iama.tex | 107 |
| docs/book/part4/p4_09_cscore.tex | 78 |
| docs/book/part4/p4_10_temperature.tex | 74 |
| docs/book/part4/p4_11_translation.tex | 124 |
| docs/book/part4/p4_12_instrument.tex | 153 |
| docs/book/part4/p4_13_separation.tex | 136 |
| docs/book/part4/p4_14_atlas.tex | 237 |
| docs/book/part4/p4_15_identity.tex | 80 |
| docs/book/part4/p4_16a_skytools.tex | 263 |
| docs/book/part4/p4_16_sky.tex | 91 |
| docs/book/part4/p4_17_serial.tex | 87 |
| docs/book/part4/p4_18_discipline.tex | 124 |
| docs/book/part4/p4_19_chain.tex | 123 |
| docs/book/part4/p4_20_report.tex | 96 |
| docs/book/part4/p4_21_firstreadings.tex | 107 |
| docs/book/part4/p4_22_leukocyte.tex | 87 |
| docs/book/part4/p4_22b_salmonid.tex | 223 |
| docs/book/part4/p4_23_reach.tex | 125 |
| docs/book/part4/p4_24_status.tex | 69 |
| docs/book/part5/p5_01_interpretation.tex | 66 |
| docs/book/part5/p5_01b_bh_information.tex | 242 |
| docs/book/part5/p5_03_time.tex | 150 |
| docs/book/part5/p5_04_measurement.tex | 259 |
| docs/book/part5/p5_05_gravdec.tex | 281 |
| docs/book/part5/p5_05b_virial_partners.tex | 158 |
| docs/book/part5/p5_05c_virial_decoherence.tex | 179 |
| docs/book/part5/p5_06_nonlocality.tex | 67 |
| docs/book/part5/p5_08_synthesis.tex | 114 |
| docs/book/part3/p3_08_one_gauge.tex | 158 |
| docs/book/part3/p3_09_reach.tex | 125 |
| docs/book/part5/p5_07_predictions.tex | 290 |
| docs/book/part5/p5_09_open.tex | 146 |
| docs/book/part5/p5_02_exploratory.tex | 536 |
| docs/book/part5/p5_11_status_all.tex | 120 |
| docs/book/part5/p5_10_conclusion.tex | 37 |
| docs/book/appendices/app_A_canon.tex | 53 |
| docs/book/appendices/app_A2_frozen_values.tex | 23 |
| docs/book/appendices/app_N_notation.tex | 338 |
| docs/book/appendices/app_E_formulas.tex | 255 |
| docs/book/appendices/app_C3_derivations.tex | 373 |
| docs/book/appendices/app_F_glossary.tex | 718 |
| docs/book/appendices/app_C_reproduce_physics.tex | 38 |
| docs/book/appendices/app_C2_reproduce_cells.tex | 37 |
| docs/book/appendices/app_G_predictions_register.tex | 195 |
| docs/book/appendices/app_I_provenance.tex | 397 |
| **total** | **20662** |

How the chapters were read for the glossary: every Part 0-5 chapter was cut into 133 chunks (about 18,000 characters each) and each chunk was sent for term extraction; 110 chunks returned terms (64 of them truncated at the output limit, so only partly), 23 returned nothing because the per-session model budget ran out (Open item 1). All chapters were also scanned in full by script: acronym sweep, text search of every glossary headword, number tracing, pointer recomputation, retired-word scan. I did not read the chapter lines one by one myself.

## 2. Bibliography

Method: every source reference was resolved to a DOI on CrossRef (a printed DOI was checked on /works and validated against title, year, first author, volume/page; otherwise `query.bibliographic`, top 5-10 candidates validated the same way). 713 unique non-self references: 203 printed DOIs validated, 375 found by search, 34 arXiv-only, 101 unresolved (books, conference reprints, short refs without a title). 128 source references are the author's own papers and were not considered (stand-alone rule). Each source citation sentence was located in the book by 4-word shingle overlap; the 494 contexts with overlap >= 0.25 are tabulated in 2.6; the 854 below 0.25 are treated as not carried (threshold choice, Open item 6).

### 2.1 (a) Carried content missing its citation -- 32 insertion blocks

Format: target file | anchor line copied exactly (occurs once at 3290e60) | replace | new line. Keys marked * are new (in bib_additions.bib).

**B01** `docs/book/part1/p1_04_virial_identity.tex` (line 147 at 3290e60) | REPLACE | keys: Jacobson1995 | source: iam_thermodynamic_identity.tex:770 (also iam_baryon_asymmetry:611, Evidence_Baryon:338, iam_cosmological_constant:928, iam_missing_satellites:514, iam_decoherence_virial_partition:549)

anchor:
```latex
spacetime. \interp\ The thermodynamic framework of Jacobson is the foundation upon which this work is built.
```
replacement:
```latex
spacetime. \interp\ The thermodynamic framework of Jacobson~\cite{Jacobson1995} is the foundation upon which this work is built.
```

**B02** `docs/book/part1/p1_02_iams_law.tex` (line 684 at 3290e60) | REPLACE | keys: Planck2018VI, Hojjati2011, Torrado2021 | source: iam_missing_satellites.tex:232-233

anchor:
```latex
free parameter. Seventeen MCMC chains were run against the full Planck 2018 likelihood (TT, TE, EE, low-$\ell$ and lensing). All converged to $R-1\le0.010$
```
replacement:
```latex
free parameter. Seventeen MCMC chains were run against the full Planck 2018 likelihood~\cite{Planck2018VI} (TT, TE, EE, low-$\ell$ and lensing) using MGCAMB~\cite{Hojjati2011} and Cobaya~\cite{Torrado2021}. All converged to $R-1\le0.010$
```

**B03** `docs/book/part1/p1_02_iams_law.tex` (line 114 at 3290e60) | REPLACE | keys: Zurek2003 | source: iam_theory_paper.tex:183

anchor:
```latex
where $\{|s_i\rangle\}$ are the pointer states selected by the system--environment interaction. When the environmental states become approximately
```
replacement:
```latex
where $\{|s_i\rangle\}$ are the pointer states selected by the system--environment interaction~\cite{Zurek2003}. When the environmental states become approximately
```

**B04** `docs/book/part1/p1_02_iams_law.tex` (line 130 at 3290e60) | REPLACE | keys: Landauer1961 | source: iam_cosmological_constant.tex:207

anchor:
```latex
Landauer's principle establishes that producing --- equivalently, irreversibly encoding --- one bit of classical information requires a minimum energy
```
replacement:
```latex
Landauer's principle~\cite{Landauer1961} establishes that producing --- equivalently, irreversibly encoding --- one bit of classical information requires a minimum energy
```

**B05** `docs/book/part1/p1_02_iams_law.tex` (line 253 at 3290e60) | REPLACE | keys: Wald1984 | source: iam_theory_paper.tex:362

anchor:
```latex
law), and the factor 2 is the trace structure of $G_{ab}=R_{ab}-\tfrac12Rg_{ab}$ in the weak-field limit. For a static weak field,
```
replacement:
```latex
law), and the factor 2 is the trace structure of $G_{ab}=R_{ab}-\tfrac12Rg_{ab}$ in the weak-field limit~\cite{Wald1984}. For a static weak field,
```

**B06** `docs/book/part1/p1_01_encoding_surfaces.tex` (line 55 at 3290e60) | REPLACE | keys: Bennett1982 | source: Mahaffey_2026_cell_thermodynamics.tex:236 (RETIRED MethylPhys)

anchor:
```latex
energy~\cite{Landauer1961}
```
replacement:
```latex
energy~\cite{Landauer1961,Bennett1982}
```

**B07** `docs/book/part1/p1_01_encoding_surfaces.tex` (line 101 at 3290e60) | REPLACE | keys: Bestor2000*, Jurkowska2011* | source: Mahaffey_2026_cell_thermodynamics.tex:246 (RETIRED MethylPhys)

anchor:
```latex
operation: each correct maintenance event restores one bit of epigenomic information, at Landauer cost $k_BT_{\rm body}\ln2=2.97\times10^{-21}$\,J per
```
replacement:
```latex
operation: each correct maintenance event restores one bit of epigenomic information~\cite{Bestor2000,Jurkowska2011}, at Landauer cost $k_BT_{\rm body}\ln2=2.97\times10^{-21}$\,J per
```

**B08** `docs/book/part1/p1_03_virial_law.tex` (line 82 at 3290e60) | REPLACE | keys: Clausius1870 | source: IAM_Virial_37_Orders_Paper.tex:177

anchor:
```latex
analytical consequence of the $1/r$ potential, Eq.~\eqref{eq:vl_virial}. These systems show that quantum mechanics obeys Eq.~\eqref{eq:vl_virial} with the
```
replacement:
```latex
analytical consequence of the $1/r$ potential~\cite{Clausius1870}, Eq.~\eqref{eq:vl_virial}. These systems show that quantum mechanics obeys Eq.~\eqref{eq:vl_virial} with the
```

**B09** `docs/book/part2/p2_01_blackholes.tex` (line 59 at 3290e60) | REPLACE | keys: Landauer1961 | source: arXiv_Master_file.tex:591

anchor:
```latex
The maximum rate at which a black-hole horizon can encode information is set by the ratio of the available radiated power to the Landauer cost per
```
replacement:
```latex
The maximum rate at which a black-hole horizon can encode information is set by the ratio of the available radiated power to the Landauer cost~\cite{Landauer1961} per
```

**B10** `docs/book/part2/p2_01_blackholes.tex` (line 64 at 3290e60) | REPLACE | keys: Hawking1975 | source: arXiv_Master_file.tex:596; IAM_BH_Thermodynamics_Paper.tex:87

anchor:
```latex
Using the black-body Hawking luminosity $P=\hbar c^6/(15360\pi G^2M^2)$ (Section~\ref{sec:bh_sb}) and the Hawking temperature $T_{\rm BH}=\hbar c^3/(8\pi GM\kB)$:
```
replacement:
```latex
Using the black-body Hawking luminosity~\cite{Hawking1975} $P=\hbar c^6/(15360\pi G^2M^2)$ (Section~\ref{sec:bh_sb}) and the Hawking temperature $T_{\rm BH}=\hbar c^3/(8\pi GM\kB)$:
```

**B11** `docs/book/part2/p2_02_virial.tex` (line 45 at 3290e60) | REPLACE | keys: Jacobson1995, CaiKim2005 | source: iam_virial_dark_sector.tex:135

anchor:
```latex
Landauer's principle~\cite{Landauer1961}. This information accumulates on the cosmological apparent horizon as informational entropy $S_{\rm info}$.
```
replacement:
```latex
Landauer's principle~\cite{Landauer1961}. This information accumulates on the cosmological apparent horizon as informational entropy $S_{\rm info}$~\cite{Jacobson1995,CaiKim2005}.
```

**B12** `docs/book/part2/p2_02_virial.tex` (line 58 at 3290e60) | REPLACE | keys: Landauer1961 | source: zurek_paper.tex:260; iam_decoherence_virial_partition.tex:150

anchor:
```latex
describes. The kinetic half $K=|V|/2$ is the decoherence energy: the Landauer cost of converting quantum superpositions to classical outcomes at each
```
replacement:
```latex
describes. The kinetic half $K=|V|/2$ is the decoherence energy: the Landauer cost~\cite{Landauer1961} of converting quantum superpositions to classical outcomes at each
```

**B13** `docs/book/part2/p2_02_virial.tex` (line 73 at 3290e60) | REPLACE | keys: Tinker2008 | source: IAM_Virial_Efficiency_neff_Paper.tex:132

anchor:
```latex
where $f_{\rm coll}$ is the collapsed fraction from mass functions and $\eta_{\rm vir}$ is the virial efficiency --- the fraction of collapsed matter's
```
replacement:
```latex
where $f_{\rm coll}$ is the collapsed fraction from mass functions~\cite{Tinker2008} and $\eta_{\rm vir}$ is the virial efficiency --- the fraction of collapsed matter's
```

**B14** `docs/book/part2/p2_03_theory.tex` (line 49 at 3290e60) | REPLACE | keys: Bekenstein1973, Hawking1975 | source: iam_decoherence_bridge_paper.tex:85

anchor:
```latex
horizons, provided the entropy is proportional to the horizon area. Cai and Kim~\cite{CaiKim2005} extended this to the FRW apparent horizon, recovering the
```
replacement:
```latex
horizons, provided the entropy is proportional to the horizon area~\cite{Bekenstein1973,Hawking1975}. Cai and Kim~\cite{CaiKim2005} extended this to the FRW apparent horizon, recovering the
```

**B15** `docs/book/part2/p2_03_theory.tex` (line 146 at 3290e60) | REPLACE | keys: Bekenstein1973 | source: arXiv_Master_file.tex:385

anchor:
```latex
\emph{Entropy variation.} Assuming entropy proportional to horizon area, $S=\eta A$:
```
replacement:
```latex
\emph{Entropy variation.} Assuming entropy proportional to horizon area~\cite{Bekenstein1973}, $S=\eta A$:
```

**B16** `docs/book/part2/p2_03_theory.tex` (line 222 at 3290e60) | REPLACE | keys: GibbonsHawking1977 | source: arXiv_Master_file.tex:440

anchor:
```latex
with area $A_H=4\pi/H^2$, geometric entropy and temperature
```
replacement:
```latex
with area $A_H=4\pi/H^2$, geometric entropy and temperature~\cite{GibbonsHawking1977}
```

**B17** `docs/book/part2/p2_03_theory.tex` (line 242 at 3290e60) | REPLACE | keys: Landauer1961 | source: Holographic_Derivation_of_IAM.tex:108

anchor:
```latex
This temperature has direct physical significance through Landauer's principle: encoding or erasing one bit of information on the horizon requires a
```
replacement:
```latex
This temperature has direct physical significance through Landauer's principle~\cite{Landauer1961}: encoding or erasing one bit of information on the horizon requires a
```

**B18** `docs/book/part2/p2_03_theory.tex` (line 742 at 3290e60) | REPLACE | keys: CaiKim2005 | source: IAM_Virial_37_Orders_Paper.tex:217

anchor:
```latex
The constrained scalar field $\varphi=1-1/a$ is defined by the Cai--Kim first law applied to the cosmic apparent horizon. The apparent horizon is a
```
replacement:
```latex
The constrained scalar field $\varphi=1-1/a$ is defined by the Cai--Kim first law~\cite{CaiKim2005} applied to the cosmic apparent horizon. The apparent horizon is a
```

**B19** `docs/book/part2/p2_03_theory.tex` (line 899 at 3290e60) | REPLACE | keys: Einstein1915 | source: iam_theory_paper.tex:1846; Holographic_Derivation_of_IAM.tex:376

anchor:
```latex
\item \emph{Gravitational interaction.} Matter interacts gravitationally, initiating collapse of overdense regions (general relativity).
```
replacement:
```latex
\item \emph{Gravitational interaction.} Matter interacts gravitationally, initiating collapse of overdense regions (general relativity~\cite{Einstein1915}).
```

**B20** `docs/book/part2/p2_03_theory.tex` (line 900 at 3290e60) | REPLACE | keys: Peebles1980* | source: iam_theory_paper.tex:1848; Holographic:377; arXiv_Master_file:252

anchor:
```latex
\item \emph{Gravitational collapse.} Matter clusters into bound systems: halos, galaxies, stars (structure-formation theory).
```
replacement:
```latex
\item \emph{Gravitational collapse.} Matter clusters into bound systems: halos, galaxies, stars (structure-formation theory~\cite{Peebles1980}).
```

**B21** `docs/book/part2/p2_03_theory.tex` (line 975 at 3290e60) | REPLACE | keys: CarrollChen2004* | source: iam_theory_paper.tex:1986; Holographic_Derivation_of_IAM.tex:401

anchor:
```latex
open problems in physics~\cite{Penrose1989}. The standard account attributes irreversibility to the low-entropy initial condition of the Big Bang, but offers
```
replacement:
```latex
open problems in physics~\cite{Penrose1989,CarrollChen2004}. The standard account attributes irreversibility to the low-entropy initial condition of the Big Bang, but offers
```

**B22** `docs/book/part2/p2_03_theory.tex` (line 995 at 3290e60) | REPLACE | keys: Jacobson1995 | source: arXiv_Master_file.tex:1983

anchor:
```latex
\emph{Step 1 (Jacobson):} $\delta Q=T\,dS$ on local Rindler horizons, with $S=\eta A$, implies $G_{ab}+\Lambda g_{ab}=8\pi GT_{ab}$.
```
replacement:
```latex
\emph{Step 1 (Jacobson~\cite{Jacobson1995}):} $\delta Q=T\,dS$ on local Rindler horizons, with $S=\eta A$, implies $G_{ab}+\Lambda g_{ab}=8\pi GT_{ab}$.
```

**B23** `docs/book/part2/p2_03_theory.tex` (line 997 at 3290e60) | REPLACE | keys: CaiKim2005 | source: arXiv_Master_file.tex:1988

anchor:
```latex
\emph{Step 2 (Cai--Kim):} $-dE=T\,dS$ on the FRW apparent horizon, with $S_{\rm geo}=A_H/(4G)$ and $T=H/(2\pi)$, implies $H^2=(8\pi G/3)\rho+\Lambda/3$.
```
replacement:
```latex
\emph{Step 2 (Cai--Kim~\cite{CaiKim2005}):} $-dE=T\,dS$ on the FRW apparent horizon, with $S_{\rm geo}=A_H/(4G)$ and $T=H/(2\pi)$, implies $H^2=(8\pi G/3)\rho+\Lambda/3$.
```

**B24** `docs/book/part2/p2_03_theory.tex` (line 1023 at 3290e60) | REPLACE | keys: DESY3Ext | source: iam_theory_paper.tex:2075; iam_decoherence_bridge_paper.tex:560

anchor:
```latex
The $\mu$--$\Sigma$ framework is used extensively in observational analyses~\cite{PogosianSilvestri2016,Andrade2024,DESI2024VII}. Existing theoretical motivations from
```
replacement:
```latex
The $\mu$--$\Sigma$ framework is used extensively in observational analyses~\cite{PogosianSilvestri2016,DESY3Ext,Andrade2024,DESI2024VII}. Existing theoretical motivations from
```

**B25** `docs/book/part2/p2_08_s8_trend.tex` (line 90 at 3290e60) | REPLACE | keys: Planck2018VI | source: iam_ocolgain_note.tex:150

anchor:
```latex
inferred value is $S_8^{\rm Planck}\,D_{\rm IAM}(z)/D_{\Lambda\rm CDM}(z)$, with $S_8^{\rm Planck}=0.832$ and both growth factors from the linear
```
replacement:
```latex
inferred value is $S_8^{\rm Planck}\,D_{\rm IAM}(z)/D_{\Lambda\rm CDM}(z)$, with $S_8^{\rm Planck}=0.832$~\cite{Planck2018VI} and both growth factors from the linear
```

**B26** `docs/book/part2/p2_19_missing_satellites.tex` (line 32 at 3290e60) | REPLACE | keys: ReadGilmore2005* | source: iam_missing_satellites.tex:98 (source key Read2005 has garbled metadata; Read & Gilmore 2005 MNRAS 356, 107 is the paper on feedback-driven core formation)

anchor:
```latex
density~\cite{PontzenGovernato2012}. These act on baryons; they do not address whether the dark-matter subhalo population itself is suppressed.
```
replacement:
```latex
density~\cite{ReadGilmore2005,PontzenGovernato2012}. These act on baryons; they do not address whether the dark-matter subhalo population itself is suppressed.
```

**B27** `docs/book/part2/p2_22_electroweak.tex` (line 21 at 3290e60) | REPLACE | keys: Clausius1870 | source: iam_electroweak_matter_sector.tex:93

anchor:
```latex
\derived{} For the $1/r$ potentials of gravity and electrostatics, $k=-1$ and $2\langle K\rangle+\langle V\rangle=0$. This holds for atoms and
```
replacement:
```latex
\derived{} For the $1/r$ potentials of gravity and electrostatics, $k=-1$ and $2\langle K\rangle+\langle V\rangle=0$~\cite{Clausius1870}. This holds for atoms and
```

**B28** `docs/book/part5/p5_01b_bh_information.tex` (line 175 at 3290e60) | REPLACE | keys: Hawking1975 | source: IAM_BH_Thermodynamics_Paper.tex:269

anchor:
```latex
Hawking's original calculation suggested that information falling into a black hole is irretrievably lost upon evaporation, violating unitarity. The paradox
```
replacement:
```latex
Hawking's original calculation~\cite{Hawking1975} suggested that information falling into a black hole is irretrievably lost upon evaporation, violating unitarity. The paradox
```

**B29** `docs/book/part5/p5_04_measurement.tex` (line 189 at 3290e60) | REPLACE | keys: Bell1964, Aspect1982, Hensen2015 | source: iam_entanglement_decoherence.tex:34

anchor:
```latex
Entangled particles show correlations that violate Bell inequalities. IAM's Law is not a local hidden-variable theory, and it changes nothing in quantum
```
replacement:
```latex
Entangled particles show correlations that violate Bell inequalities~\cite{Bell1964,Aspect1982,Hensen2015}. IAM's Law is not a local hidden-variable theory, and it changes nothing in quantum
```

**B30** `docs/book/part5/p5_05b_virial_partners.tex` (line 35 at 3290e60) | REPLACE | keys: Clausius1870 | source: IAM_Virial_37_Orders_Paper.tex:108

anchor:
```latex
theorem. The virial theorem states that for any system bound by a $1/r$ potential --- gravitational or Coulomb --- the time-averaged kinetic and potential
```
replacement:
```latex
theorem. The virial theorem~\cite{Clausius1870} states that for any system bound by a $1/r$ potential --- gravitational or Coulomb --- the time-averaged kinetic and potential
```

**B31** `docs/book/part5/p5_05c_virial_decoherence.tex` (line 73 at 3290e60) | REPLACE | keys: Landauer1961 | source: iam_two_faces_of_time.tex:130

anchor:
```latex
Landauer cost $k_BT_H\ln2$ per bit, where $T_H=\hbar H/2\pi k_B$ is the Gibbons--Hawking temperature~\cite{GibbonsHawking1977} ($2.65\times10^{-30}$\,K
```
replacement:
```latex
Landauer cost~\cite{Landauer1961} $k_BT_H\ln2$ per bit, where $T_H=\hbar H/2\pi k_B$ is the Gibbons--Hawking temperature~\cite{GibbonsHawking1977} ($2.65\times10^{-30}$\,K
```

**B32** `docs/book/part1/p1_02_iams_law.tex` (line 601 at 3290e60) | REPLACE | keys: Einstein1915 | source: iam_theory_paper.tex:1846; Holographic_Derivation_of_IAM.tex:376 (same list carried in Ch. iams_law)

anchor:
```latex
\item Gravitational interaction. Matter interacts gravitationally, initiating collapse of overdense regions (general relativity).
```
replacement:
```latex
\item Gravitational interaction. Matter interacts gravitationally, initiating collapse of overdense regions (general relativity~\cite{Einstein1915}).
```

Bib entries for the new keys: Peebles1980 (10.1515/9780691206714), CarrollChen2004 (arXiv hep-th/0410270, 10.48550/arXiv.hep-th/0410270), ReadGilmore2005 (10.1111/j.1365-2966.2004.08424.x), Bestor2000 (10.1093/hmg/9.16.2395), Jurkowska2011 (10.1002/cbic.201000195) -- all in docs/book/bib_additions.bib, DOI-checked.

Not added (decisions): Adil et al. 2023 at p2_08 and the two O Colgain analyses at p2_09b -- those chapters' naming rule cites them by journal and arXiv number in the text; Frusciante2025 at the p2_03 Euclid prediction -- left to the sec:lt_euclid rule; self-citations -- stand-alone rule.

### 2.2 (b) iam.bib entries never cited (39 at 3290e60, with the new glossary and the 32 blocks applied) -- list only, nothing deleted

Andersson2026, BelleII2023, Dirac1937, Klimov2018, Koski2014, Planck2015MG, Schlor2019, Simon2011Segue, Simon2019, Theis2017, Wang2014, deGraaf2020, ForemanMackey2013, Snyder2016, Cristiano2019, Roadmap2015, Lister2011, Jung2015, Lehne2015, CLSIEP28, Everett1996, Kirby2013Segue2, CaldwellBigRip2003, Ishak2025, Brannen2010, GoogleWillowSpec2024, Jenkins2001, LaceyCole1993, Reed2003, Watson2013, Burnett2014, Diamond2022PRXQ, Lenander2011, Bal2024, Catelani2011PRB, Greytak1964, Yelton2024, Kittel2005, Nadler2020

Of these, 10 became uncited when p3_02_xqp was restored (06aabc5): Wang2014, deGraaf2020, Burnett2014, Diamond2022PRXQ, Lenander2011, Bal2024, Catelani2011PRB, Greytak1964, Yelton2024, Kittel2005. Andersson2026 and CaldwellBigRip2003 are also duplicates (2.4).

### 2.3 (c) iam.bib entries lacking a DOI

145 entries lacked a DOI. 126 now have one in corrected_entries.bib (CrossRef-validated; 13 arXiv-only entries get their DataCite arXiv DOI 10.48550/arXiv.<id>, each arXiv id checked against the title on the arXiv API). Search hits rejected as not the cited work (book reviews, reissues): Goldstein, Smolin1997, Smolin2013; Roadmap2015 got its own DOI 10.1038/nature14248 by direct lookup. No DOI found for: AMD9950X, Abragam1970, Brannen2006, CLSIEP28, Carroll2010, Chandrasekhar1939, ESA2026DR1, Goldstein, GoogleWillowSpec2024, Hoffman2014 (JMLR has none), Kittel2005, Koide1982, Koomey2016, Nelson2017, Sakharov1967 (the 1991 Sov. Phys. Usp. reprint has 10.1070/PU1991v034n05ABEH002497 if a reprint DOI is wanted), Smolin1997, Smolin2013, Thorne1972, Zwicky1933.

Existing DOIs (354 entries; 353 distinct keys): 350 validated automatically; Gillessen2009 failed once on the network and validated on retry; WheelerItFromBit and Riggs1975 failed the year test only (CrossRef has no year for the chapter / gives the 2008 online date for Riggs) while title (exact) and pages (309-336, 9-25) match -- confirmed by hand. The 4 entries added at 3290e60 (Bondi1957, Olum1998, BartlettVanBuren1986, Singh2023) validate.

### 2.4 (d) Duplicates

- Duplicate KEY: `Reyes2010` is defined twice in iam.bib (full entry with eprint, first occurrence; a shorter one later in the file). BibTeX keeps the first and warns; delete the second.
- Same DOI under two keys: DESY3 = AbbottDESY3_2022 (10.1103/PhysRevD.105.023520); BelleII2023 = BelleII2023tau (10.1103/PhysRevD.108.032006); CaldwellBigRip2003 = Caldwell2003 (10.1103/PhysRevLett.91.071301); Clementi1974 = ClementiRoetti1974 (10.1016/S0092-640X(74)80016-1).
- Same arXiv paper: Andersson2026 = Sundelin2026 (arXiv 2602.01945; the arXiv first author is S. Sundelin, so Andersson2026 also carries the wrong first author; Andersson2026 is uncited).
- Same paper, two printings: Maldacena1998 (Adv. Theor. Math. Phys.) and Maldacena1999 (Int. J. Theor. Phys. reprint).
- PDG2022 and PDG2024 are different editions (not duplicates).

### 2.5 Source-paper DOIs that failed validation (bibliographic only; repo record; provisional)

These DOIs printed in the source papers did not match the cited work on CrossRef (another article, or not registered). Two look like validator false alarms (de Mattia 2021 and Gompertz 1825: titles match). None affects the book unless the book copies the DOI.

| source reference (start) | printed DOI | CrossRef title for it | better match found | source file |
|---|---|---|---|---|
| de Mattia A., et al., 2021, MNRAS, 501, 5616. 10.1093/mnras/staa3891 | 10.1093/mnras/staa3891 | The Completed SDSS-IV extended Baryon Oscillation  | -- | iam_desi_paper.tex |
| Freedman W. L., et al., 2024, ApJ, 919, 16. 10.3847/1538-4357/ac7c74 | 10.3847/1538-4357/ac7c74 | The Astropy Project: Sustaining and Growing a Comm | -- | iam_desi_paper.tex |
| O Colgain E., Sheikh-Jabbari M. M., 2025, MNRAS Lett., 542, L24. 10.10 | 10.1093/mnrasl/slaf055 | Interacting dark energy constraints from the full- | 10.1093/mnrasl/slaf042 | iam_desi_paper.tex |
| Wang Z., et al., 2023, JCAP, 2023, 022. 10.1088/1475-7516/2023/10/022 | 10.1088/1475-7516/2023/10/022 | JUNO sensitivity to                     <sup>7</su | -- | iam_desi_paper.tex |
| Adelman, E.R., Huang, H.T., Roisman, A., et al. (2019). Aging human he | 10.1016/j.stem.2019.06.012 | Context-Specific Transcription Factor Functions Re | -- | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Ahrens, M., Ammerpohl, O., von Schonfels, W., et al. (2013). DNA methy | 10.1038/ncomms3617 | Reconstructing targetable pathways in lung cancer  | 10.1016/j.cmet.2013.07.004 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Cancer Genome Atlas Research Network. (2012). Comprehensive genomic ch | 10.1038/nature11385 | None | 10.1038/nature11404 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Cancer Genome Atlas Research Network. (2017). Genomic classification o | 10.1016/j.ccell.2017.10.016 | Glut3 Addiction Is a Druggable Vulnerability for a | -- | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Cancer Genome Atlas Research Network. (2018). Comprehensive molecular  | 10.1038/s41588-018-0103-7 | None | 10.1093/med/9780198813033.003.0022 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Cancer Genome Atlas Research Network. (2018). Genomic and molecular la | 10.1016/j.ccell.2018.03.010 | Comparative Molecular Analysis of Gastrointestinal | 10.3410/f.732976037.793546302 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Fleischer, T., Frigessi, A., Johnson, K.C., et al. (2017). Genome-wide | 10.1186/s13059-016-1163-8 | None | -- | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Gompertz, B. (1825). On the nature of the function expressive of the l | 10.1098/rstl.1825.0026 | XXIV. On the nature of the function expressive of  | -- | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Hata, M., Bhatt, D.L., and Bhatt, D.L. (2020). DNA methylation dynamic | 10.1038/s41588-020-0589-1 | None | -- | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Horvath, S., Haghani, A., Zoller, J.A., et al. (2022). Epigenetic cloc | 10.1126/science.abn4689 | None | 10.1101/2021.03.30.437604 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Kozlenkov, A., Roussos, P., Timashpolsky, A., et al. (2014). Differenc | 10.1093/hmg/ddu196 | The genetic contributions of SNCA and LRRK2 genes  | 10.1093/nar/gkt838 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Schulze, K., Imbeaud, S., Letoure, E., et al. (2015). Exome sequencing | 10.1038/ng.3264 | The two sides of GIGANTEA | 10.1038/ng.3252 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex |
| Bal, M., et al. (2024). Atomic-scale characterization of the tantalum  | 10.1021/acsnano.4c05251 | Structure and Formation Mechanisms in Tantalum and | -- | Biological_Physics_MethylPhys_papers_IAM_Hubble2Methyl_Alpha_Omega_5.tex |
| Cai, R.-G. and Kim, S. P. First law of thermodynamics and Friedmann eq | 10.1088/1475-7516/2005/02/050 | None | 10.1088/1126-6708/2005/02/050 | iam_decoherence_virial_partition.tex, iam_missing_satellites.tex |
| Rovelli, C. Memory and entropy Entropy 24 1394 2022 | 10.3390/e24101394 | A Fault Detection Method Based on an Oil Temperatu | 10.3390/e24081022 | iam_decoherence_virial_partition.tex |
| Euclid Collaboration Euclid preparation: forecasts for modified gravit | 10.1051/0004-6361/202347045 | None | -- | iam_missing_satellites.tex |
| Read, J. I. and Pontzen, A. and Walker, M. and Steger, P. Dark matter  | 10.1111/j.1365-2966.2005.09956.x | Laser Interferometer Space Antenna double black ho | 10.1111/j.1365-2966.2006.10720.x | iam_missing_satellites.tex |
| Wang, C. and others Clocks of aging in dogs and humans Cell Reports 33 | 10.1016/j.celrep.2020.108273 | Integrated Single-Cell Transcriptomics and Chromat | -- | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_iam_vertebrate_lifespan.tex |
| Lyko, F. The DNA methyltransferase family: a versatile toolkit for epi | 10.1038/nrg.2017.81 | Identifying global RNA–chromatin interactions by G | 10.1038/nrg.2017.80 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_iam_vertebrate_lifespan.tex |
| Bertucci, E. M. and others Epigenetic aging and exposure to ionizing r | 10.18632/aging.203731 | Metabolomic profiling of plasma from middle-aged a | 10.18632/aging.203624 | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_iam_vertebrate_lifespan.tex |
| Bashtrykov, P. and others Elevated expression of DNMT3A in leukemia me | 10.1038/leu.2014.101 | Nicole Muller-Bérat Killmann 1932–2014 Leukemia pi | -- | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_iam_vertebrate_lifespan.tex |
| Bhanu, N. V. and others Quantitative and site-specific remodeling of p | 10.1021/acs.jproteome.8b00243 | None | -- | Biological_Physics_RETIRED_2026-10_MethylPhys_papers_iam_vertebrate_lifespan.tex |
| Horvath, S. and Haghani, A. and Zoller, J. A. and Lu, A. T. and Raj, K | 10.1126/science.abn4689 | None | 10.1101/2021.03.30.437604 | Biological_Physics_MethylPhys_papers_IAM_for_physicists_IAM_for_physicists.tex |

### 2.6 Item-by-item carriage table (source citation -> book), 494 contexts with overlap >= 0.25

| source file:line | source key | book file:line (best match) | status |
|---|---|---|---|
| Gravitational_Decoherence.tex:141 | Mahaffey2026repo | - | EXCLUDED: self-citation (stand-alone rule) |
| Gravitational_Decoherence.tex:162 | Mahaffey2026repo | - | EXCLUDED: self-citation (stand-alone rule) |
| Gravitational_Decoherence.tex:178 | Penrose1996 | part5/p5_05_gravdec.tex:56 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.29) |
| Gravitational_Decoherence.tex:178 | Diosi1987 | part5/p5_05_gravdec.tex:56 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.29) |
| Gravitational_Decoherence.tex:185 | Landauer1961 | part5/p5_05_gravdec.tex:61 | already cited at this place |
| iam_cosmological_constant.tex:51 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_cosmological_constant.tex:113 | desi2025 | part2/p2_12b_lambda_history.tex:145 | already cited at this place (DESI2025) |
| iam_cosmological_constant.tex:134 | weinberg1989 | part2/p2_12_lambda.tex:31 | already cited at this place (Weinberg1989) |
| iam_cosmological_constant.tex:134 | martin2012 | part2/p2_12_lambda.tex:31 | already cited at this place (Martin2012) |
| iam_cosmological_constant.tex:146 | riess1998 | part2/p2_12_lambda.tex:37 | already cited at this place (Riess1998) |
| iam_cosmological_constant.tex:146 | perlmutter1999 | part2/p2_12_lambda.tex:37 | already cited at this place (Perlmutter1999) |
| iam_cosmological_constant.tex:164 | witten2000 | part2/p2_12_lambda.tex:48 | cited at this place as Witten2001 (published version of hep-ph/0002297) |
| iam_cosmological_constant.tex:165 | bousso2000 | part2/p2_12_lambda.tex:48 | already cited at this place (BoussoPolchinski2000) |
| iam_cosmological_constant.tex:166 | caldwell1998 | part2/p2_12_lambda.tex:48 | already cited at this place (Caldwell1998) |
| iam_cosmological_constant.tex:167 | weinberg1989 | part2/p2_12_lambda.tex:49 | already cited at this place (Weinberg1989) |
| iam_cosmological_constant.tex:180 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_cosmological_constant.tex:194 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_cosmological_constant.tex:196 | jacobson1995 | part2/p2_12_lambda.tex:80 | already cited at this place (Jacobson1995) |
| iam_cosmological_constant.tex:205 | mahaffey2026_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_cosmological_constant.tex:207 | landauer1961 | part2/p2_03_theory.tex:121 | already cited at this place (Landauer1961) |
| iam_cosmological_constant.tex:210 | gibbons1977 | part2/p2_12_lambda.tex:87 | already cited at this place (GibbonsHawking1977) |
| iam_cosmological_constant.tex:224 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_cosmological_constant.tex:245 | mahaffey2026_higgs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_cosmological_constant.tex:750 | desi2025 | part2/p2_12b_lambda_history.tex:34 | already cited at this place (DESI2025) |
| iam_cosmological_constant.tex:760 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_cosmological_constant.tex:810 | weinberg1989 | part2/p2_12b_lambda_history.tex:105 | already cited at this place (Weinberg1989) |
| iam_cosmological_constant.tex:813 | witten2000 | part2/p2_12b_lambda_history.tex:105 | cited at this place as Witten2001 (published version of hep-ph/0002297) |
| iam_cosmological_constant.tex:813 | bousso2000 | part2/p2_12b_lambda_history.tex:106 | already cited at this place (BoussoPolchinski2000) |
| iam_cosmological_constant.tex:815 | weinberg1989 | part2/p2_12b_lambda_history.tex:107 | already cited at this place (Weinberg1989) |
| iam_cosmological_constant.tex:827 | verlinde2011 | part2/p2_12b_lambda_history.tex:115 | already cited at this place (Verlinde2011) |
| iam_cosmological_constant.tex:828 | padmanabhan2010 | part2/p2_12b_lambda_history.tex:115 | already cited at this place (Padmanabhan2010) |
| iam_cosmological_constant.tex:928 | jacobson1995 | part1/p1_04_virial_identity.tex:147 | ACCEPTED -> insertion block B01 |
| Biological_Physics_MethylPhys_papers_Landauer_Metrology_of_the_Methylome.tex:37 | sanchez2016 | part4/p4_02_landauer.tex:203 | already cited at this place (Sanchez2016) |
| Biological_Physics_MethylPhys_papers_Landauer_Metrology_of_the_Methylome.tex:42 | sanchez2019 | part4/p4_02_landauer.tex:220 | already cited at this place (Sanchez2019) |
| Biological_Physics_MethylPhys_papers_Landauer_Metrology_of_the_Methylome.tex:46 | sanchez2016 | part4/p4_02_landauer.tex:230 | Sanchez2016 cited in the same paragraph of p4_02 |
| iam_baryon_asymmetry.tex:98 | cyburt2016 | part2/p2_13_baryon.tex:32 | already cited at this place (Cyburt2016) |
| iam_baryon_asymmetry.tex:99 | planck2020 | part2/p2_13_baryon.tex:32 | already cited at this place (Planck2018VI) |
| iam_baryon_asymmetry.tex:104 | sakharov1967 | part2/p2_13_baryon.tex:39 | already cited at this place (Sakharov1967) |
| iam_baryon_asymmetry.tex:124 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:128 | landauer1961 | part2/p2_13_baryon.tex:52 | already cited at this place (Landauer1961) |
| iam_baryon_asymmetry.tex:130 | mahaffey2026_cc | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:140 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:142 | jacobson1995 | part2/p2_12_lambda.tex:80 | already cited at this place (Jacobson1995) |
| iam_baryon_asymmetry.tex:150 | mahaffey2026_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:152 | landauer1961 | part2/p2_12_lambda.tex:87 | already cited at this place (Landauer1961) |
| iam_baryon_asymmetry.tex:170 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:174 | mahaffey2026_virial_dm | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:197 | mahaffey2026_cc | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:254 | mahaffey2026_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:266 | mahaffey2026_virial_dm | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:359 | mahaffey2026_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:479 | planck2020 | part2/p2_13b_baryon_chain.tex:150 | already cited at this place (Planck2018VI) |
| iam_baryon_asymmetry.tex:483 | mahaffey2026_cc | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:514 | rubakov1996 | part2/p2_13_baryon.tex:179 | cited at this place as RubakovShaposhnikov1996 (English edition of the same review) |
| iam_baryon_asymmetry.tex:559 | mahaffey2026_virial_dm | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:570 | mahaffey2026_cc | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:571 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:587 | mahaffey2026_cc | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:588 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_baryon_asymmetry.tex:611 | jacobson1995 | part1/p1_04_virial_identity.tex:147 | ACCEPTED -> insertion block B01 |
| arXiv_Master_file.tex:186 | Jacobson1995 | part2/p2_03_theory.tex:48 | already cited at this place |
| arXiv_Master_file.tex:190 | CaiKim2005 | part2/p2_03_theory.tex:219 | already cited at this place |
| arXiv_Master_file.tex:252 | Peebles1980 | part1/p1_02_iams_law.tex:601 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.36) |
| arXiv_Master_file.tex:256 | Zurek2003 | part2/p2_03_theory.tex:901 | already cited at this place |
| arXiv_Master_file.tex:265 | Landauer1961 | part2/p2_03_theory.tex:904 | already cited at this place |
| arXiv_Master_file.tex:270 | tHooft1993 | part2/p2_03_theory.tex:80 | already cited at this place |
| arXiv_Master_file.tex:270 | Susskind1995 | part2/p2_03_theory.tex:80 | already cited at this place |
| arXiv_Master_file.tex:385 | Bekenstein1973 | part2/p2_03_theory.tex:146 | ACCEPTED -> insertion block B15 |
| arXiv_Master_file.tex:433 | CaiKim2005 | part2/p2_03_theory.tex:219 | already cited at this place |
| arXiv_Master_file.tex:440 | GibbonsHawking1977 | part2/p2_03_theory.tex:222 | ACCEPTED -> insertion block B16 |
| arXiv_Master_file.tex:591 | Landauer1961 | part2/p2_01_blackholes.tex:59 | ACCEPTED -> insertion block B09 |
| arXiv_Master_file.tex:596 | Hawking1975 | part2/p2_01_blackholes.tex:64 | ACCEPTED -> insertion block B10 |
| arXiv_Master_file.tex:619 | Mahaffey2026bh | - | EXCLUDED: self-citation (stand-alone rule) |
| arXiv_Master_file.tex:695 | Thorne1972 | part2/p2_01_blackholes.tex:268 | already cited at this place |
| arXiv_Master_file.tex:1145 | Pogosian2016 | part2/p2_03_theory.tex:479 | already cited at this place (PogosianSilvestri2016) |
| arXiv_Master_file.tex:1835 | Verlinde2011 | part2/p2_03_theory.tex:921 | already cited at this place |
| arXiv_Master_file.tex:1836 | Padmanabhan2010 | part2/p2_03_theory.tex:921 | already cited at this place |
| arXiv_Master_file.tex:1849 | DESI2025 | part2/p2_03_theory.tex:873 | already cited at this place (DESI2024VI) |
| arXiv_Master_file.tex:1884 | Frusciante2025 | part2/p2_03_theory.tex:1056 | Euclid sentence at p2_03:1053 left to the sec:lt_euclid rule (lead decision) |
| arXiv_Master_file.tex:1983 | Jacobson1995 | part2/p2_03_theory.tex:995 | ACCEPTED -> insertion block B22 |
| arXiv_Master_file.tex:1988 | CaiKim2005 | part2/p2_03_theory.tex:997 | ACCEPTED -> insertion block B23 |
| arXiv_Master_file.tex:2004 | Bekenstein1973 | part2/p2_03_theory.tex:1004 | already cited at this place |
| arXiv_Master_file.tex:2005 | GibbonsHawking1977 | part2/p2_03_theory.tex:1004 | already cited at this place |
| arXiv_Master_file.tex:2006 | Unruh1976 | part2/p2_03_theory.tex:1004 | already cited at this place |
| arXiv_Master_file.tex:2007 | Jacobson1995 | part2/p2_03_theory.tex:1005 | already cited at this place |
| arXiv_Master_file.tex:2008 | CaiKim2005 | part2/p2_03_theory.tex:1006 | already cited at this place |
| arXiv_Master_file.tex:2009 | Landauer1961 | part2/p2_03_theory.tex:1006 | already cited at this place |
| arXiv_Master_file.tex:2010 | Zurek2003 | part2/p2_03_theory.tex:1006 | already cited at this place |
| arXiv_Master_file.tex:2071 | Jacobson1995 | part2/p2_03_theory.tex:1019 | already cited at this place |
| arXiv_Master_file.tex:2072 | Verlinde2011 | part2/p2_03_theory.tex:1019 | already cited at this place |
| arXiv_Master_file.tex:2073 | Padmanabhan2010 | part2/p2_03_theory.tex:1019 | already cited at this place |
| arXiv_Master_file.tex:2079 | Pogosian2016 | part2/p2_03_theory.tex:1023 | already cited at this place (PogosianSilvestri2016) |
| arXiv_Master_file.tex:2079 | Andrade2024 | part2/p2_03_theory.tex:1023 | already cited at this place |
| arXiv_Master_file.tex:2079 | DESI2025 | part2/p2_03_theory.tex:1023 | book cites DESI2025 / DESI2024VII at this place |
| Holographic_Derivation_of_IAM.tex:102 | Gibbons1977 | part5/p5_05b_virial_partners.tex:79 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.27) |
| Holographic_Derivation_of_IAM.tex:108 | Landauer1961 | part2/p2_03_theory.tex:242 | ACCEPTED -> insertion block B17 |
| Holographic_Derivation_of_IAM.tex:118 | Cai2005 | part2/p2_09_sector_tension.tex:90 | already cited at this place (CaiKim2005) |
| Holographic_Derivation_of_IAM.tex:337 | Sheth1999 | part2/p2_03_theory.tex:715 | already cited at this place (ShethTormen1999) |
| Holographic_Derivation_of_IAM.tex:376 | Einstein1915 | part1/p1_02_iams_law.tex:601 | ACCEPTED -> insertion block B32 |
| Holographic_Derivation_of_IAM.tex:377 | Peebles1980 | part2/p2_03_theory.tex:900 | ACCEPTED -> insertion block B20 |
| Holographic_Derivation_of_IAM.tex:378 | Zurek2003 | part2/p2_03_theory.tex:901 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:380 | Landauer1961 | part2/p2_03_theory.tex:904 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:381 | Bekenstein1973 | part2/p2_03_theory.tex:905 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:381 | tHooft1993 | part2/p2_03_theory.tex:905 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:381 | Susskind1995 | part2/p2_03_theory.tex:905 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:382 | Jacobson1995 | part2/p2_03_theory.tex:906 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:383 | Mahaffey2026main | - | EXCLUDED: self-citation (stand-alone rule) |
| Holographic_Derivation_of_IAM.tex:393 | Verlinde2011 | part2/p2_03_theory.tex:921 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:393 | Padmanabhan2010 | part2/p2_03_theory.tex:921 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:401 | Penrose1989 | part2/p2_03_theory.tex:974 | already cited at this place |
| Holographic_Derivation_of_IAM.tex:401 | Carroll2004 | part2/p2_03_theory.tex:974 | ACCEPTED -> insertion block B21 |
| Holographic_Derivation_of_IAM.tex:407 | Penrose1989 | part2/p2_03_theory.tex:986 | already cited at this place |
| zurek_paper.tex:103 | Zurek1981 | part2/p2_14_quantum_records.tex:25 | already cited at this place |
| zurek_paper.tex:107 | Zurek2009 | part2/p2_14_quantum_records.tex:26 | already cited at this place |
| zurek_paper.tex:124 | Landauer1961 | part2/p2_14_quantum_records.tex:80 | already cited at this place |
| zurek_paper.tex:128 | Jacobson1995 | part2/p2_03_theory.tex:635 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.31) |
| zurek_paper.tex:128 | CaiKim2005 | part2/p2_03_theory.tex:635 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.31) |
| zurek_paper.tex:138 | IAM_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| zurek_paper.tex:171 | Zurek1981 | part2/p2_14_quantum_records.tex:38 | already cited at this place |
| zurek_paper.tex:260 | Landauer1961 | part2/p2_02_virial.tex:58 | ACCEPTED -> insertion block B12 |
| zurek_paper.tex:265 | Gibbons1977 | part5/p5_05c_virial_decoherence.tex:73 | already cited at this place (GibbonsHawking1977) |
| zurek_paper.tex:274 | IAM_theory | - | EXCLUDED: self-citation (stand-alone rule) |
| zurek_paper.tex:608 | IAM_bh | - | EXCLUDED: self-citation (stand-alone rule) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:86 | PlanckCollaboration2020b | part2/p2_07_late_time_growth.tex:30 | already cited at this place (Planck2018VI) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:92 | PlanckCollaboration2020b | part2/p2_06_dual_sector_perturbation.tex:35 | already cited at this place (Planck2018VI) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:95 | Riess2022 | part2/p2_06_dual_sector_perturbation.tex:36 | already cited at this place |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:97 | Freedman2021 | part2/p2_06_dual_sector_perturbation.tex:38 | already cited at this place |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:98 | Wong2020 | part2/p2_06_dual_sector_perturbation.tex:38 | already cited at this place |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:113 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:186 | PlanckCollaboration2020b | part2/p2_06_dual_sector_perturbation.tex:88 | already cited at this place (Planck2018VI) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:189 | Mahaffey2026theory | - | EXCLUDED: self-citation (stand-alone rule) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:292 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:309 | Mahaffey2026theory | - | EXCLUDED: self-citation (stand-alone rule) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:326 | Efstathiou2021 | part2/p2_06_dual_sector_perturbation.tex:213 | book cites Rosenberg2022 (NPIPE CamSpec actually used) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:329 | PlanckCollaboration2020a | part2/p2_06_dual_sector_perturbation.tex:213 | already cited at this place (Planck2018VIII) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:333 | Alam2017 | part2/p2_06_dual_sector_perturbation.tex:217 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.27) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:333 | Alam2021 | part2/p2_06_dual_sector_perturbation.tex:217 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.27) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:816 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.tex:840 | Abbott2017siren | part2/p2_06_dual_sector_perturbation.tex:484 | already cited at this place (Abbott2017Siren) |
| IAM_Virial_Efficiency_neff_Paper.tex:132 | Tinker2008 | part2/p2_02_virial.tex:73 | ACCEPTED -> insertion block B13 |
| IAM_ThreeWay_Cluster_Paper.tex:125 | Mahaffey2026a | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_ThreeWay_Cluster_Paper.tex:232 | Nelson2014 | appendices/app_G_predictions_register.tex:48 | book line is the predictions register table (app_G), no citations by design |
| IAM_ThreeWay_Cluster_Paper.tex:232 | Shi2014 | appendices/app_G_predictions_register.tex:48 | book line is the predictions register table (app_G), no citations by design |
| IAM_ThreeWay_Cluster_Paper.tex:309 | Neto2007 | part2/p2_18_three_way_clusters.tex:211 | already cited at this place |
| IAM_ThreeWay_Cluster_Paper.tex:309 | Planelles2017 | part2/p2_18_three_way_clusters.tex:211 | already cited at this place |
| IAM_ThreeWay_Cluster_Paper.tex:313 | MacCrann2022 | part2/p2_18_three_way_clusters.tex:212 | already cited at this place |
| iam_theory_paper.tex:61 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_theory_paper.tex:86 | Penrose1989 | part2/p2_03_theory.tex:43 | already cited at this place |
| iam_theory_paper.tex:88 | Zurek2003 | part2/p2_03_theory.tex:45 | already cited at this place |
| iam_theory_paper.tex:88 | Schlosshauer2007 | part2/p2_03_theory.tex:45 | already cited at this place |
| iam_theory_paper.tex:92 | Landauer1961 | part2/p2_03_theory.tex:45 | already cited at this place |
| iam_theory_paper.tex:94 | Jacobson1995 | part2/p2_03_theory.tex:48 | already cited at this place |
| iam_theory_paper.tex:97 | CaiKim2005 | part2/p2_03_theory.tex:49 | already cited at this place |
| iam_theory_paper.tex:107 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_theory_paper.tex:183 | Zurek2003 | part2/p2_03_theory.tex:109 | already cited at this place |
| iam_theory_paper.tex:206 | Landauer1961 | part2/p2_03_theory.tex:121 | already cited at this place |
| iam_theory_paper.tex:227 | Jacobson1995 | part2/p2_03_theory.tex:130 | already cited at this place |
| iam_theory_paper.tex:227 | CaiKim2005 | part2/p2_03_theory.tex:130 | already cited at this place |
| iam_theory_paper.tex:362 | Wald1984 | part1/p1_02_iams_law.tex:253 | ACCEPTED -> insertion block B05 |
| iam_theory_paper.tex:409 | CaiKim2005 | part2/p2_03_theory.tex:219 | already cited at this place |
| iam_theory_paper.tex:472 | Jacobson1995 | part2/p2_03_theory.tex:249 | already cited at this place |
| iam_theory_paper.tex:889 | PogosianSilvestri2016 | part2/p2_03_theory.tex:479 | already cited at this place |
| iam_theory_paper.tex:1103 | PlanckCollaboration2020 | part2/p2_03_theory.tex:603 | already cited at this place (Planck2018VI) |
| iam_theory_paper.tex:1109 | ShethTormen1999 | part2/p2_03_theory.tex:608 | already cited at this place |
| iam_theory_paper.tex:1110 | Tinker2008 | part2/p2_03_theory.tex:608 | already cited at this place |
| iam_theory_paper.tex:1112 | EisensteinHu1998 | part2/p2_03_theory.tex:609 | already cited at this place |
| iam_theory_paper.tex:1347 | ShethTormen1999 | part2/p2_03_theory.tex:715 | already cited at this place |
| iam_theory_paper.tex:1609 | Bernardeau2002 | part2/p2_03_theory.tex:805 | already cited at this place |
| iam_theory_paper.tex:1635 | Bernardeau2002 | part2/p2_03_theory.tex:813 | already cited at this place |
| iam_theory_paper.tex:1689 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_theory_paper.tex:1712 | Heymans2021 | part2/p2_03_theory.tex:851 | already cited at this place |
| iam_theory_paper.tex:1712 | Abbott2022 | part2/p2_03_theory.tex:851 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.36) |
| iam_theory_paper.tex:1772 | DESIDR1 | part2/p2_03_theory.tex:873 | book cites DESI2025 / DESI2024VII at this place (source bibitem has mismatched title and volume) |
| iam_theory_paper.tex:1846 | Einstein1915 | part1/p1_02_iams_law.tex:601 | ACCEPTED -> insertion block B32 |
| iam_theory_paper.tex:1848 | Peebles1980 | part2/p2_03_theory.tex:900 | ACCEPTED -> insertion block B20 |
| iam_theory_paper.tex:1851 | Zurek2003 | part2/p2_03_theory.tex:901 | already cited at this place |
| iam_theory_paper.tex:1857 | Landauer1961 | part2/p2_03_theory.tex:904 | already cited at this place |
| iam_theory_paper.tex:1860 | Bekenstein1973 | part2/p2_03_theory.tex:905 | already cited at this place |
| iam_theory_paper.tex:1860 | tHooft1993 | part2/p2_03_theory.tex:905 | already cited at this place |
| iam_theory_paper.tex:1860 | Susskind1995 | part2/p2_03_theory.tex:905 | already cited at this place |
| iam_theory_paper.tex:1863 | Jacobson1995 | part2/p2_03_theory.tex:906 | already cited at this place |
| iam_theory_paper.tex:1883 | Verlinde2011 | part2/p2_03_theory.tex:921 | already cited at this place |
| iam_theory_paper.tex:1884 | Padmanabhan2010 | part2/p2_03_theory.tex:921 | already cited at this place |
| iam_theory_paper.tex:1917 | Thorne1972 | part2/p2_03_theory.tex:938 | already cited at this place |
| iam_theory_paper.tex:1986 | Penrose1989 | part2/p2_03_theory.tex:974 | already cited at this place |
| iam_theory_paper.tex:1986 | Carroll2004 | part2/p2_03_theory.tex:974 | ACCEPTED -> insertion block B21 |
| iam_theory_paper.tex:2004 | Penrose1989 | part2/p2_03_theory.tex:986 | already cited at this place |
| iam_theory_paper.tex:2041 | Bekenstein1973 | part2/p2_03_theory.tex:1003 | already cited at this place |
| iam_theory_paper.tex:2041 | Hawking1975 | part2/p2_03_theory.tex:1003 | already cited at this place |
| iam_theory_paper.tex:2042 | GibbonsHawking1977 | part2/p2_03_theory.tex:1004 | already cited at this place |
| iam_theory_paper.tex:2043 | Unruh1976 | part2/p2_03_theory.tex:1004 | already cited at this place |
| iam_theory_paper.tex:2045 | Jacobson1995 | part2/p2_03_theory.tex:1005 | already cited at this place |
| iam_theory_paper.tex:2046 | CaiKim2005 | part2/p2_03_theory.tex:1005 | already cited at this place |
| iam_theory_paper.tex:2047 | Landauer1961 | part2/p2_03_theory.tex:1006 | already cited at this place |
| iam_theory_paper.tex:2048 | Zurek2003 | part2/p2_03_theory.tex:1006 | already cited at this place |
| iam_theory_paper.tex:2050 | Press1974 | part2/p2_03_theory.tex:1006 | already cited at this place (PressSchechter1974) |
| iam_theory_paper.tex:2050 | ShethTormen1999 | part2/p2_03_theory.tex:1006 | already cited at this place |
| iam_theory_paper.tex:2067 | Jacobson1995 | part2/p2_03_theory.tex:1019 | already cited at this place |
| iam_theory_paper.tex:2068 | Verlinde2011 | part2/p2_03_theory.tex:1019 | already cited at this place |
| iam_theory_paper.tex:2069 | Padmanabhan2010 | part2/p2_03_theory.tex:1019 | already cited at this place |
| iam_theory_paper.tex:2075 | PogosianSilvestri2016 | part2/p2_03_theory.tex:1023 | already cited at this place |
| iam_theory_paper.tex:2075 | Abbott2023 | part2/p2_03_theory.tex:1023 | ACCEPTED -> insertion block B24 (DESY3Ext) |
| iam_theory_paper.tex:2075 | Andrade2024 | part2/p2_03_theory.tex:1023 | already cited at this place |
| iam_theory_paper.tex:2075 | DESIDR1 | part2/p2_03_theory.tex:1023 | book cites DESI2025 / DESI2024VII at this place (source bibitem has mismatched title and volume) |
| iam_theory_paper.tex:2091 | tHooft1993 | part2/p2_03_theory.tex:1033 | already cited at this place |
| iam_theory_paper.tex:2091 | Susskind1995 | part2/p2_03_theory.tex:1033 | already cited at this place |
| iam_theory_paper.tex:2204 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| Variational_Derivation_of_IAM.tex:33 | Mahaffey2026holographic | - | EXCLUDED: self-citation (stand-alone rule) |
| Variational_Derivation_of_IAM.tex:299 | Bekenstein1973 | part2/p2_03_theory.tex:1004 | already cited at this place |
| Variational_Derivation_of_IAM.tex:299 | Hawking1975 | part2/p2_03_theory.tex:1004 | already cited at this place |
| Variational_Derivation_of_IAM.tex:300 | Gibbons1977 | part2/p2_03_theory.tex:1004 | already cited at this place (GibbonsHawking1977) |
| Variational_Derivation_of_IAM.tex:301 | Unruh1976 | part2/p2_03_theory.tex:1004 | already cited at this place |
| Variational_Derivation_of_IAM.tex:303 | Jacobson1995 | part2/p2_03_theory.tex:1005 | already cited at this place |
| Variational_Derivation_of_IAM.tex:304 | Cai2005 | part2/p2_03_theory.tex:1005 | already cited at this place (CaiKim2005) |
| Variational_Derivation_of_IAM.tex:305 | Landauer1961 | part2/p2_03_theory.tex:1005 | already cited at this place |
| Variational_Derivation_of_IAM.tex:306 | Zurek2003 | part2/p2_03_theory.tex:1006 | already cited at this place |
| Variational_Derivation_of_IAM.tex:307 | Press1974 | part2/p2_03_theory.tex:1006 | already cited at this place (PressSchechter1974) |
| Variational_Derivation_of_IAM.tex:307 | Sheth1999 | part2/p2_03_theory.tex:1006 | already cited at this place (ShethTormen1999) |
| Variational_Derivation_of_IAM.tex:508 | PlanckCollaboration2020 | part2/p2_03_theory.tex:603 | already cited at this place (Planck2018VI) |
| Variational_Derivation_of_IAM.tex:512 | Sheth1999 | part2/p2_03_theory.tex:609 | already cited at this place (ShethTormen1999) |
| Variational_Derivation_of_IAM.tex:512 | Tinker2008 | part2/p2_03_theory.tex:609 | already cited at this place |
| Variational_Derivation_of_IAM.tex:512 | EisensteinHu1998 | part2/p2_03_theory.tex:609 | already cited at this place |
| Variational_Derivation_of_IAM.tex:563 | Mahaffey2026holographic | - | EXCLUDED: self-citation (stand-alone rule) |
| Variational_Derivation_of_IAM.tex:687 | DESI2025DR2 | part2/p2_03_theory.tex:873 | already cited at this place (DESI2025) |
| Variational_Derivation_of_IAM.tex:766 | Sheth1999 | part2/p2_03_theory.tex:26 | clause "verified against N-body mass functions" not carried at p2_03:26 |
| Variational_Derivation_of_IAM.tex:766 | Tinker2008 | part2/p2_03_theory.tex:26 | clause not carried |
| iam_entanglement_decoherence.tex:34 | Bell1964 | part5/p5_04_measurement.tex:189 | ACCEPTED -> insertion block B29 |
| iam_entanglement_decoherence.tex:34 | Aspect1982 | part5/p5_04_measurement.tex:189 | ACCEPTED -> insertion block B29 |
| iam_entanglement_decoherence.tex:34 | Hensen2015 | part5/p5_04_measurement.tex:189 | ACCEPTED -> insertion block B29 |
| iam_entanglement_decoherence.tex:74 | Landauer1961 | part2/p2_14_quantum_records.tex:9 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.27) |
| iam_entanglement_decoherence.tex:91 | IAM_time | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_entanglement_decoherence.tex:104 | Jacobson1995 | appendices/app_E_formulas.tex:71 | book line is the formula sheet (app_E), attribution in text |
| iam_entanglement_decoherence.tex:107 | CaiKim2005 | part2/p2_01a_bekenstein.tex:416 | already cited at this place |
| iam_entanglement_decoherence.tex:132 | IAM_obs | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_BH_Thermodynamics_Paper.tex:52 | Bekenstein1973 | part2/p2_01_blackholes.tex:18 | already cited at this place |
| IAM_BH_Thermodynamics_Paper.tex:52 | Hawking1975 | part2/p2_01_blackholes.tex:18 | already cited at this place |
| IAM_BH_Thermodynamics_Paper.tex:55 | Jacobson1995 | part2/p2_01_blackholes.tex:20 | already cited at this place |
| IAM_BH_Thermodynamics_Paper.tex:57 | CaiKim2005 | part2/p2_01_blackholes.tex:21 | already cited at this place |
| IAM_BH_Thermodynamics_Paper.tex:60 | Mahaffey2026a | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_BH_Thermodynamics_Paper.tex:60 | Mahaffey2026b | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_BH_Thermodynamics_Paper.tex:87 | Hawking1975 | part2/p2_01_blackholes.tex:64 | ACCEPTED -> insertion block B10 |
| IAM_BH_Thermodynamics_Paper.tex:192 | GibbonsHawking1977 | part2/p2_01_blackholes.tex:205 | already cited at this place |
| IAM_BH_Thermodynamics_Paper.tex:269 | Hawking1975 | part5/p5_01b_bh_information.tex:175 | ACCEPTED -> insertion block B28 |
| IAM_BH_Thermodynamics_Paper.tex:286 | Page1993 | part5/p5_01b_bh_information.tex:184 | already cited at this place |
| IAM_BH_Thermodynamics_Paper.tex:307 | Thorne1972 | part2/p2_01_blackholes.tex:268 | already cited at this place |
| IAM_BH_Thermodynamics_Paper.tex:334 | Mahaffey2026c | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_BH_Thermodynamics_Paper.tex:343 | Carr1975 | part2/p2_01_blackholes.tex:288 | already cited at this place |
| iam_thermodynamic_identity.tex:115 | Clausius1870 | part1/p1_02_iams_law.tex:23 | already cited at this place |
| iam_thermodynamic_identity.tex:246 | Landauer1961 | part1/p1_04_virial_identity.tex:64 | already cited at this place |
| iam_thermodynamic_identity.tex:485 | IAM_obs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_thermodynamic_identity.tex:715 | Jacobson1995 | part1/p1_04_virial_identity.tex:143 | already cited at this place |
| iam_thermodynamic_identity.tex:770 | Jacobson1995 | part1/p1_04_virial_identity.tex:147 | ACCEPTED -> insertion block B01 |
| Evidence_Baryon.tex:46 | mahaffey2026_cc | - | EXCLUDED: self-citation (stand-alone rule) |
| Evidence_Baryon.tex:77 | cyburt2016 | part2/p2_13b_baryon_chain.tex:22 | already cited at this place (Cyburt2016) |
| Evidence_Baryon.tex:83 | rubakov1996 | part2/p2_13b_baryon_chain.tex:25 | already cited at this place (RubakovShaposhnikov1996) |
| Evidence_Baryon.tex:83 | davidson2008 | part2/p2_13b_baryon_chain.tex:25 | already cited at this place (Davidson2008) |
| Evidence_Baryon.tex:102 | mahaffey2026_cc | - | EXCLUDED: self-citation (stand-alone rule) |
| Evidence_Baryon.tex:253 | planck2020 | part2/p2_13b_baryon_chain.tex:150 | already cited at this place (Planck2018VI) |
| Evidence_Baryon.tex:272 | mahaffey2026_cc | - | EXCLUDED: self-citation (stand-alone rule) |
| Evidence_Baryon.tex:338 | jacobson1995 | part1/p1_04_virial_identity.tex:147 | ACCEPTED -> insertion block B01 |
| iam_electroweak_matter_sector.tex:69 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_electroweak_matter_sector.tex:93 | clausius1870 | part2/p2_22_electroweak.tex:21 | ACCEPTED -> insertion block B27 |
| iam_electroweak_matter_sector.tex:96 | mahaffey2026_thermo_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_electroweak_matter_sector.tex:116 | wu1957 | part2/p2_22_electroweak.tex:39 | already cited at this place (Wu1957) |
| iam_electroweak_matter_sector.tex:117 | cronin1964 | part2/p2_22_electroweak.tex:39 | cited at this place as Christenson1964 (same PRL 13, 138 paper) |
| iam_electroweak_matter_sector.tex:117 | ckm1973 | part2/p2_22_electroweak.tex:39 | already cited at this place (KobayashiMaskawa1973) |
| iam_electroweak_matter_sector.tex:123 | sakharov1967 | part2/p2_22_electroweak.tex:43 | already cited at this place (Sakharov1967) |
| iam_electroweak_matter_sector.tex:164 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_electroweak_matter_sector.tex:210 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_electroweak_matter_sector.tex:210 | mahaffey2026_thermo_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_electroweak_matter_sector.tex:211 | planck2020 | part2/p2_22_electroweak.tex:88 | already cited at this place (Planck2018VI) |
| iam_mu_sigma_paper.tex:47 | Planck2018VI | part2/p2_07_late_time_growth.tex:30 | already cited at this place |
| iam_mu_sigma_paper.tex:48 | DESY3 | part2/p2_07_late_time_growth.tex:32 | already cited at this place |
| iam_mu_sigma_paper.tex:48 | KiDS1000 | part2/p2_07_late_time_growth.tex:32 | already cited at this place (Heymans2021) |
| iam_mu_sigma_paper.tex:48 | HSC | part2/p2_07_late_time_growth.tex:32 | already cited at this place (HSCY3) |
| iam_mu_sigma_paper.tex:51 | BeanTangmatitham2010 | part2/p2_07_late_time_growth.tex:40 | already cited at this place |
| iam_mu_sigma_paper.tex:51 | Daniel2010 | part2/p2_07_late_time_growth.tex:40 | already cited at this place |
| iam_mu_sigma_paper.tex:51 | PogosianSilvestri2016 | part2/p2_07_late_time_growth.tex:40 | already cited at this place |
| iam_mu_sigma_paper.tex:76 | Mahaffey2026theory | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_mu_sigma_paper.tex:162 | Planck2018V | part2/p2_07_late_time_growth.tex:152 | already cited at this place |
| iam_mu_sigma_paper.tex:162 | Planck2018VI | part2/p2_07_late_time_growth.tex:152 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.25) |
| iam_mu_sigma_paper.tex:175 | Pantheon | part2/p2_07_late_time_growth.tex:169 | already cited at this place (Brout2022) |
| iam_mu_sigma_paper.tex:397 | KiDS1000 | part2/p2_07_late_time_growth.tex:352 | already cited at this place (Heymans2021) |
| iam_mu_sigma_paper.tex:397 | DESY3 | part2/p2_07_late_time_growth.tex:352 | already cited at this place |
| iam_mu_sigma_paper.tex:405 | PogosianSilvestri2016 | part2/p2_07_late_time_growth.tex:360 | already cited at this place |
| iam_decoherence_bridge_paper.tex:70 | Penrose1989 | part2/p2_03_theory.tex:43 | already cited at this place |
| iam_decoherence_bridge_paper.tex:72 | Zurek2003 | part2/p2_03_theory.tex:44 | already cited at this place |
| iam_decoherence_bridge_paper.tex:72 | Schlosshauer2007 | part2/p2_03_theory.tex:44 | already cited at this place |
| iam_decoherence_bridge_paper.tex:85 | Bekenstein1973 | part2/p2_03_theory.tex:49 | ACCEPTED -> insertion block B14 |
| iam_decoherence_bridge_paper.tex:85 | Hawking1975 | part2/p2_03_theory.tex:49 | ACCEPTED -> insertion block B14 |
| iam_decoherence_bridge_paper.tex:86 | CaiKim2005 | part1/p1_02_iams_law.tex:283 | already cited at this place |
| iam_decoherence_bridge_paper.tex:149 | Landauer1961 | part2/p2_03_theory.tex:121 | already cited at this place |
| iam_decoherence_bridge_paper.tex:171 | Jacobson1995 | part2/p2_03_theory.tex:130 | already cited at this place |
| iam_decoherence_bridge_paper.tex:172 | CaiKim2005 | part2/p2_03_theory.tex:130 | already cited at this place |
| iam_decoherence_bridge_paper.tex:477 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_decoherence_bridge_paper.tex:551 | Jacobson1995 | part2/p2_03_theory.tex:1019 | already cited at this place |
| iam_decoherence_bridge_paper.tex:552 | Verlinde2011 | part2/p2_03_theory.tex:1019 | already cited at this place |
| iam_decoherence_bridge_paper.tex:553 | Padmanabhan2010 | part2/p2_03_theory.tex:1019 | already cited at this place |
| iam_decoherence_bridge_paper.tex:560 | PogosianSilvestri2016 | part2/p2_03_theory.tex:1023 | already cited at this place |
| iam_decoherence_bridge_paper.tex:560 | Abbott2023 | part2/p2_03_theory.tex:1023 | ACCEPTED -> insertion block B24 (DESY3Ext) |
| iam_decoherence_bridge_paper.tex:560 | Andrade2024 | part2/p2_03_theory.tex:1023 | already cited at this place |
| iam_decoherence_bridge_paper.tex:560 | DESIDR1 | part2/p2_03_theory.tex:1023 | book cites DESI2025 / DESI2024VII at this place (source bibitem has mismatched title and volume) |
| iam_decoherence_bridge_paper.tex:578 | tHooft1993 | part2/p2_03_theory.tex:1033 | already cited at this place |
| iam_decoherence_bridge_paper.tex:578 | Susskind1995 | part2/p2_03_theory.tex:1033 | already cited at this place |
| iam_decoherence_bridge_paper.tex:620 | Mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_Lensing_Dynamics_Paper.tex:103 | Mahaffey2026a | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_Lensing_Dynamics_Paper.tex:103 | Mahaffey2026b | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_Lensing_Dynamics_Paper.tex:323 | LSST2009 | part2/p2_17_lensing_dynamics.tex:241 | book cites Ivezic2019 (LSST overview paper) at this place |
| IAM_wz_FarFuture_Paper.tex:95 | Riess1998 | part2/p2_11_dark_energy.tex:18 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:95 | Perlmutter1999 | part2/p2_11_dark_energy.tex:18 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:95 | PlanckVI2020 | part2/p2_11_dark_energy.tex:18 | already cited at this place (Planck2018VI) |
| IAM_wz_FarFuture_Paper.tex:97 | DESIDR2_2025 | part2/p2_11_dark_energy.tex:26 | already cited at this place (DESI2025) |
| IAM_wz_FarFuture_Paper.tex:99 | Mahaffey2026_obs | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_wz_FarFuture_Paper.tex:99 | Jacobson1995 | part2/p2_11_dark_energy.tex:32 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:99 | CaiKim2005 | part2/p2_11_dark_energy.tex:32 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:110 | CaiKim2005 | part2/p2_11_dark_energy.tex:48 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:115 | Mahaffey2026_theory | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_wz_FarFuture_Paper.tex:157 | CPL2001 | part2/p2_11_dark_energy.tex:120 | already cited at this place (ChevallierPolarski2001) |
| IAM_wz_FarFuture_Paper.tex:157 | Linder2003 | part2/p2_11_dark_energy.tex:120 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:183 | Mahaffey2026_CAMB | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_wz_FarFuture_Paper.tex:238 | PlanckVI2020 | part2/p2_11_dark_energy.tex:215 | already cited at this place (Planck2018VI) |
| IAM_wz_FarFuture_Paper.tex:301 | GibbonsHawking1977 | part2/p2_20_wz_far_future.tex:51 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:307 | Bekenstein1973 | part2/p2_20_wz_far_future.tex:61 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:307 | Hawking1975 | part2/p2_20_wz_far_future.tex:61 | already cited at this place |
| IAM_wz_FarFuture_Paper.tex:327 | DESIDR2_2025 | part2/p2_20_wz_far_future.tex:94 | already cited at this place (DESI2025) |
| IAM_wz_FarFuture_Paper.tex:380 | Caldwell2002 | part2/p2_20_wz_far_future.tex:168 | already cited at this place |
| iam_decoherence_virial_partition.tex:61 | Jacobson1995 | part5/p5_05c_virial_decoherence.tex:19 | already cited at this place |
| iam_decoherence_virial_partition.tex:61 | CaiKim2005 | part5/p5_05c_virial_decoherence.tex:19 | already cited at this place |
| iam_decoherence_virial_partition.tex:94 | IAM_theory | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_decoherence_virial_partition.tex:94 | IAM_obs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_decoherence_virial_partition.tex:99 | Landauer1961 | part5/p5_05c_virial_decoherence.tex:37 | already cited at this place |
| iam_decoherence_virial_partition.tex:106 | Jacobson1995 | part5/p5_05c_virial_decoherence.tex:40 | already cited at this place |
| iam_decoherence_virial_partition.tex:106 | CaiKim2005 | part5/p5_05c_virial_decoherence.tex:40 | already cited at this place |
| iam_decoherence_virial_partition.tex:106 | Gibbons1977 | part5/p5_05c_virial_decoherence.tex:40 | already cited at this place (GibbonsHawking1977) |
| iam_decoherence_virial_partition.tex:150 | Landauer1961 | part2/p2_02_virial.tex:59 | ACCEPTED -> insertion block B12 |
| iam_decoherence_virial_partition.tex:168 | IAM_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_decoherence_virial_partition.tex:206 | IAM_time | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_decoherence_virial_partition.tex:211 | BarbourBertotti1982 | part5/p5_03_time.tex:20 | already cited at this place |
| iam_decoherence_virial_partition.tex:224 | Gibbons1977 | part5/p5_05c_virial_decoherence.tex:73 | already cited at this place (GibbonsHawking1977) |
| iam_decoherence_virial_partition.tex:228 | Sakharov1967 | part5/p5_05c_virial_decoherence.tex:75 | already cited at this place |
| iam_decoherence_virial_partition.tex:239 | Jacobson1995 | part5/p5_05c_virial_decoherence.tex:20 | already cited at this place |
| iam_decoherence_virial_partition.tex:239 | CaiKim2005 | part5/p5_05c_virial_decoherence.tex:20 | already cited at this place |
| iam_decoherence_virial_partition.tex:297 | IAM_BH | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_decoherence_virial_partition.tex:309 | IAM_BH | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_decoherence_virial_partition.tex:432 | Smolin2013 | part5/p5_05c_virial_decoherence.tex:124 | already cited at this place |
| iam_decoherence_virial_partition.tex:440 | Smolin2013 | part5/p5_05c_virial_decoherence.tex:127 | already cited at this place |
| iam_decoherence_virial_partition.tex:454 | England2013 | part5/p5_05c_virial_decoherence.tex:134 | already cited at this place |
| iam_decoherence_virial_partition.tex:454 | England2015 | part5/p5_05c_virial_decoherence.tex:134 | already cited at this place |
| iam_decoherence_virial_partition.tex:488 | Connes1994 | part5/p5_05c_virial_decoherence.tex:150 | already cited at this place (ConnesRovelli1994) |
| iam_decoherence_virial_partition.tex:488 | Rovelli1993 | part5/p5_05c_virial_decoherence.tex:150 | already cited at this place |
| iam_decoherence_virial_partition.tex:493 | Rovelli2019 | part5/p5_05c_virial_decoherence.tex:152 | already cited at this place |
| iam_decoherence_virial_partition.tex:493 | Rovelli2020 | part5/p5_05c_virial_decoherence.tex:152 | already cited at this place (Rovelli2022) |
| iam_decoherence_virial_partition.tex:519 | Smolin1992 | part5/p5_05c_virial_decoherence.tex:165 | already cited at this place |
| iam_decoherence_virial_partition.tex:519 | Smolin1997 | part5/p5_05c_virial_decoherence.tex:165 | already cited at this place |
| iam_decoherence_virial_partition.tex:549 | Jacobson1995 | part1/p1_04_virial_identity.tex:147 | ACCEPTED -> insertion block B01 |
| iam_ocolgain_note.tex:71 | Adil2023 | part2/p2_08_s8_trend.tex:33 | EXCLUDED by p2_08 naming rule (trend paper cited by journal and arXiv number only) |
| iam_ocolgain_note.tex:119 | Mahaffey2026a | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_ocolgain_note.tex:138 | Adil2023 | part2/p2_08_s8_trend.tex:78 | EXCLUDED by p2_08 naming rule (trend paper cited by journal and arXiv number only) |
| iam_ocolgain_note.tex:150 | Planck2018 | part2/p2_08_s8_trend.tex:90 | ACCEPTED -> insertion block B25 |
| iam_bekenstein_coefficient.tex:88 | Jacobson1995 | part2/p2_03_theory.tex:48 | already cited at this place |
| iam_bekenstein_coefficient.tex:120 | CaiKim2005 | part2/p2_01a_bekenstein.tex:63 | already cited at this place |
| iam_bekenstein_coefficient.tex:120 | Padmanabhan2010 | part2/p2_01a_bekenstein.tex:63 | already cited at this place |
| iam_bekenstein_coefficient.tex:122 | Rovelli1995 | part2/p2_01a_bekenstein.tex:64 | already cited at this place (RovelliSmolin1995) |
| iam_bekenstein_coefficient.tex:123 | Strominger1996 | part2/p2_01a_bekenstein.tex:64 | already cited at this place (StromingerVafa1996) |
| iam_bekenstein_coefficient.tex:168 | Landauer1961 | part2/p2_01a_bekenstein.tex:88 | already cited at this place |
| iam_bekenstein_coefficient.tex:261 | Unruh1976 | part2/p2_01a_bekenstein.tex:145 | already cited at this place |
| iam_bekenstein_coefficient.tex:261 | Bisognano1976 | part2/p2_01a_bekenstein.tex:145 | already cited at this place (BisognanoWichmann1976) |
| iam_bekenstein_coefficient.tex:300 | Wald1984 | part2/p2_01a_bekenstein.tex:184 | already cited at this place |
| iam_bekenstein_coefficient.tex:389 | Bisognano1976 | part2/p2_01a_bekenstein.tex:294 | already cited at this place (BisognanoWichmann1976) |
| iam_bekenstein_coefficient.tex:514 | tHooft1993 | part2/p2_01a_bekenstein.tex:359 | already cited at this place |
| iam_bekenstein_coefficient.tex:514 | Susskind1995 | part2/p2_01a_bekenstein.tex:359 | already cited at this place |
| iam_bekenstein_coefficient.tex:606 | Landauer1961 | part2/p2_01a_bekenstein.tex:397 | already cited at this place |
| iam_bekenstein_coefficient.tex:607 | Zurek2003 | part2/p2_01a_bekenstein.tex:398 | already cited at this place |
| iam_bekenstein_coefficient.tex:611 | Bisognano1976 | part2/p2_01a_bekenstein.tex:400 | already cited at this place (BisognanoWichmann1976) |
| iam_bekenstein_coefficient.tex:612 | Wald1984 | part2/p2_01a_bekenstein.tex:401 | already cited at this place |
| iam_bekenstein_coefficient.tex:635 | Jacobson1995 | part2/p2_01a_bekenstein.tex:413 | already cited at this place |
| iam_bekenstein_coefficient.tex:636 | Unruh1976 | part2/p2_01a_bekenstein.tex:413 | already cited at this place |
| iam_bekenstein_coefficient.tex:637 | Davies1975 | part2/p2_01a_bekenstein.tex:413 | already cited at this place |
| iam_bekenstein_coefficient.tex:638 | Bisognano1976 | part2/p2_01a_bekenstein.tex:414 | already cited at this place (BisognanoWichmann1976) |
| docs_RETIRED_2026-10_top_level_development_IAM_Manuscript.tex:42 | tHooft1993 | part5/p5_01b_bh_information.tex:143 | already cited at this place |
| docs_RETIRED_2026-10_top_level_development_IAM_Manuscript.tex:42 | Susskind1995 | part5/p5_01b_bh_information.tex:143 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.27) |
| docs_RETIRED_2026-10_top_level_development_IAM_Manuscript.tex:44 | Zurek2003 | part2/p2_03_theory.tex:901 | already cited at this place |
| docs_RETIRED_2026-10_top_level_development_IAM_Manuscript.tex:52 | Cai2005 | part2/p2_03a_entropic_gravity.tex:9 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.29) |
| Dual_Sector_Validation_Paper.tex:37 | PlanckCollaboration2020 | part2/p2_10_dual_sector_validation.tex:28 | already cited at this place (Planck2018VI) |
| Dual_Sector_Validation_Paper.tex:37 | Riess2022 | part2/p2_10_dual_sector_validation.tex:28 | already cited at this place |
| Dual_Sector_Validation_Paper.tex:37 | Kamionkowski2023 | part2/p2_10_dual_sector_validation.tex:30 | already cited at this place |
| Dual_Sector_Validation_Paper.tex:37 | Amendola2020 | part2/p2_10_dual_sector_validation.tex:30 | book cites Amendola2018 at this place (Living Rev. Rel. review); source key carries wrong volume/year |
| Dual_Sector_Validation_Paper.tex:37 | DiValentino2021 | part2/p2_10_dual_sector_validation.tex:30 | already cited at this place |
| Dual_Sector_Validation_Paper.tex:39 | IAMManuscript2025 | - | EXCLUDED: self-citation (stand-alone rule) |
| Dual_Sector_Validation_Paper.tex:43 | Brout2022 | part2/p2_10_dual_sector_validation.tex:49 | already cited at this place |
| Dual_Sector_Validation_Paper.tex:306 | IAMCompendium2025 | - | EXCLUDED: self-citation (stand-alone rule) |
| Dual_Sector_Validation_Paper.tex:400 | Riess2022 | part2/p2_10_dual_sector_validation.tex:406 | already cited at this place |
| Dual_Sector_Validation_Paper.tex:410 | Riess2022 | part2/p2_10_dual_sector_validation.tex:413 | already cited at this place |
| Dual_Sector_Validation_Paper.tex:418 | Kamionkowski2023 | part2/p2_10_dual_sector_validation.tex:427 | already cited at this place |
| Dual_Sector_Validation_Paper.tex:492 | Amendola2020 | part2/p2_10_dual_sector_validation.tex:501 | book cites Amendola2018 at this place (Living Rev. Rel. review); source key carries wrong volume/year |
| electron_mass.tex:73 | jacobson1995 | part5/p5_05c_virial_decoherence.tex:20 | already cited at this place (Jacobson1995) |
| electron_mass.tex:73 | cai2005 | part5/p5_05c_virial_decoherence.tex:20 | already cited at this place (CaiKim2005) |
| electron_mass.tex:139 | iam_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| electron_mass.tex:158 | iam_virial | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_BH_Cosmology_Paper_B.tex:55 | MahaffeyBHThermo2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_BH_Cosmology_Paper_B.tex:73 | HawkingPerry2016 | part5/p5_01b_bh_information.tex:22 | already cited at this place (HawkingPerryStrominger2016) |
| IAM_BH_Cosmology_Paper_B.tex:94 | Hawking1975 | part5/p5_01b_bh_information.tex:28 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:94 | Page1993 | part5/p5_01b_bh_information.tex:28 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:94 | Maldacena1997 | part5/p5_01b_bh_information.tex:28 | already cited at this place (Maldacena1998) |
| IAM_BH_Cosmology_Paper_B.tex:94 | AMPS2012 | part5/p5_01b_bh_information.tex:28 | already cited at this place (AMPS2013) |
| IAM_BH_Cosmology_Paper_B.tex:94 | MaldacenaSusskind2013 | part5/p5_01b_bh_information.tex:28 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:94 | Penrose2004 | part5/p5_01b_bh_information.tex:28 | book cites Penrose1989 at this place |
| IAM_BH_Cosmology_Paper_B.tex:100 | HawkingPerry2016 | part5/p5_01b_bh_information.tex:31 | already cited at this place (HawkingPerryStrominger2016) |
| IAM_BH_Cosmology_Paper_B.tex:108 | Mahaffey2026theory | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_BH_Cosmology_Paper_B.tex:112 | Landauer1961 | part5/p5_01b_bh_information.tex:37 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:233 | HawkingPerry2016 | part5/p5_01b_bh_information.tex:96 | already cited at this place (HawkingPerryStrominger2016) |
| IAM_BH_Cosmology_Paper_B.tex:317 | tHooft1993 | part2/p2_03_theory.tex:80 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:317 | Susskind1995 | part2/p2_03_theory.tex:80 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:355 | Hawking1975 | part5/p5_01b_bh_information.tex:131 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:361 | Susskind1993 | part5/p5_01b_bh_information.tex:135 | already cited at this place (SusskindThorlaciusUglum1993) |
| IAM_BH_Cosmology_Paper_B.tex:377 | tHooft1993 | part5/p5_01b_bh_information.tex:143 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:410 | AMPS2012 | part5/p5_01b_bh_information.tex:167 | already cited at this place (AMPS2013) |
| IAM_BH_Cosmology_Paper_B.tex:427 | Page1993 | part2/p2_01_blackholes.tex:188 | already cited at this place |
| IAM_BH_Cosmology_Paper_B.tex:525 | MahaffeyParticle2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_desi_paper.tex:111 | DESI2025DR2BAO | part2/p2_09b_phantom_crossing.tex:134 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.38) |
| iam_desi_paper.tex:146 | Planck2018params | part2/p2_09_sector_tension.tex:37 | already cited at this place (Planck2018VI) |
| iam_desi_paper.tex:153 | Planck2018params | part2/p2_09_sector_tension.tex:41 | already cited at this place (Planck2018VI) |
| iam_desi_paper.tex:166 | Asgari2021kids | part2/p2_07_late_time_growth.tex:353 | already cited at this place (Asgari2021) |
| iam_desi_paper.tex:167 | Abbott2022desy3 | part2/p2_07_late_time_growth.tex:353 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.38) |
| iam_desi_paper.tex:168 | Li2023hsc | part2/p2_07_late_time_growth.tex:353 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.38) |
| iam_desi_paper.tex:175 | DESI2024DR1BAO | part2/p2_02b_virial_tests.tex:104 | book sentence updated to DR2 and cites DESI2025 |
| iam_desi_paper.tex:175 | DESI2024DR1cosmo | part2/p2_02b_virial_tests.tex:104 | book sentence updated to DR2 and cites DESI2025 |
| iam_desi_paper.tex:250 | Jacobson1995 | part2/p2_09_sector_tension.tex:89 | already cited at this place |
| iam_desi_paper.tex:254 | CaiKim2005 | part2/p2_09_sector_tension.tex:91 | already cited at this place |
| iam_desi_paper.tex:518 | Alam2017boss | part2/p2_09_sector_tension.tex:196 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.28) |
| iam_desi_paper.tex:519 | deMattia2021 | part2/p2_09_sector_tension.tex:196 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.39) |
| iam_desi_paper.tex:519 | deMattia2021 | part2/p2_09_sector_tension.tex:196 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.39) |
| iam_desi_paper.tex:636 | Wright2025kids | part2/p2_09_sector_tension.tex:259 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.31) |
| iam_desi_paper.tex:642 | Abbott2022desy3 | part2/p2_09_sector_tension.tex:258 | not carried: the book sentence matches only partly and does not make the claim the source cites for (cov 0.30) |
| iam_desi_paper.tex:883 | DESI2024DR1FS | part2/p2_09b_phantom_crossing.tex:55 | already cited at this place (DESI2024V) |
| iam_desi_paper.tex:949 | Ocolgain2024 | part2/p2_09b_phantom_crossing.tex:96 | EXCLUDED by p2_09b naming rule (consistency analyses cited by journal and arXiv number in the text) |
| iam_desi_paper.tex:953 | Ocolgain2025 | part2/p2_09b_phantom_crossing.tex:97 | EXCLUDED by p2_09b naming rule (journal and arXiv number in the text) |
| iam_virial_dark_sector.tex:91 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_virial_dark_sector.tex:133 | landauer1961 | part2/p2_02_virial.tex:44 | already cited at this place (Landauer1961) |
| iam_virial_dark_sector.tex:135 | jacobson1995 | part2/p2_02_virial.tex:45 | ACCEPTED -> insertion block B11 |
| iam_virial_dark_sector.tex:135 | cai2005 | part2/p2_02_virial.tex:45 | ACCEPTED -> insertion block B11 |
| iam_virial_dark_sector.tex:192 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_virial_dark_sector.tex:327 | desi2025_dr2 | part2/p2_02b_virial_tests.tex:104 | already cited at this place (DESI2025) |
| iam_virial_dark_sector.tex:335 | mahaffey2026_desi | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_virial_dark_sector.tex:476 | ocolgain2025 | part2/p2_02b_virial_tests.tex:112 | already cited at this place (Colgain2025) |
| iam_virial_dark_sector.tex:476 | liu2024 | part2/p2_02b_virial_tests.tex:112 | already cited at this place (Liu2024LRG) |
| IAM_Virial_37_Orders_Paper.tex:81 | Mahaffey2026a | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_Virial_37_Orders_Paper.tex:108 | Clausius1870 | part5/p5_05b_virial_partners.tex:35 | ACCEPTED -> insertion block B30 |
| IAM_Virial_37_Orders_Paper.tex:167 | Clementi1974 | part1/p1_04_virial_identity.tex:127 | already cited at this place |
| IAM_Virial_37_Orders_Paper.tex:167 | Bunge1993 | part1/p1_04_virial_identity.tex:127 | already cited at this place |
| IAM_Virial_37_Orders_Paper.tex:177 | Clausius1870 | part1/p1_03_virial_law.tex:81 | ACCEPTED -> insertion block B08 |
| IAM_Virial_37_Orders_Paper.tex:194 | Landauer1961 | part1/p1_04_virial_identity.tex:113 | already cited at this place |
| IAM_Virial_37_Orders_Paper.tex:217 | CaiKim2005 | part2/p2_03_theory.tex:742 | ACCEPTED -> insertion block B18 |
| IAM_Virial_37_Orders_Paper.tex:354 | Wright2025 | part2/p2_09_sector_tension.tex:50 | already cited at this place |
| IAM_Virial_37_Orders_Paper.tex:556 | Zhang2007 | part2/p2_02b_virial_tests.tex:55 | already cited at this place (Zhang2007EG) |
| IAM_Virial_37_Orders_Paper.tex:556 | Reyes2010 | part2/p2_02b_virial_tests.tex:55 | already cited at this place |
| iam_two_faces_of_time.tex:73 | BarbourBertotti1982 | part5/p5_03_time.tex:9 | already cited at this place |
| iam_two_faces_of_time.tex:97 | BarbourBertotti1982 | part5/p5_03_time.tex:19 | already cited at this place |
| iam_two_faces_of_time.tex:97 | Barbour1994 | part5/p5_03_time.tex:19 | already cited at this place |
| iam_two_faces_of_time.tex:130 | Landauer1961 | part5/p5_05c_virial_decoherence.tex:72 | ACCEPTED -> insertion block B31 |
| iam_two_faces_of_time.tex:165 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_two_faces_of_time.tex:228 | PlanckCollaboration2020 | part5/p5_03_time.tex:93 | already cited at this place (Planck2018VI) |
| iam_two_faces_of_time.tex:235 | Riess2022 | part5/p5_03_time.tex:96 | already cited at this place |
| iam_two_faces_of_time.tex:235 | mahaffey2026obs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_two_faces_of_time.tex:250 | mahaffey2026 | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_two_faces_of_time.tex:286 | mahaffey2026higgs | - | EXCLUDED: self-citation (stand-alone rule) |
| Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex:236 | Landauer1961 | part1/p1_01_encoding_surfaces.tex:54 | already cited at this place |
| Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex:236 | Bennett1982 | part1/p1_01_encoding_surfaces.tex:54 | ACCEPTED -> insertion block B06 |
| Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex:246 | Bestor2000 | part1/p1_01_encoding_surfaces.tex:101 | ACCEPTED -> insertion block B07 |
| Biological_Physics_RETIRED_2026-10_MethylPhys_papers_Mahaffey_2026_cell_thermodynamics.tex:246 | Jurkowska2011 | part1/p1_01_encoding_surfaces.tex:101 | ACCEPTED -> insertion block B07 |
| IAM_Survey_Predictions_Paper.tex:58 | Mahaffey2026a | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_Survey_Predictions_Paper.tex:58 | Mahaffey2026b | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_Survey_Predictions_Paper.tex:199 | Giannantonio2008 | part2/p2_16_survey_predictions.tex:140 | already cited at this place |
| IAM_Survey_Predictions_Paper.tex:199 | Planck2016ISW | part2/p2_16_survey_predictions.tex:140 | already cited at this place (Planck2015ISW) |
| IAM_Survey_Predictions_Paper.tex:401 | DESI2025 | part2/p2_16_survey_predictions.tex:207 | book cites DESI2025 / DESI2024VII at this place |
| Paper9_Measurement_Problem.tex:137 | Mahaffey2026a | - | EXCLUDED: self-citation (stand-alone rule) |
| IAM_Saridakis_Bridge.tex:167 | Saridakis2020barrow | part2/p2_03a_entropic_gravity.tex:58 | already cited at this place |
| IAM_Saridakis_Bridge.tex:192 | Berut2012 | part2/p2_03a_entropic_gravity.tex:75 | already cited at this place |
| IAM_Saridakis_Bridge.tex:192 | Jun2014 | part2/p2_03a_entropic_gravity.tex:75 | already cited at this place |
| iam_missing_satellites.tex:88 | Klypin1999 | part2/p2_19_missing_satellites.tex:26 | already cited at this place |
| iam_missing_satellites.tex:88 | Moore1999 | part2/p2_19_missing_satellites.tex:26 | already cited at this place |
| iam_missing_satellites.tex:96 | Bullock2000 | part2/p2_19_missing_satellites.tex:31 | already cited at this place |
| iam_missing_satellites.tex:96 | Benson2002 | part2/p2_19_missing_satellites.tex:31 | already cited at this place (Benson2002b) |
| iam_missing_satellites.tex:97 | Mayer2006 | part2/p2_19_missing_satellites.tex:31 | already cited at this place |
| iam_missing_satellites.tex:98 | Read2005 | part2/p2_19_missing_satellites.tex:31 | ACCEPTED -> insertion block B26 (ReadGilmore2005; source bibitem metadata garbled) |
| iam_missing_satellites.tex:98 | Pontzen2012 | part2/p2_19_missing_satellites.tex:31 | already cited at this place (PontzenGovernato2012) |
| iam_missing_satellites.tex:174 | Jacobson1995 | part5/p5_05c_virial_decoherence.tex:20 | already cited at this place |
| iam_missing_satellites.tex:174 | CaiKim2005 | part5/p5_05c_virial_decoherence.tex:20 | already cited at this place |
| iam_missing_satellites.tex:188 | IAM_obs | - | EXCLUDED: self-citation (stand-alone rule) |
| iam_missing_satellites.tex:219 | PressSchechter1974 | part2/p2_19_missing_satellites.tex:100 | already cited at this place |
| iam_missing_satellites.tex:232 | PlanckVI2020 | part1/p1_02_iams_law.tex:684 | ACCEPTED -> insertion block B02 |
| iam_missing_satellites.tex:232 | Hojjati2011 | part1/p1_02_iams_law.tex:684 | ACCEPTED -> insertion block B02 |
| iam_missing_satellites.tex:233 | Torrado2021 | part1/p1_02_iams_law.tex:684 | ACCEPTED -> insertion block B02 |
| iam_missing_satellites.tex:514 | Jacobson1995 | part1/p1_04_virial_identity.tex:147 | ACCEPTED -> insertion block B01 |

## 3. Glossary (docs/book/appendices/app_F_glossary.tex, OWNED)

677 entries (539 at HEAD): 515 carried unchanged with recomputed chapter pointers, 15 corrected to the current chapters or rulings, 147 new; 9 removed. Same longtable format, alphabetical by first word, pointers to chapter labels (up to three: defining chapter(s) first, then the chapters that use the term most). Canon wording kept where the canon defines the term (A, H_min, IAM floor, Met-A, IAM-A, C-score, Mahaffey number, n, Normal band, healthy reference, one gauge); no product names, no retired terms, no group statistics, no clinical claims; the 'IAM' entry is not expanded.

**Removed (term no longer used in any chapter at 3290e60):** Apple M1; EPLG (replaced by 'Error per layered gate', the wording the walls chapter uses); London penetration depth; Lyman-alpha forest; Median absolute error; Natural units; Salmon-A; Shrinkage; x86, ARM.

**Corrected entries** (old text no longer matched the chapter it points to; each new value traced to the chapter line; no change to the physics):

- Asymptotic expansion rates: p2_11 now quotes the photon-sector H0 67.16 (55.57, 70.86) first, Planck base values second
- Background (cosmological): author wording restored at 06aabc5 (term lives in the entropy functional; background placement moves H0 to 61.5, 10.9 sigma)
- $C(d)$ (genome-distance correlation): p4_16a numbers: halves just below 1 kb, 0.32 at 1-1.8 kb, 0.05 at 3-6 kb
- Coherence optimum ($T_1^*$): p3_05: 0.65 needs a/b=3.45; class microsecond values no longer in the chapter; model labelled conjecture as in the chapter
- DGP gravity: the book cites Koyama2007 for the ghost and Fang2008 for the tension; no 5.3 sigma in the book
- Hairpin-bisulfite sequencing: p4_02 quotes 0.90-0.98 (Genereux2005)
- Hydrostatic bias: C_NT factor no longer in the book
- Koomey's law: p3_06 says about every 2.7 years
- Level 1, Level 2, Level 2b: Level 2b author wording restored at 06aabc5
- Mammalian methylation array: 348-species figure not in the book
- Missing satellites problem: Mechanism B retired; p2_19 wording (order of magnitude; under one per cent)
- Photonic qubit: 0.05 is the photonic two-qubit error in p3_03, not a material floor
- Render-time guard: render-time guard text without group vocabulary
- Senescence: p4_06 table values for senescent IMR90
- TOV limit: GW170817 bound not in the book; p2_01 figure gives about 2.3 M_sun, A about 1.64
- IAM: expansion removed (author ruling 2026-10-03), as at 3290e60.

**New entries:** Across-array SD; ACT, WMAP; Activation energy $E_a$; Affine parameter; AMPS argument, firewall; BCS theory; Bead count; BGS, LRG, ELG, QSO (DESI tracers); Bisognano--Wichmann theorem; Bit, nat; Black-hole information paradox; Blast (leukaemic blast); Block mean; Boltzmann solver; Boost Killing vector; Bottom-up and top-down exponents; Boundary capacity; Bulk flow; CODATA 2018; Coho salmon; Complementarity, ER = EPR, island formula; Composition error; Conical singularity; Constant-field scaling; Conversion failure; Cosmic censorship conjecture; Cosmological natural selection; Crossover mass; de Sitter state; Decoherence time $\tau_D$; $\delta_c$, $\sigma_M$ (collapse threshold, mass variance); Dissipation-driven adaptation; Distance ladder; Double-slit experiment; Einstein tensor; Electroweak baryogenesis, leptogenesis; Electroweak crossover; Emergency haematopoiesis; EM-seq; E-MTAB-7309; Encoding throughput $\Gamma_{\rm thermal}$; Error per layered gate; F0, F1 (generations); Fine-grained entropy; Fine-tuning problem; First law (thermodynamics, horizons); Flavour space; Floor breach; Gauss's law; GEO; Gravitational slip; $g_{*s}$ (entropy degrees of freedom); GSE135205; GSE250556; GSE329728; Half-split test; Hawking luminosity; Heat death; hg19; Higgs field, Higgs boson; History integral; Horizon capacity; Hydrostatic equilibrium; Inflection point (of $E(a)$); Informational completion; Intake line; Integrity hash (SHA-256); Intracluster medium; Isolated error; Jarlskog invariant; JWST; KMS period; Known-pattern control; Lensing (Weyl) potential; Leptogenesis; Local hidden-variable theory; Local Volume Database; Logically irreversible; Luminosity distance $d_L$, distance modulus; $M_{\rm lens}/M_{\rm dyn}$ (lensing-to-dynamical mass ratio); M87; Map-making; Masking rule; Mass-sheet degeneracy; Measure, don't compare; Median tare; MICROSCOPE mission, E\"otv\"os parameter; MOND; Motional mode; Negative mass; Nine-step chain; Noob, dye-bias correction, probe-type normalisation; Nucleated red cell; Null congruence; oxWGBS; Page curve; Pair breaking; Photon exemption; Pipeline offset; Planck energy $E_P$; Planck time; Poisson's equation; Polytrope; pOOBAH; Posterior-predictive check; Progenitor floor; Pseudo-$C_\ell$, split-half cross-spectra; Purity; QCD transition; Qualifying molecule; Quantinuum Sol; Quantum inequalities; Rearing-temperature experiment; Refusal (platform refusal); Running mass; Runs A, C, D (Level~2 chains); SAH, methionine adenosyltransferase; Schr\"odinger's cat; SDSS; Second law of thermodynamics; Sentrix identifier; Sex mismatch; Shape dynamics, Barbour--Bertotti; ShapeFit; $\sigma$ (Cornell linear coefficient); Silica sphere (test mass); Slide effect; Slide tare; SNe; Soft hair; Span (reference span); Stochastic gravitational-wave background; $T_{2,\rm CPMG}$; $\tau_{\rm IAM}$ (IAM decoherence time); Thermal de Broglie wavelength; Thermal time hypothesis; Tissue profile; Trace cell; Two-horizon framework; Unitarity; Virial partition; Withheld; Within-person SD; WtG, CCCP, LoCuSS; Yukawa coupling; Zone I, Zone II; Zwicky's virial mass

**Checks (verify_bib_glossary_output.txt):** every \ref resolves (incl. part:1-5 in main.tex); every \cite key is in iam.bib; braces and $ balance per entry; every number of two or more digits occurs in the chapter/appendix text; every macro used occurs elsewhere in the book or preamble; CANON/canon_check.py: 0 LIVE BLOCK hits repo-wide (sparse checkout: docs/book, docs/verification, docs/papers, CANON) and 0 hits of any severity in the glossary.


**Lead note 2026-10-03 (applied):** chain v3 Stage T is the median same-run tare only ($A_{rel}=A/$median of >=3 same-run healthy references; nothing fitted), and the noise index N gates the gauge state (withheld outside the references' range, 0.122-0.149 on the six reference arrays, p4_19 l.83-85). Entries rewritten: 'Noise index $N$', 'Stage T (tare)', 'Withheld' (N gate added). The fitted $A=a+b f_{neu}+c N$ tare is no longer named anywhere in the glossary: 'GSE250556' and 'Within-person SD' no longer quote the development-run-3 results that used it, and the 'PROC- and DEV- identifiers' example is now DEV-NOISE-01 (noise index). Open item: p4_12 l.125/133, p4_13 l.93 and p4_17 l.35/55/66 at 3290e60 still describe the fitted noise-corrected tare (not owned).

## 4. Open items

1. 23 of 133 chapter chunks returned no extracted terms (model budget exhausted); new terms there were found only by the acronym sweep, headword search and (p5_02 propulsion section) by hand: p0_giants 1-193, p1_02 418-614 and 810-970, p1_04 1-162, p2_01 213-402, p2_02b 1-155, p2_03 378-941, p2_03a 1-200, p2_07 204-411, p2_15b 1-199, p4_16a 1-210, p5_04 190-259, p5_05 1-186, p3_09 1-125, p5_07 1-288, p5_09 1-146, p5_02 1-362, p5_11 1-120, p5_10 1-37. A later pass may add entries.
2. The 515 carried glossary entries were checked by number tracing, retired-word scan and pointer recomputation, not re-read sentence by sentence against every rewritten chapter.
3. 10 iam.bib keys orphaned by the x_qp restore (2.2): lead to decide.
4. main.tex part titles (lines 18, 50) and app_E section titles (lines 47, 164) contain the words 'Informational Actualization' as part titles, not as an expansion of IAM; p2_16 line 1 comment names the source paper title. Not owned, not touched.
5. Rezzolla2018 is no longer cited by the glossary (TOV entry follows p2_01) but is still cited in p4_11.
6. The 854 citation contexts with shingle overlap < 0.25 were not reviewed one by one; a heavily paraphrased carriage there could still lack its citation.
