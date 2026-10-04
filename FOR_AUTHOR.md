# FOR_AUTHOR

Items a check raised that would change a result, a prediction, a locked value, an equation of IAM or the framing of a
claim. The book was left unchanged for each; the decision is the author's. Every FAIL of `verify_book.py` is listed here.

Items: 25

## 1. `docs/book/part2/p2_02_virial.tex:L86-87`

- **Now:** the reciprocal |U|/2T of these values lies between 0.77 and 0.91 within the virial radius
- **Proposed:** no change needed if the rounded summary 2T/|U| ~ 1.1-1.3 is intended; with the traced read-offs (Bett 1.2-1.3, Neto 1.12-1.26, Power 1.15-1.25) the reciprocal range is 0.77-0.89
- **Why it matters:** 0.91 = 1/1.1 comes from the one-decimal summary, not from the lowest traced value 1.12 (NBODY_TRACE.md)
- **Recommendation:** optional: print 0.77-0.89, or say the range is the reciprocal of the rounded 1.1-1.3; the check reproduces 0.91 from the rounded envelope

## 2. `docs/book/part2/p2_02_virial.tex:L143`

- **Now:** the coefficient of 1/a within 2 % and the constant within 7 %
- **Proposed:** leave, or 2.1 % and 7.3 %
- **Why it matters:** the unrounded fit is exp(0.927 - 1.021/a): 2.1 % and 7.3 %; the printed 2 % and 7 % follow from the two-decimal fit exp(0.93 - 1.02/a) the book quotes in ch:theory line 399, so 'within 2 %' is very slightly generous
- **Recommendation:** low priority; the check compares the two-decimal coefficients, as the book quotes them

## 3. `docs/book/part2/p2_02b_virial_tests.tex:L31`

- **Now:** errors $8$--$13\,\%$ per bin~\cite{DESI2024V}
- **Proposed:** name the f sigma8 values the range comes from, or print the range of the committed ShapeFit+BAO values (about 9-19 %, BGS the widest)
- **Why it matters:** the committed DESI DR1 f sigma8 tables (verify_shapefit_chi2.py, verify_sector_tension.py) put the BGS error at 19-25 %, outside 8-13 %
- **Recommendation:** check the per-bin errors in DESI 2024 V and either keep 8-13 % with the table it comes from, or correct the range

## 4. `docs/book/part2/p2_05_dual_sector_note.tex:L154 (and p2_07_late_time_growth.tex:L107)`

- **Now:** a Limber estimate lowers the lensing power by 0.05--0.3 % for 30 <= L <= 1000
- **Proposed:** commit the script that produces the L-dependent range, or print the range a committed script gives
- **Why it matters:** the range cannot be rerun from the repository; an Eisenstein-Hu linear-power Limber integral with the IAM growth (G_eff = mu G, same early amplitude) gives 0.24 % at L = 30 falling to 0.03 % at L = 1000, i.e. the opposite L ordering and a lower floor
- **Recommendation:** add the Limber calculation to verify_obs_chapters.py with its output, then check it in verify_book.py

## 5. `docs/book/part2/p2_08_s8_trend.tex:L101`

- **Now:** the low-redshift weak-lensing surveys sit 6--9 % below Planck
- **Proposed:** 7--9 %
- **Why it matters:** the cited surveys give 6.7 % (DES Y3 0.776) and 8.8 % (KiDS-1000 shear 0.759) below S8 = 0.832; 6.7 rounds to 7
- **Recommendation:** optional: print 7--9 %, or keep 6--9 % as a loose bracket

## 6. `docs/book/part2/p2_08_s8_trend.tex:L204`

- **Now:** the growth index moves 40 % of the way to the measured value
- **Proposed:** 39 %
- **Why it matters:** (0.5852 - 0.5543)/(0.633 - 0.5543) = 39.2 %; the check accepts 40 % as a round figure (tol 0.02)
- **Recommendation:** optional: print 'about 40 %' or '39 %'

## 7. `docs/book/part2/p2_12b_lambda_history.tex:92`

- **Now:** accumulated virial heat of baryons $3.2\times10^{-8}\,\rho_\Lambda c^2$
- **Proposed:** $3.1\times10^{-8}$ (or keep 3.2)
- **Why it matters:** The unrounded value is 3.1499e-8, which rounds to 3.1e-8; 3.2e-8 comes from rounding the committed 3.15e-8 (verify_cc_and_baryon_output.txt; Table lambda_numbers, p2_12_lambda.tex:394) a second time. No result changes; it is a two-digit rounding edge.
- **Recommendation:** Optional: print 3.1e-8, or 3.15e-8 as in the table. The check passes either way within 2 %.

## 8. `docs/book/part2/p2_13b_baryon_chain.tex:L53 and L57 (and L188)`

- **Now:** line 57 quotes Omega_Lambda = 0.6847 (Planck central value); the +0.79 % on lines 53 and 188 comes from ch:lambda (Eq. corr), which uses Omega_Lambda = 1 - Omega_m - Omega_r = 0.6846
- **Proposed:** no change needed; optionally say on line 53 that the offset uses the ch:lambda inputs (Omega_Lambda = 0.6846)
- **Why it matters:** with 0.6847 the same offset is +0.78 %, so a reader recomputing from the inputs printed in this chapter gets the last digit one lower
- **Recommendation:** leave as is, or add the input in a parenthesis; the checks use 0.6846 as ch:lambda does

## 9. `docs/book/part2/p2_13b_baryon_chain.tex:L56 (eq:bc_etaob)`

- **Now:** eta = 2.739e-8 Omega_b h^2 (Steigman 2006)
- **Proposed:** no change; the factor is reproduced from first principles only with T0 = 2.725 K and a mean baryon mass with Y_P ~ 0.24 (as in that paper); with today's T0 = 2.7255 K it is 2.737e-8
- **Why it matters:** all eta values in the chapter carry this factor; the 0.1 % difference does not change any statement
- **Recommendation:** optional footnote naming T0 = 2.725 K

## 10. `docs/book/part2/p2_01_blackholes.tex:L76`

- **Now:** caption says 'CODATA 2018; M_sun = 1.98847e30 kg'
- **Proposed:** either keep 1.98847e30 and drop the implication that it follows from CODATA 2018, or use GM_sun(IAU 2015 nominal) / G(CODATA 2018) = 1.98841e30 kg
- **Why it matters:** 1.98847e30 is GM_sun / G with the CODATA 2014 G (6.67408e-11); with CODATA 2018 G (6.67430e-11) it is 1.98841e30. The difference (3e-5) changes no printed digit in the table, so this is wording/consistency only.
- **Recommendation:** low priority; no printed result changes

## 11. `docs/book/part4/p4_12_instrument.tex:L105`

- **Now:** a second laboratory's purified neutrophils read 0.86--1.26 against the frozen reference with no tare
- **Proposed:** either state that the range is of the 33 arrays read in test T2, or give the range of all 48 arrays re-read in the diagnostic (0.86--1.29)
- **Why it matters:** PROC_NEUT_TEST_01_T2_OUTCOME.md gives 0.86-1.26 for the 33 arrays of the scored test; the committed diagnostic re-read of all 48 arrays (doors/data/t2_diag.csv, also noise_index.csv) reaches 1.286 (GSM7885063). The check reads 1.26 from the T2 record and passes; a reader rebuilding the range from the committed per-array file gets 1.29.
- **Recommendation:** low priority; name the 33 arrays or widen to 1.29

## 12. `docs/book/part4/p4_13_separation.tex:L44-46`

- **Now:** a solver built on references from several platforms under-read EPIC neutrophils by about 0.05 and split T cells into subtypes that have no purified EPIC profile, and the expectation built from its fractions was wrong
- **Proposed:** re-read against DEV_ATLAS_EPIC_01 (2026-10-03): on FACS-counted EPIC bloods the atlas solver does not under-read neutrophils (bias -0.007 to -0.013; NNLS8 -0.030), and on the six blood-like DNA mixtures every method reads low, the chain's own NNLS8 included (-0.053; atlas -0.043 to -0.047)
- **Why it matters:** The sentence frames the 0.05 under-read as a property of the multi-platform solver; the committed later record (doors/DEV_ATLAS_EPIC_01.md, results/metrics_by_set.csv) finds it is a property of the mixtures shared by every method. Changing it changes the framing of a finding, so it is not edited here.
- **Recommendation:** author to decide whether to keep the sentence as the history of the choice, or add the later measurement

## 13. `docs/book/part4/p4_13_separation.tex:L61 and L76`

- **Now:** shift for a 2 % loss in the 0.40-0.50 bin: 0.033
- **Proposed:** 0.032 if the bin is [0.40, 0.50) over the 656 arrays of doors/data/lowfrac_readings.csv
- **Why it matters:** The other three bins reproduce from lowfrac_readings.csv (0.0401, 0.0498, 0.0638); the first bin gives a median of 0.0321 with the same rule. Not in this batch's rows, so not checked or changed here; the binning of the original run may differ.
- **Recommendation:** check the bin edges used by DEV-LOWFRAC-01 before changing anything

## 14. `docs/book/part4/p4_16a_skytools.tex:L142-143`

- **Now:** smoothing over the 32 nearest pixels left a spread of 0.171, against 0.131 +- 0.001 for the same values shuffled across the sphere: a factor of 1.31, at 57 sigma
- **Proposed:** no change to the numbers (they match doors/PROC_CEIL_01_OUTCOME.md, finding 3); consider whether the measurement is now verified
- **Why it matters:** doors/REPORT_LINE_AUDIT_2026-09-26.md line 98 lists this line as 'UNVERIFIED | 2026-09-22 measurement, open item'; the only committed record is the outcome text, no data file or script output. The checks read the numbers from that text.
- **Recommendation:** low priority: either commit the measurement's output or mark the sentence's source

## 15. `docs/book/part4/p4_21_firstreadings.tex:L59-L60 (and Biological_Physics/MethylPhys/doors/PROC_DNMT_01_PARTA_OUTCOME.md)`

- **Now:** by 400 nM to 0.36--0.60 ... observed 2.66--2.85, against a ceiling of about 2.8
- **Proposed:** no change to the book; state in the Part A record which arrays the two ranges cover (and how the ceiling 1/H(floor) = 2.8 was computed, or commit the per-site floor of the three lines)
- **Why it matters:** In data/dnmt_arrays_readings.csv the active-drug arrays at >= 400 nM span methylated-site beta 0.34-0.91, not 0.36-0.60. The printed ranges match the 15 active-compound arrays (both compounds) with beta between 0.365 and 0.604 (A_meth 2.657-2.845), which leaves out NOMO-1 at 2,000 nM (beta 0.34, A_meth 2.61). The checks therefore read 0.36, 0.60, 2.66 and 2.8 from the record text, not from the per-array file; 2.85 is checked from the file.
- **Recommendation:** Add the array selection (e.g. 'arrays with methylated-site beta 0.36-0.60') to the Part A record, so the four numbers can be recomputed from the committed file.

## 16. `docs/book/part5/p5_05b_virial_partners.tex:L29`

- **Now:** The ratio of dark matter to dark energy density today --- approximately 0.38
- **Proposed:** keep 0.38 (it is 0.26/0.69 from lines 25-26, as verify_virial_papers.py prints), or print 0.39 if the ratio is meant from Planck 2018 directly
- **Why it matters:** From Planck 2018 (Omega_c h^2 = 0.1200, h = 0.6736, Omega_Lambda = 0.6847) Omega_c/Omega_Lambda = 0.386, which rounds to 0.39; the printed 0.38 is the ratio of the book's rounded 26 % and 69 %.
- **Recommendation:** No change needed for the check (it uses the book's 26 % and 69 %); the author may decide whether the sentence means the rounded or the full Planck ratio.

## 17. `docs/book/figscripts/fig_p3.py (fig_holding_energy) vs docs/book/part3/p3_08_one_gauge.tex:L143 and L154`

- **Now:** text and caption: DNMT1 preference 'about 30--40x' (Goyal 2006); the figure script plots this entry as 'several reports, 30-50x' (ln 50 = 3.9 kT)
- **Proposed:** use one range in both, as the cited paper gives it
- **Why it matters:** the plotted bar and the printed range differ (3.4-3.9 kT plotted vs 3.4-3.7 kT from the text); the overall range 1.9-4.4 kT is unaffected
- **Recommendation:** confirm the Goyal 2006 value and align the figure label and bar with the text (or the text with the figure).

## 18. `docs/book/part5/p5_11_status_all.tex:L57`

- **Now:** KiDS-Legacy S_8=0.815, 0.3sigma; DES Y3 3x2pt 0.776, 2.3sigma
- **Proposed:** compute both distances from the same IAM S_8: with the unrounded Level 2 chain value (0.8215 +- 0.0111) they are 0.33sigma and 2.24sigma (2.2); with the rounded 0.822 +- 0.011 they are 0.36sigma (0.4) and 2.27sigma (2.3)
- **Why it matters:** the two printed distances come from two different roundings of the same input (0.3 matches part2/p2_02b_virial_tests 0.33 from the unrounded chain; 2.3 matches verify_sector_tension_output.txt item 11, which uses 0.822)
- **Recommendation:** either print 0.3sigma and 2.2sigma (unrounded chain S_8, as p2_02b) or 0.4sigma and 2.3sigma (rounded 0.822, as the sector-tension script); both checks pass as written, each against its own input

## 19. `docs/book/appendices/app_C3_derivations.tex:L291, L297-L298`

- **Now:** Line 291 states Omega_Lambda = 0.6847; the +0.79 % (L297) and the exponent 0.521 (L298) are reproduced only with Omega_Lambda = 1 - Omega_m - Omega_r = 0.6846 (as ch:lambda Eq. lam_corr_num uses); with 0.6847 they come out +0.78 % and 0.5205.
- **Proposed:** Either name Omega_Lambda = 0.6846 (radiation included) for the two numbers on L297-L298, or keep as is; the checks app:derivations:L297:+0.79 and L298:0.521 use 0.6846 and pass.
- **Why it matters:** The appendix gives one Omega_Lambda (0.6847) two lines earlier; a reader recomputing with it gets 0.78 and 0.520.
- **Recommendation:** Optional wording: add 'Omega_Lambda = 0.6846' after '+0.79 %' or leave; no number change needed.

## 20. `docs/book/appendices/app_C3_derivations.tex:L302`

- **Now:** eta = 273.9e-10 Omega_b h^2 is used without a source; recomputing with m_p and T_CMB = 2.7255 K gives 273.4 (helium-weighted mean baryon mass gives about 273.7).
- **Proposed:** Cite the source of the 273.9 coefficient (big-bang nucleosynthesis review) at L302.
- **Why it matters:** The coefficient sets the 5.03e-10 and 6.08e-10 values; a 0.2 % convention difference does not change their printed figures.
- **Recommendation:** Add a citation; no number change.

## 21. `docs/book/appendices/app_F_glossary.tex:L68`

- **Now:** 55.78 and 71.12 with the Planck 2018 base values H_0=67.4, Omega_m=0.315
- **Proposed:** no change needed if beta_m stays the fixed 0.15765; if beta_m is meant to follow Omega_m = 0.315 (0.1575, as line 89 says), the matter rate is 71.11
- **Why it matters:** 71.12 needs beta_m = 0.15765 (67.4 sqrt(0.685 + 0.15765 e) = 71.123); with beta_m = 0.315/2 = 0.1575 it is 71.110, which prints 71.11. Line 89 says 0.1575 is written where Omega_m = 0.315 is used.
- **Recommendation:** Keep 71.12 (beta_m is fixed in every chain and never sampled) or state beside it that beta_m stays 0.15765; the check app:glossary:L68:71.12 uses the fixed beta_m and passes.

## 22. `docs/book/appendices/app_F_glossary.tex:438`

- **Now:** per-site efficiencies of about 0.95--0.98 per division
- **Proposed:** state the same range as line 328 and ch:landauer line 172 (0.90--0.98, Genereux2005), or cite the source of 0.95--0.98
- **Why it matters:** the glossary gives two different ranges for the same quantity (line 328: 0.90--0.98 with a citation; line 438: about 0.95--0.98 without one), and no repository file holds either
- **Recommendation:** author to choose the range and add the citation; both ends are in SOURCES_NEEDED

## 23. `docs/book/appendices/app_F_glossary.tex:446`

- **Now:** need 1-b=0.58+-0.04 against about 0.80 from simulations
- **Proposed:** keep the number; consider naming it as the baseline of the 2013 Planck SZ analysis, as ch:lensdyn line 184 does
- **Why it matters:** ch:lensdyn line 184 calls 1-b = 0.8 the baseline of the 2013 analysis and gives the simulation range as b of about 0.1--0.15 (1-b of about 0.85--0.90); the glossary attributes 0.80 to simulations
- **Recommendation:** wording check only; the check app:glossary:L446:0.80 passes against b = 0.2 (Planck 2015 XXIV)

## 24. `docs/book/appendices/app_G_predictions_register.tex:L53 (COS-172; generated from CANON/predictions_triage_2026-10-02.json by docs/book/figscripts/make_app_G.py)`

- **Now:** Compared with the three-way cluster sample (0<z<0.5): IAM R = 1.07-1.16
- **Proposed:** IAM R = 1.05-1.16 (1/mu over 0<z<0.5 runs from 1.055 at z = 0.5 to 1.158 at z = 0), or keep 1.07-1.16 and give the redshift range it belongs to (1.07 is 1/mu at z = 0.39-0.40)
- **Why it matters:** With the book's own mu(z), 1/mu at z = 0.5 is 1.055 (also printed on line 46 of the same appendix), so the lower end 1.07 does not match the stated range 0<z<0.5; the triage record itself notes the recomputed range 1.055-1.158. The check app:register:L53:1.07 FAILS until this is resolved.
- **Recommendation:** Change the range in the triage statement (or an app_G_overrides.json edit) and regenerate the appendix with make_app_G.py rather than editing the generated .tex by hand.

## 25. `docs/book/appendices/app_G_predictions_register.tex:L55 and L61 (COS-217, COS-272; generated from CANON/predictions_triage_2026-10-02.json)`

- **Now:** WtG observed 1.31 +/- 0.11 (IAM ~2.0 sigma below); CCCP observed 1.20 +/- 0.12 (~1.0 sigma)
- **Proposed:** Use the traced values recorded in docs/book/read_ledgers/st_MANIFEST_clusters_satellites.md (LD4): Planck 2015 XXIV Table 2 priors 1-b = 0.688 +/- 0.072 (WtG, ratio 1.45 +/- 0.15) and 0.780 +/- 0.092 (CCCP, ratio 1.28 +/- 0.15), with the sigma distances recomputed, or drop the numeric comparison
- **Why it matters:** The ledger and docs/verification/PAPER_ERRATA.md (LD4) say 1.20 +/- 0.12 and 1.31 +/- 0.11 were replaced by traced values in the lensing-dynamics chapter, but the register rows still print the untraced numbers; the sigma distances 2.0 and 1.0 (checked here from the printed inputs) would change (about 2.4 and 1.3 sigma with the traced values).
- **Recommendation:** Update the triage statements of COS-217 and COS-272 (or add overrides) and regenerate the appendix with make_app_G.py; the rows 1.31, 0.11, 1.20, 0.12 are listed in SOURCES_NEEDED until then.

