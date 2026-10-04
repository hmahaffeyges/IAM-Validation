# FOR_AUTHOR

Items a check raised that would change a result, a prediction, a locked value, an equation of IAM or the framing of a
claim. The book was left unchanged for each; the decision is the author's. Every FAIL of `verify_book.py` is listed here.

Items: 11

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

