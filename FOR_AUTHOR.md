# FOR_AUTHOR

Items a check raised that would change a result, a prediction, a locked value, an equation of IAM or the framing of a
claim. The book was left unchanged for each; the decision is the author's. Every FAIL of `verify_book.py` is listed here.

Items: 5

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

## 5. `docs/book/part2/p2_01_blackholes.tex:L76`

- **Now:** caption says 'CODATA 2018; M_sun = 1.98847e30 kg'
- **Proposed:** either keep 1.98847e30 and drop the implication that it follows from CODATA 2018, or use GM_sun(IAU 2015 nominal) / G(CODATA 2018) = 1.98841e30 kg
- **Why it matters:** 1.98847e30 is GM_sun / G with the CODATA 2014 G (6.67408e-11); with CODATA 2018 G (6.67430e-11) it is 1.98841e30. The difference (3e-5) changes no printed digit in the table, so this is wording/consistency only.
- **Recommendation:** low priority; no printed result changes

