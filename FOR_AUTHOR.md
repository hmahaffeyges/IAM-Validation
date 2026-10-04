# FOR_AUTHOR

Items a check raised that would change a result, the framing of a claim, or that need a source or a script before they can be checked. The book is unchanged for each open item.

Open items: 6 (sources or scripts the lead is supplying). Resolved 2026-10-04: 21.

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

## 17. `docs/book/figscripts/fig_p3.py (fig_holding_energy) vs docs/book/part3/p3_08_one_gauge.tex:L143 and L154`

- **Now:** text and caption: DNMT1 preference 'about 30--40x' (Goyal 2006); the figure script plots this entry as 'several reports, 30-50x' (ln 50 = 3.9 kT)
- **Proposed:** use one range in both, as the cited paper gives it
- **Why it matters:** the plotted bar and the printed range differ (3.4-3.9 kT plotted vs 3.4-3.7 kT from the text); the overall range 1.9-4.4 kT is unaffected
- **Recommendation:** confirm the Goyal 2006 value and align the figure label and bar with the text (or the text with the figure).

## Resolved 2026-10-04

- 12. p4_13 solver sentence - author 2026-10-04: no development findings added to the book
- 25. WtG/CCCP register rows - author approved 2026-10-04: traced Planck 2015 XXIV values (1.45 +/- 0.15, ~2.4 sigma; 1.28 +/- 0.15, ~1.3 sigma), as ch:lensdyn
- 1. `docs/book/part2/p2_02_virial.tex:L86-87` - printed 0.77-0.89 (traced read-offs)
- 2. `docs/book/part2/p2_02_virial.tex:L143` - kept (the book quotes the two-decimal fit)
- 5. `docs/book/part2/p2_08_s8_trend.tex:L101` - printed 7-9 %
- 6. `docs/book/part2/p2_08_s8_trend.tex:L204` - 'about 40 %'
- 7. `docs/book/part2/p2_12b_lambda_history.tex:92` - printed 3.15e-8, as in the lambda table
- 8. `docs/book/part2/p2_13b_baryon_chain.tex:L53 and L57 (and L188)` - input named (Omega_Lambda = 0.6846)
- 9. `docs/book/part2/p2_13b_baryon_chain.tex:L56 (eq:bc_etaob)` - kept (0.1 % convention difference, no statement changes)
- 10. `docs/book/part2/p2_01_blackholes.tex:L76` - M_sun = 1.98841e30 kg (IAU 2015 GM_sun / CODATA 2018 G); no table digit changes
- 11. `docs/book/part4/p4_12_instrument.tex:L105` - 'the 33 arrays of the scored test' named
- 16. `docs/book/part5/p5_05b_virial_partners.tex:L29` - kept 0.38 (ratio of the book's 26 % and 69 %)
- 18. `docs/book/part5/p5_11_status_all.tex:L57` - kept: each distance matches the chapter it summarises (0.3 sigma as p2_02b; 2.3 sigma as the sector-tension table)
- 19. `docs/book/appendices/app_C3_derivations.tex:L291, L297-L298` - input named (Omega_Lambda = 0.6846, radiation included)
- 20. `docs/book/appendices/app_C3_derivations.tex:L302` - Steigman2006 cited for 273.9
- 21. `docs/book/appendices/app_F_glossary.tex:L68` - 'with beta_m = 0.15765 fixed' added
- 22. `docs/book/appendices/app_F_glossary.tex:438` - 0.90-0.98 with Genereux2005, as line 328
- 23. `docs/book/appendices/app_F_glossary.tex:446` - 0.80 named as the 2013 Planck SZ baseline, as ch:lensdyn
- 24. `docs/book/appendices/app_G_predictions_register.tex:L53 (COS-172; generated from CANON/predictions_triage_2026-10-02.json by docs/book/figscripts/make_app_G.py)` - register range 1.05-1.16 via app_G_overrides.json, appendix regenerated; check passes
- 26. `docs/book/part2/p2_03_theory.tex:1085` - 'within about 2 %'
- 27. `docs/book/part2/p2_01_blackholes.tex:136 (caption of fig:smarr)` - '|Delta| <~ 2e-16'
