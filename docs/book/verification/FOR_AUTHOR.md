# FOR_AUTHOR

Items a check raised that would change a result, the framing of a claim, or that need a source or a script before they can be checked. The book is unchanged for each open item.

Open items: 2 (both are development measurements still in the book; author to decide). Resolved 2026-10-04: 25.

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


## Resolved 2026-10-04

- 3. `docs/book/part2/p2_02b_virial_tests.tex:L31` - DESI per-bin errors printed as 9-23 % (BGS widest), from the committed DESI 2024 V table in verify_shapefit_chi2.py
- 4. `docs/book/part2/p2_05_dual_sector_note.tex:L154 (and p2_07_late_time_growth.tex:L107)` - Limber estimate committed (docs/verification/scripts/verify_limber_lensing.py + output): 0.04-0.24 % for 30 <= L <= 1000, most at low L; p2_05 and p2_07 corrected, four checks added; the 0.08 % in p2_09/p2_16 is the scale-free estimate already checked
- 15. `docs/book/part4/p4_21_firstreadings.tex:L59-L60 (and Biological_Physics/MethylPhys/doors/PROC_DNMT_01_PARTA_OUTCOME.md)` - chapter retired (development readings moved to development/, author 2026-10-04)
- 17. `docs/book/figscripts/fig_p3.py (fig_holding_energy) vs docs/book/part3/p3_08_one_gauge.tex:L143 and L154` - figure aligned with Goyal et al. 2006 (30-40-fold, abstract checked); label overlap fixed
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
