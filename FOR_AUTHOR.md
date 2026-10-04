# FOR_AUTHOR

Items a check raised that would change a result, a prediction, a locked value, an equation of IAM or the framing of a
claim. The book was left unchanged for each; the decision is the author's. Every FAIL of `verify_book.py` is listed here.

Items: 1

## 1. `docs/book/part2/p2_01_blackholes.tex:L76`

- **Now:** caption says 'CODATA 2018; M_sun = 1.98847e30 kg'
- **Proposed:** either keep 1.98847e30 and drop the implication that it follows from CODATA 2018, or use GM_sun(IAU 2015 nominal) / G(CODATA 2018) = 1.98841e30 kg
- **Why it matters:** 1.98847e30 is GM_sun / G with the CODATA 2014 G (6.67408e-11); with CODATA 2018 G (6.67430e-11) it is 1.98841e30. The difference (3e-5) changes no printed digit in the table, so this is wording/consistency only.
- **Recommendation:** low priority; no printed result changes

