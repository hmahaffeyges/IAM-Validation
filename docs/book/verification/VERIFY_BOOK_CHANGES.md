# CHANGES

Number fixes made to the book while completing `verify_book.py`. Only the number changed; no wording, no result, no
prediction, no locked value. Each fix is a rounding or transcription slip whose correct value is already established elsewhere in
the book or in a committed output.

Fixes: 0

| file:line | old -> new | why |
|---|---|---|

## Every book edit made and reverted on this branch

Edits that were made and later undone are listed too, so the branch history can be read against this file. Net change to the
book from these: none.

| commit | file:line | old -> new | why |
|---|---|---|---|
| `fcbfdf6` | `docs/book/part2/p2_12b_lambda_history.tex:92` | `3.2\times10^{-8}` -> `3.1\times10^{-8}` | drafted as a rounding fix (unrounded 3.1499e-8 rounds to 3.1e-8); the edit was swept into the ch:bhinfo commit by mistake |
| `4b97c8d` | `docs/book/part2/p2_12b_lambda_history.tex:92` | `3.1\times10^{-8}` -> `3.2\times10^{-8}` | reverted: 3.2e-8 is the committed 3.15e-8 (verify_cc_and_baryon_output.txt; Table lambda_numbers, p2_12_lambda.tex:394) rounded up, a rounding edge rather than a slip; left to the author in FOR_AUTHOR.md |
