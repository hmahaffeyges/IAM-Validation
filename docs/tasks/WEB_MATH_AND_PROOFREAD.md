# Task: web-edition math errors, long equations, and a proofreading pass

Branch off `main`, open one pull request, do not merge it. The author reviews it.

## 1. The ones the author found (formula sheet, Appendix D, web edition)
- **D.36, D.55, D.252** show math input errors (D.252 has three).
- **D.358, D.414**: the equation is too long and runs over itself; it must wrap onto two lines.
- **D.558**: the description sentence runs into the equation number.

Note: a citation inside math (`\text{\cite{...}}`) was already moved out of D.37 and the DESI w0/wa block in commit c09b833c; check
whether D.36 is still wrong after that deploy before changing it.

## 2. Find every other one, book-wide
Build the site and scan the LaTeXML output for every failure: `ltx_ERROR`, `ltx_math_unparsed`, unknown-macro warnings, and math
rendered as raw TeX. Also find equations wider than the text column (check the 600 px and 1200 px widths), and any equation number
overlapping text. List them all in the pull request with the source file and line.

## 3. How to fix
- Fix at the **source** (`docs/book/...`) or in `website/latexml/*.ltxml` bindings. Prefer a binding when one macro breaks many places.
- **Never change the math itself: no value, sign, symbol or exponent.** Line breaks (`\begin{aligned}`/`multlined`), spacing, moving a
  citation or a comment out of math into the sentence, and replacing a macro LaTeXML does not know are fine.
- The formula sheet `docs/book/appendices/app_E_formulas.tex` must be edited **by targeted replacement only**. Do **not** rerun
  `figscripts/make_app_E.py`: it would overwrite about 98 hand fixes. Keep the chapter equation and its formula-sheet entry identical.
- The PDF must still compile (tectonic) with 0 errors and look the same apart from line breaks.
- `python3 docs/book/verify_book.py` must still report 0 FAIL. If a fix moves a line, update the check's `line=` (the script reports it).

## 4. Proofreading pass (whole book)
Spelling, grammar, punctuation, repeated words, inconsistent spelling of the same term, broken or `??` references, figure and table
captions that do not match their figure. Fix these directly.

Rules:
- Do not change any number, result, conclusion, status label or the meaning of a sentence. Anything like that goes in the PR
  description under **For the author**, with a recommendation.
- The tone is the author's: confident and positive. Do not add hedges, caveats or criticism of IAM.
- No population/cohort wording. Do not add names of companies, products or places that are not already in the book.
- One commit per Part so the review is easy.

## 5. Deliver
The PR description lists: every web error found and how it was fixed; every proofreading change (file, line, before, after);
the For the author list; verify_book.py summary; PDF compile result; screenshots of D.252, D.358 and D.558 after the fix.
