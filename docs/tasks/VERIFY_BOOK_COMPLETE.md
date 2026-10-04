> **Completed** (pull request #13, merged 2026-10-04). Kept as a record. The script is now `docs/book/verify_book.py`, and its
> companion files are in `docs/book/verification/`.

# Task: complete `verify_book.py` so it checks every derivation and number in the book

**Who runs this:** a Claude Code cloud session on this repository, started by the author. Do this task BEFORE `WEBSITE_BUILD.md`.
**How to deliver:** branch `verify-complete`, one pull request against `main`. Do not push to `main`.

## Where things are

- The book: `docs/book/main.tex` and the chapter files it `\input`s (`docs/book/part*/`, `docs/book/appendices/`).
- IAM's constants: `CANON/iam_canon.json`. Locked values (never change, never "correct"): mu0 = -0.136, beta_m = Omega_m/2 = 0.15765,
  H0 photon sector 67.16 and matter sector 72.26, every chain result, the Level 2b wording.
- The script: `verify_book.py` (repo root). One `@check(label=..., chapter=..., part=..., status=..., printed=..., tol=..., heavy=...)`
  function per item, in book order; `run(label=None, part=None)`; CLI `--label --part --list --json --fails`; pure Python + numpy, scipy,
  sympy; files read only through `load_data(relpath)` and listed in `DATA_FILES` (it must keep running in a browser via Pyodide).
- What is not yet checked: `VERIFY_BOOK_INVENTORY.md`, 3,407 items, each with its reason. Current run: 2,927 PASS, 0 FAIL.

## What to do

Work chapter by chapter in book order. For each inventory row marked "not run", read the sentence and its derivation in the chapter, then:

1. **"not yet run" / "not yet checked" (about 1,720 derived or calculated numbers) and "displayed equation, not yet checked" (107):**
   write a real check. Recompute the number from the book's own equation and IAM's constants (or the published inputs the chapter
   cites, written in the check with their source), or verify the equation symbolically with sympy. Every check must:
   - compute the value, never copy the printed number into the answer;
   - carry a negative control: the same check with the book's printed value (or the key coefficient) changed by 5 % must FAIL;
   - not be vacuous (both sides of an identity being the same expression, or a coefficient cancelling against its own reciprocal,
     does not count). Run the negative controls in CI-style before committing.
2. **"measured, source not named" (437):** if the value is a published measurement the chapter cites, write the published value and the
   reference (DOI) in the check and compare (status `observed`). If it comes from a file in this repository, find the file and
   check against it. If you cannot find any source, leave it "measured, source not named" and list it in `SOURCES_NEEDED.md`
   with chapter and line. Never invent a source.
3. **"not found in the files the chapter names" (235):** search the repository for the right file; check against it, or list it in
   `SOURCES_NEEDED.md`.
4. **"too few printed digits" (593):** check that the printed value is the correctly rounded form of the source value
   (tolerance = half the last printed digit).
5. Definitions, inputs and restatements stay "not run" with that reason; restated values point to the check they restate.

## When a check fails

- If the book's number is a rounding or transcription error and the correct value is already established elsewhere in the book or in a
  committed output, fix the book (change only that number) and list the fix in `CHANGES.md` (file:line, old -> new, why).
- If the fix would change a result, a prediction, a locked value, an equation of IAM, or the framing of a claim: do NOT change the
  book. Write it in `FOR_AUTHOR.md` (where, now, proposed, why it matters, recommendation).
- If the failure is your check's error, fix the check.

## Rules (hard)

- Do not change any locked value or any result. Do not rephrase the author's text except a number you are fixing as above.
- Introduce no product, project, vendor or place names that do not already appear in the book; no "population" or "cohort" language.
- Keep the browser-safe structure; heavy checks (chains, CAMB, the methylation chain) read committed outputs and print the rerun command.
- Regenerate `verify_book_output.txt`, `verify_book_output.json` and `VERIFY_BOOK_INVENTORY.md` from the final run.

## Done means

- Every inventory row is either PASS, FAIL (listed in FOR_AUTHOR.md), or "not run" for a stated reason that is not "not yet run" or
  "not yet checked".
- The PR description gives the final counts (PASS / FAIL / not run by reason), the number of book numbers fixed, the list in
  FOR_AUTHOR.md, and the number of entries in SOURCES_NEEDED.md.
- A random 5 % sample of the new PASS checks was re-read for vacuity; report how many were vacuous and that they were fixed.
