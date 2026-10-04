# Contributing

IAM is meant to be tested. The most useful contribution is an honest attempt to break it.

## Found a problem in the book
1. Run the checks: `python3 docs/book/verify_book.py` (or `--part N`, `--label L`, `--fails`).
2. If a check fails on your machine, or you think a derivation, number or sign is wrong even though its check passes, open an issue
   with the **"I tried to break it"** template. Give the equation or check label, what you got, and what you expected.
3. Corrections that hold up are made in the book and in its check, and the change is recorded in the commit message.

## Rerunning the work
- Book derivations: `docs/book/verify_book.py` (standard library plus numpy, scipy and sympy).
- Cosmology chains: input files and outputs are in `Cosmological_Physics/` (Cobaya with MGCAMB and modified CAMB).
- Book figures: each one is drawn by a script in `docs/book/figscripts/`.
- Cell instrument: in development; the procedure is in `Biological_Physics/MethylPhys/sop/`, and findings are logged in `development/`.

## Proposing a change
Open a pull request against `main` with a short description of what changes and why. A change to a result, a fixed constant
(`CANON/iam_canon.json`) or a conclusion is discussed in an issue first.

## Working together
Researchers who want to collaborate on any part of the program — cosmology, quantum devices, or the cell instrument — are welcome.
Open an issue titled "Collaboration" and say what you would like to work on.
