# IAM's Law and Order — the book (one book, five parts)

*IAM's Law and Order: How Decoherence Writes the Classical World, and the Energy That Holds It Against Thermal Noise* — Heath W. Mahaffey.
Working edition. This folder is the master copy: every chapter is edited here.

## Build
Overleaf: upload this folder (or the zip of it) and compile `main.tex` (XeLaTeX or pdfLaTeX; BibTeX with `iam.bib`).
Command line: `tectonic -X compile main.tex`.

## Layout
- `main.tex` — the whole book; `preamble.tex` — packages, status labels (`\derived \calc \calibrated \measured \observed \fitted \conjecture \analogy \prediction \openprob`), title.
- `part0/` preface, abstract, how to read.
- `part1/` Introduction to IAM's Law and the Virial Theorem.
- `part2/` Informational Actualization of General Relativity: the Cosmological Dynamic.
- `part3/` The Quantum Order of Informational Actualization: qubits and semiconductors.
- `part4/` Cellular Physics: Thermodynamics of the Methylome.
- `part5/` Thirty-Seven Orders of Magnitude and the Web that is IAM.
- `appendices/` constants (generated from `CANON/iam_canon.json`), corrections to the source papers, reproduction, glossary.
- `figures/<part>/` every figure the book includes; `figscripts/` the scripts that make them.
- `RETIRED_drafts_2026-10/` the earlier per-part draft folders, kept until the author confirms they can go.

## Status
What is done and what is left: `BOOK_TODO.md`.
