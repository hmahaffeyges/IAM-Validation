# IAM's Law and Order

*IAM's Law and Order: The Actualization of Reality — The Cost of Recording It, and the Price to Maintain It* — Heath W. Mahaffey.
This folder is the master copy of the book.

## Build
Compile `main.tex` with XeLaTeX or pdfLaTeX and BibTeX (`iam.bib`, style `iamnat.bst`), or `tectonic -X compile main.tex`.

## Check the derivations
`python3 docs/book/verify_book.py` from the repository root. Every derivation and calculated number in the book is checked and
passes. Options: `--part N`, `--label L`, `--fails`, `--json`, `--controls`. Its inventory, outputs and source files are in `verification/`.

## Layout
| Folder or file | Contents |
|---|---|
| `main.tex`, `preamble.tex` | the whole book; packages, title and the status labels (DERIVED, CALCULATED, CALIBRATED, MEASURED, OBSERVED, FITTED, CONJECTURE, PREDICTION, OPEN) |
| `part0/` | preface, abstract, the giants and how to read the book |
| `part1/` | Part I — Introduction to IAM's Law and the Virial Theorem |
| `part2/` | Part II — Informational Actualization of General Relativity: The Cosmological Dynamic; Part III — Black Holes and Horizons; Part IV — Particles and Quantum Records |
| `part3/` | Part V — The Quantum Order of Informational Actualization: Qubits and Semiconductors |
| `part4/` | Part VI — Cellular Physics: Thermodynamics of the Methylome |
| `part5/` | Part VII — Thirty-Seven Orders of Magnitude and the Web that is IAM |
| `appendices/` | constants (generated from `CANON/iam_canon.json`), formulas, notation, glossary, predictions register, provenance |
| `figures/`, `figscripts/` | every figure in the book, and the script that draws each one |
| `tables/` | generated tables |
| `verify_book.py`, `verification/` | the derivation check, its inventory, outputs and source files |

The folder names `part0`–`part5` are kept so links stay stable; the table gives the Part each holds.

Development work on the cell instrument is logged in [`development/`](../../development/); the book carries results once the chain
that produced them is commissioned.
