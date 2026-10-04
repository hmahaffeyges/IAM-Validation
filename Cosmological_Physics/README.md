# Cosmological Physics

IAM in gravitation and cosmology: the book's Parts II and III. The book (`docs/book/`) gives the physics and the results; this folder holds
the chains and data behind them.

| Folder | What it holds |
|---|---|
| [`mgcamb_validation/`](mgcamb_validation/) | Level 1 chains (modified-gravity CAMB): the eighteen converged Planck chains, their outputs and scripts |
| [`camb_validation/`](camb_validation/) | Level 2 chains (dual-sector perturbation CAMB), outputs and scripts |
| [`data/`](data/) | Pantheon+ supernovae and the distance covariance used by the background tests |
| [`tests/`](tests/) | The early cosmology tests run before the chains (growth, supernovae, Fisher forecasts) |
| [`results/`](results/) | Figures from those early tests |

Every number the book takes from these files is checked by [`docs/book/verify_book.py`](../docs/book/verify_book.py), which reads the
committed chain outputs here.
