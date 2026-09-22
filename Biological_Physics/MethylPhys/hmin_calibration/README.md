# H_min calibration — the MCMC that produced the floors, published for inspection

Deposited, with a verified reproduction: [10.5281/zenodo.22905819](https://doi.org/10.5281/zenodo.22905819) — samplers, the 37 reference cells as a table,
the reproduction log and `reproduce.sh`. Re-running the calibration returns every floor inside its own
posterior standard deviation (largest difference 0.000245; R-hat < 1.001).

Every file here is the calibration behind the constants the chain divides by. If a reviewer wants to check
H_min, this is the directory.

| file | what it is |
|---|---|
| `gape_mcmc_g002.py` | **G-002** — the calibration that produced the eight methylation floors. emcee ensemble, 32 walkers, 5 independent chains, 500 burn-in and 5,000 production steps, SIGMA_A = 0.020, and its own reference database (`_RAW_DB`) of **37 published reference cells**, 4–6 per class, all FACS-sorted or microdissected (Roadmap/ENCODE/Lister). Every methylation chain converged with R-hat < 1.001. The immune floor moved 0.795 → **0.838889 ± 0.0012** during this calibration, when six immune cell types replaced neutrophils alone. |
| `gape_mcmc_g003b.py` | **G-003b** — the follow-on sampler. |
| `g003_mcmc_framework.py` | the framework G-003b runs on (added 2026-09-22; it had not been committed). |
| `gape_mcmc_g008.py` | **G-008**. |
| `gape_mcmc_e_a_bio.py` | the E/A_bio sampler. |
| `gape_mcmc_nbio_ordering.py` | the class **ordering** of the retired per-class n_bio (ρ = 0.905, p = 0.002). The ordering is all that was ever established; the absolute per-class values awaited a G-007 run that **was never made** (PROC-HISTORY-01), and the quantity itself is retired — superseded by the Mahaffey number M = 20.94, one number for the cell. See Issue 003 §5.0.4. |
| `gape_bootstrap_comparison.py` | the script that produced the comparison below (added 2026-09-22; the table was committed without it). |
| `bootstrap_vs_mcmc_comparison.tsv` | bootstrap against MCMC, 32 rows. |

## Read the comparison table's scope before citing it

`bootstrap_vs_mcmc_comparison.tsv` has 32 rows: **8 classes × 4 substrates — nucleosome occupancy, fuzziness,
WPS and fragment size.** **Methylation is not in it.** The title invites the opposite reading, so it is stated
plainly here: the eight methylation floors the commissioned chain actually uses were **not** cross-checked
against a bootstrap by this file. Their support is G-002's own convergence (R-hat < 1.001 on every methylation
chain) and the 37-cell reference database inside `gape_mcmc_g002.py`.

## What is not in this directory

The **posterior chains themselves do not exist**: the samplers hold their samples in memory and print the
posterior summaries — no run wrote an `.h5` or `.npy`, and there is no chain file to deposit anywhere. What
replaces a sample archive is a re-run: the deposit above carries `reproduce.sh`, and re-running the
calibration unmodified on 2026-09-22 returned all eight floors inside their own posterior standard
deviations (largest difference 0.000245, R-hat < 1.001 across five chains, thirteen seconds).
