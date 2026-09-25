# walther_clinical.py - the v1 conductor, retired 2026-09-25

1,769 lines. It was the first clinical conductor: its own intake, calibration, deconvolution, A-scoring,
tiers, trajectory, baseline persistence and a drop-and-run folder entry. `cpg_conductor.py` replaced it and
has said so in its own docstring since it was written.

**Why it is here rather than in the chain.** While it sat in `chain/`, every mention of its name was
ambiguous - a few were the live path and most were the superseded implementation. Its one live function,
`stage_8_dual_matching` (disease-pattern concordance), was extracted verbatim to
`chain/disease_matching.py`, which is where the conductor now looks. Nothing in the live chain reads this
file.

**What is only here.** The trajectory and baseline code (`_compute_trajectory`, `_save_baseline`,
`_load_prior_baselines`) and the `run_from_folder` drop-and-run entry have no equivalent in the current
chain. If serial draws ever arrive - see `doors/ENHANCEMENTS.md` A11 - this is the prior art for it, which is
why the file is retired rather than deleted.
