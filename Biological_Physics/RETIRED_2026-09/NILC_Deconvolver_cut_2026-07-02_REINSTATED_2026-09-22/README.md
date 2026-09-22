# This cut was reversed. The live NILC deconvolver is in the chain.

`nilc_deconvolver-2.py` here is the July copy, kept as history. It was cut from the chain on 2026-07-02 (commit
c1be0c3) on the grounds that it collapsed on correlated blood mixtures and deleted correct calls.

**That reading was wrong, and the module's own docstring said so:** it marks the patients and classes where the
cell composition is *genuinely ill-defined by the atlas reference*. The divergence was a diagnostic to be read, not
a defect to be removed. It was rerun in September against the same arrays (PROC-NILC-01), vindicated, and wired
back in.

| what | where |
|---|---|
| the live solver | [`MethylPhys/chain/nilc_celltype_deconvolver.py`](../../MethylPhys/chain/nilc_celltype_deconvolver.py) |
| how it is called | `stage_2b_second_opinion` in [`MethylPhys/chain/cpg_conductor.py`](../../MethylPhys/chain/cpg_conductor.py) |
| the procedure | SOP section 9 |

It is compared with Walther's NNLS **at class level** — where the atlas is separable (PROC-SEP-03) — and reported as
an agreement flag, never as the composition the report stands on. Cell-level disagreement inside one lineage is
expected, because the atlas cannot split the blood classes.
