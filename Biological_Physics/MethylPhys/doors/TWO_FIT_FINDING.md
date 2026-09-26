# The chain runs TWO deconvolutions, and the reported one is the pooled-first fit

**Found 2026-09-26**, while confirming the author's requirement that *"it should never be pooled into a class
before deconvolution."* Verified by running a real specimen and reading the numbers, not by reading source.

## What the deconvolver actually does

`walther_iam_deconvolver.deconvolve()` performs **two independent non-negative least-squares solves**:

| solve | reference | produces |
|---|---|---|
| 1 | `class_ref` — the **8 pooled class columns** | `class_fractions` |
| 2 | `celltype_ref` — the **114 cell-type columns** | `celltype_fractions` |

They are not related by summation. On one commissioning array, `class_fractions["immune"]` = **0.9151** while
the immune **cells** sum to **0.9902**.

**The per-cell A-scores are fine** — `stage_a_cells` scores each of the 115 cells on its own discriminative
markers against the H_min of its architecture class, which is exactly the architecture the author specified.
**But every *reported* composition number comes from solve 1**, the pooled fit: `stage_b_identity` consumes
`class_fractions`, and so do the composition guard, the tier, and every procedure sealed this week.

## The disagreement is systematic, and it is the collinearity

48 healthy whole-blood arrays across four laboratories:

| class | pooled fit | sum of cells | median difference | max |
|---|---|---|---|---|
| **immune** | 0.8233 | **0.9136** | **+0.0847** | 0.2394 |
| **progenitor** | 0.1565 | **0.0864** | **−0.0632** | 0.2280 |
| **stem_adult** | 0.0173 | 0.0000 | −0.0173 | 0.0800 |
| all others | 0.0000 | ≤0.0003 | 0.0000 | ≤0.0292 |

**The pooled fit moves roughly eight percentage points of immune mass into progenitor and stem_adult.** That
is precisely what the measured collinearity predicts: at the pooled level, progenitor vs immune correlate at
**r = +0.958** and stem_adult vs progenitor at **r = +0.989**, so the 8-column fit cannot tell them apart and
splits the mass. The 114-column fit, which has 51 distinct immune profiles to work with, does not face that
degeneracy.

**This also explains the §108 rule.** The chain pools progenitor and stem_adult into a "joint haematopoietic
component" on whole blood because they are individually unidentifiable *in the pooled fit*. That workaround
exists to paper over a degeneracy that the pooled fit creates and the cell-level fit does not.

## What it changes

The composition guard's `foreign_fraction` = 1 − (immune + progenitor + stem_adult), computed both ways on
the same 48 arrays:

| | median | max | withheld at the commissioned 0.0207 |
|---|---|---|---|
| pooled fit (today) | 0.0000 | 0.0433 | **2 of 48** |
| sum of cells | 0.0019 | 0.0213 | **1 of 48** |

So the guard's behaviour changes. Nothing is altered on that basis here: this is a measurement, and switching
the reported composition to the cell-level solve changes numbers in sealed procedures, so it requires its own
pre-registration with an invariance bar rather than an edit.

## The honest status

- **The author's architecture is correct and is already implemented for scoring.** Cells are found first, then each is scored against its class's floor.
- **It is not implemented for the reported composition**, which still comes from a pooled-first fit.
- **The cell-level solve is measurably less degenerate**, and is the one that should be reported — but that is a change to commissioned output and needs a procedure, not a patch.
