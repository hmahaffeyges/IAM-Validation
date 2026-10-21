# DEV-SELFTARE-03 - outcome: does the self-tare remove a real loss, and can it keep one? (exploration, 2026-10-21, not sealed)

**Not a milestone:** an exploration of the commissioned self-tare; no commissioned reading changes.

**The question.** Stage T step 1 (self-tare II, adopted by DEV-SELFTARE-02) maps each array onto the reference scale by its own fixed-site
anchors. A loss of a cell's pattern at every site, anchors included, has the same form on one array as a technical squeeze, so the map can
remove it (synthetic: DEV-CELLS-01). Every commissioning test planted loss at the identity sites only. Asked here on real arrays, with a
candidate fix: a **relative** self-tare that maps type II onto the array's own type I scale (removing only the difference between the two
probe designs) and leaves what both designs share to the median tare.

**Readings** ([`selftare_relative_explore.py`](data/DEV_SELFTARE_03/selftare_relative_explore.py), on GitHub; scored by
[`selftare03_score.py`](data/DEV_SELFTARE_03/selftare03_score.py), [`selftare03_summary.txt`](data/DEV_SELFTARE_03/selftare03_summary.txt)):

| self-tare, then the median tare | replicates: within-person SD (bar <= 0.020) | replicates in Normal (bar >= 95 %) | other-laboratory neutrophils in Normal (bar: all 49) | Group-A, 1 % loss at every site | Group-A, 1 % loss at identity sites |
|---|---|---|---|---|---|
| as commissioned (absolute) | 0.0158 | 62/63 | 49/49 | -0.0001, not found | +0.0167, found |
| relative | 0.0296 | 57/63 | 48/49 | +0.0139, found | +0.0174, found |
| none | 0.0367 | 48/63 | 42/49 | +0.0150, found | +0.0158, found |

**What it shows.**
- \measured The commissioned self-tare removes a 1 % loss of the neutrophils' pattern that reaches every site, on 55 real healthy bloods
  (GSE318173), and keeps the same loss at the identity sites. The synthetic finding holds on real arrays.
- \measured The relative self-tare keeps the loss but fails two of the bars that adopted the commissioned one: arrays vary one by one in a
  way both probe designs share, and only the anchor map removes it. It is not adopted. The commissioned self-tare and every commissioned
  reading stay as they are.
- \derived On one array a shared technical squeeze and a loss at every site have the same affine form; the anchors alone cannot separate them.
  What separates them is what the self-tare divides by: the array's anchor range against healthy arrays on its own slide. On the replicates its
  spread is 0.012, so one array could show a loss at every site of about 5.7 % of the neutrophils' pattern, and a group of 55 about 0.76 %
  ([`anchor_range_spread.py`](data/DEV_SELFTARE_03/anchor_range_spread.py), [`anchor_range_spread_summary.txt`](data/DEV_SELFTARE_03/anchor_range_spread_summary.txt)).

**Next.** The anchor range read as its own companion reading (a loss at every site), beside the commissioned Met-A, labeled preliminary,
with its own detection limit; synthetic first through the chain, then the planted test on real healthy arrays.
