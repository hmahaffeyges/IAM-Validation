# DEV-HORIZON-01 — is there a reader-independent edge of disorder? (conjecture test, simulated first, 2026-10-09)

**Source idea (large scale).** A horizon is a boundary past which information about what fell in cannot reach any outside observer.
**Cell question.** As loss rises, is there an edge past which a molecule carries no information about the state of its domain — the same
edge for every reader — or does information fade smoothly, with any "edge" set by the reader's own cutoff?
**Status.** CONJECTURE (cell-black-hole step, book saturation appendix). Simulation only; no data read.

**Model.** Two domains (normally methylated, normally unmethylated), 8 CpGs per molecule, 40,000 molecules, 60 divisions, loss raised by
s × (s = 1 healthy, 24 = both domains at β 0.5). Information = mutual information (bits) between a molecule's methylated count and its
domain (an optimal reader; 1 bit = fully informative). Compared with readers that keep molecules ≥ 60/70/80/90 % methylated.
Two holding mechanisms: independent sites, and cooperative holding (a site is held and restored better when its neighbours are
methylated; neighbour cooperativity of maintenance is established in the literature).

## Correction (same day): the sharp edge in the first version was a simulation artefact
The first table used synchronous updates with rates allowed to reach probability 1 (at s = 20 with cooperativity 4, loss = 1.0 and
restore capped at 1); sites then flip in lockstep, which produced the "collapse" and a negative neighbour correlation. Re-run with
random-order sequential updates and rates capped at 0.95:

| model | what happens as loss rises | edge independent of reader? |
|---|---|---|
| independent sites | information fades smoothly (1.00, 0.77, 0.47, 0.20, 0.04 bits at s 4, 8, 12, 16, 20) | no |
| neighbour cooperativity (linear, strength 1–8) | fades smoothly, slightly later; neighbour correlation +0.08 to +0.56; no hysteresis | no |
| domain-wide nonlinear feedback (Dodd/Sneppen-type, strength 20) | steeper fade (0.93, 0.73, 0.53, 0.34, 0.16 bits at s 12–22); **memory**: same loss gives β 0.85 from a methylated start, 0.68 from an unmethylated start (s 18); molecules all-or-none (intermediate-molecule share 0.17 vs 0.44) | no cliff, but history-dependent state |

**What survives.** No model gives a cliff in information that every reader sees. The horizon-like property that does appear is
**memory**: with domain-wide feedback, a domain pushed into disorder does not come back when the push is removed, and its molecules are
all-or-none. That is a measurable cell property.
**Revised prediction (CONJECTURE).** (1) In pure cells, the share of intermediate molecules at a given domain β is depleted below the
independent-site expectation. (2) Hysteresis: after a demethylating drug is withdrawn, an independent-site cell returns fully; a cell
with domain-wide feedback leaves some domains permanently demethylated. Tumour bulk tissue cannot test (1): a mixture of normal and
tumour cells also gives all-or-none molecules. Pure cells (cell lines, sorted cells) and drug-washout time series can.
**Next (simulate first).** The washout design: doses, time points and molecule counts needed to separate full return from memory.

## Simulation — drug washout (memory test), before any data
Cell held 60 divisions at its own loss rate (1× healthy, 8×, 14× = cancer-like), drug raises loss 5–10× for 8–16 divisions, then 32
divisions without drug. Same reading rules as above.
| holding | cell's loss | drug | before | end of drug | 32 div after | lasting deficit | molecules stuck < 25 % methylated |
|---|---|---|---|---|---|---|---|
| independent | 1×, 8×, 14× | any | — | — | back to start | 0.000–0.004 | ≤ 0.6 % |
| domain memory | 1× (healthy) | 10× / 16 div | 0.998 | 0.968 | 0.997 | 0.000 | 0 |
| domain memory | 8× | 10× / 8 div | 0.976 | 0.098 | 0.902 | **0.074** | **5.1 %** |
| domain memory | 14× | 10× / 8 div | 0.934 | 0.046 | 0.679 | **0.255** | **22.9 %** |
**Prediction (CONJECTURE).** With independent holding every cell returns fully within a few divisions of washout. With domain-wide
memory, cells whose own loss rate is raised (cancer lines) keep a lasting deficit carried by all-or-none molecules stuck unmethylated, while
healthy-like cells return fully. The deficit (0.07–0.25 in β) is far above measurement noise; power is not the limit.
**Confounds to design out.** Drug toxicity and clonal selection (a surviving subclone can look like memory: need cell counts/clonality),
and slow DNMT1 recovery (need ≥ 3 late time points showing a plateau, not a slope).
**Data to look for.** Decitabine/azacytidine washout time courses with ≥ 3 post-washout time points over ≥ 20 divisions, sequenced
(single-molecule), ideally in one cancer line and one non-cancer line.

## Data search and a confound found by simulation (2026-10-09)
**Washout data.** None found with methylation read after washout: Scelfo et al. 2024 (DLD1/HCT116 and RPE-1 DNMT1 degron with 4-day
washout) deposited Hi-C and ChIP only (GSE251932/4/5); the HCT116 DNMT1/UHRF1 degron series (GSE236026, WGBS days 0–12 in triplicate;
GSE278681) stop at depletion. Searches: GEO (washout/withdrawal/recovery/remethylation terms) and the literature.
**Can a depletion series test memory instead? No.** Simulated dispersion of each molecule's methylated count against the
independent-site expectation (1 = independent) over a DNMT1-off time course:
| day | independent sites, site-wise loss | domain memory (c = 20) | independent sites, **whole-strand** maintenance failure (30 % / 60 % per division) |
|---|---|---|---|
| 0 | 1.00 | 1.31 | 0.99 / 1.02 |
| 2 | 0.99 | 1.89 | 4.25 / 4.48 |
| 6 | 1.01 | 3.95 | 3.16 / 2.97 |
| 12 | 1.00 | 5.21 | 3.07 / 2.93 |
When DNMT1 is absent during a cell's S phase, the whole new strand goes unmethylated, so all-or-none molecules appear with no domain
memory at all. A depletion series cannot separate memory from strand-level failure; **only recovery after restoring the enzyme can**
(strand failure recovers; memory does not). The GSE236026 download was not made.
**Kept as the test.** A degron or non-toxic inhibitor (e.g. GSK-3484862) washout with single-molecule methylation at ≥ 3 late time points,
cancer line and non-cancer line. Not found public; the design is recorded for collaborators.
