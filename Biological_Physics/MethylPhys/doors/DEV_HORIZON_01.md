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
