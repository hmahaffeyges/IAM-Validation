# DEV-NOISE-02 — what the whole-blood neutrophil Met-A is reading (development, after looking; 2026-10-01)

**Data.** The 495 COVID-19 whole bloods (GSE179325) where chain v3 reported a neutrophil Met-A. Noise index N comes from the 48,528 frozen invariant sites, through the chain's Stage 1.

**Finding 1: in healthy blood, the reading is mostly instrument and composition.** Among the 76 NEGATIVE adults, untared A ≈ 0.60 + 0.205·f_neu + 2.51·N,
**R² = 0.85**. Untared A tracks N at ρ = 0.79. Only 33 of 495 arrays have N inside the Salas floor arrays' range (0.122–0.149). This lab's median is 0.19.

**Finding 2: with both removed, severe COVID-19 neutrophils read Normal.** Each array was divided by the A expected for its own f_neu and N, fitted on the
other NEGATIVE arrays (leave-one-out):

| group | n | median | above 1.05 | in Normal |
|---|---|---|---|---|
| NEGATIVE | 76 | 0.998 | 2 | 73 (96 %), SD 0.022 |
| MILD | 309 | 1.002 | 15 | — |
| SEVERE | 110 | 1.006 | 2 | — |

SEVERE vs NEGATIVE above Normal: p = 0.81. After the correction, the reading no longer depends on fraction (ρ = −0.02).

**What it means.**
1. The correction makes the healthy gauge tight (96 % Normal, SD 0.022) in a noisy lab. It is a candidate for fixes 1 and 2.
2. On this gauge, severe COVID-19 does not move the neutrophil's identity pattern. Every earlier "signal" in T4 came from array noise plus composition.
   This doesn't say the neutrophils are unchanged. It says that whatever changes in them doesn't change the identity sites' β, which is what Met-A reads.
3. Simulated 2 % damage moves the gauge. A real biological change in neutrophils that moves it has not been shown yet. That is the open question for neutrophil Met-A.

**Limits.** The correction was fitted and evaluated after seeing these data. It needs a held-out lab before it goes into the chain, and that
lab needs ≥ 20 healthy references to fit two terms. N could absorb a disease effect if disease shifted the invariant sites; SEVERE N is 0.195 vs NEGATIVE 0.190.
