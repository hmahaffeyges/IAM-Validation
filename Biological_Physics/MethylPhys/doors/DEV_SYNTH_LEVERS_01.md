# DEV-SYNTH-LEVERS-01 — synthetic driven-bit molecules: do the levers move IAM-A as the physics says? (development, 2026-10-09)

**Model.** Molecules of 8 CpGs copied division after division: a methylated site is lost with probability f, an unmethylated site
restored with probability g (independent sites). Read with Stage Q's exact rule (≥ 6 calls, ≥ 80 % methylated, isolated loss with both
neighbours methylated), optionally with instrument errors. 200,000 molecules, 80 divisions, from fully methylated.

| test | result |
|---|---|
| 1. what the rule reads | depends only on g/f (f,g = 0.02/0.24 and 0.01/0.12 both ε 0.050; 0.005/0.24 and 0.01/0.48 both 0.018). E from ε = ln(g/f) + a computable offset (+0.26 at ln(g/f) 3.18; +0.46 at 2.49; +0.14 at 3.87) from the rule's conditioning |
| 2. SAM lever (g ∝ S/(Km+S), Km 4.4 µM, from 20 µM) | 2×, 4×, 8× lower SAM: IAM-A_rel 1.10, 1.26, 1.50; ΔE −0.13, −0.31, −0.56 kT vs plain ln(g ratio) −0.17, −0.43, −0.82 |
| 3. instrument (0.5 % false loss, 1 % non-conversion) | ε 0.0312 → 0.0339; subtracting the false-loss rate over-corrects (0.0289): correction must use the model, not subtraction |
| 4. divisions | from fully methylated, steady state in ≈ 1/(f+g) ≈ 4 divisions (ε 0.009, 0.023, 0.029, 0.031 at 1, 4, 8, 16) |
| 5. molecules needed | SD of ε 3.2 % (5,000 qualifying molecules), 1.3 % (20,000), 0.8 % (80,000) |

**Lessons, used before any real data.** (a) Predictions must go through the rule's own mapping (computed, not fitted); the plain
ln(g/f) overstates the SAM response by ~30–45 %. (b) Instrument correction from a spike-in must be model-based. (c) Adult cells are at
steady state within a few divisions, so the generation-series prediction is: **ε constant across generations while the rates are
constant**; drift means the rates changed. (d) Molecule counts do not limit any test (WGBS gives millions); the limits are instrument
and laboratory offsets and, for temperature, between-animal variance.
