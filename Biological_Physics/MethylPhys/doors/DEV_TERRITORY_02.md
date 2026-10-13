# DEV-TERRITORY-02 — the territory contrast with two readings (development, 2026-10-13; NOT sealed)

v1 (DEV-TERRITORY-01) met its physics bar and failed on nuisances. v2 adds a second reading, the log-ratio L = ln eps_f - ln eps_s,
which a composition change cannot move, keeps the difference D, reads class levels by the median, and refuses a reading when more
than a quarter of sites sit at the ceiling. Six bars were committed with the script before it ran (`data/DEV_TERRITORY_02/territory_synth_02.py`).

## Outcome, run once: three bars MET, three NOT MET

From `territory_synth_02_summary.txt`. **Met:** the physics (the fragile/stable response ratio within 10 % of the parameter-free
4.158 at every shortfall), the additive offset (D does not move), and the declared blind spot. **Not met:** composition on L at -10 %,
noise on L, and the two-number signature at a 5 % shortfall.

**Why, from the numbers rather than a guess.** L's response to a maintenance shortfall is small by construction - the derivation gives
d times the gap between the two baselines - so at a 2 % shortfall it moves L by less than the cohort's own sampling noise, and the
measured 2 % move even came out with the wrong sign. Bars 2 and 4 compared nuisances against a signal the simulation could not resolve.
Where the shortfall is large enough to resolve, the signature matches the prediction with nothing fitted: -1.079 and -1.036 at 10 and
20 percent against -1.068. And the two readings fail in opposite places: an additive offset leaves D alone and wrecks L, a composition
change leaves L alone and moves D five times the signal.

**What this establishes and what it does not.** The pair can tell a maintenance shortfall from a composition change once the shortfall
is about 10 % or more. It cannot do that for small shortfalls at this depth, and in one person with realistic between-person spread
the smallest callable shortfall is about 12 % on either reading at 40,000 sites per territory. That makes it a group or a large-effect
instrument, not an early single-person one. No bar was moved; v3, if written, must raise precision rather than lower a bar.

**In plain words.** We built a second ruler that ignores changes in the mix of cells. It works, but only when the damage is large -
for small changes it is too faint to see over its own noise. The two rulers fail in opposite situations, which is why they are
worth having together.

**What it would take to finish:** a v3 with enough simulated people to resolve L at a 2 % shortfall, or bars restated at the
shortfall the instrument is meant for, decided before running; then one real array set with known composition.

**Synthetic test:** DEV_TERRITORY_02/territory_synth_02.py, DEV_TERRITORY_02/territory_synth_02_summary.txt

**Not a milestone:** a synthetic test with three of six bars not met.
