# DEV-FINGERPRINT-02 TRIAL — production trial, NOT EVIDENCE

HepG2 (liver) and A549 (both lung pairs) were read on 2026-10-11 by the loss arm (DEV_FINGERPRINT_02_OUTCOME.md), and their gain values were then computed
by `DEV_FINGERPRINT_02/why/both_directions.py`, the same definition Arm G uses. **The outcome here was known before this run.** Nothing in this folder is a test
or evidence and nothing here may be cited as a result. Purpose: exercise the sealed gain pipeline end to end (planted test, seal, commit-gated score) on real
cancer files before the honest tests (MCF-7, T47D, never read). Scorer `score_gain_02c.py` is a byte-identical copy of `DEV_FINGERPRINT_02/score_gain_02c.py`;
`sealed_<pair>.json` are copies of the rule-02b seals (file lists only). Author decision 2026-10-11.
