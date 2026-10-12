# DEV-FINGERPRINT-02 — outcome (scored 2026-10-11 by rule 02b; development)

Each pair scored by `data/DEV_FINGERPRINT_02/score_fingerprint_02b.py score <pair>`, which refused unless the pair's seal was committed unchanged and
its own sha256 matched the seal. Table written by `outcome_02.py` from the committed `scored_<pair>.json`. Pairs are added as they are scored.

| pair | cancer vs normal | Arm B excess per cancer file | bar | min excess / donor excess | verdict (Arm B) | Arm A, reported only | readable share, cancer / normal |
|---|---|---|---|---|---|---|---|
| liver | HepG2 vs hepatocyte | 0.0113 to 0.0166 | 0.0211 | 0.53 | **no fingerprint** | IAM-A_rel 0.9614, Met-A_rel 1.0371, curve B 1.0000, z 0.26 (no fingerprint) | share read 3.34 |
| lung_alveolar | A549 vs epithelial cell of alveolus of lung | 0.0067 to 0.0068 | 0.0211 | 0.32 | **no fingerprint** | IAM-A_rel 0.7667, Met-A_rel 1.0487, curve B 1.0000, z 0.24 (no fingerprint) | share read 3.29 |
| lung_bronchial | A549 vs bronchial epithelial cell | 0.0087 to 0.0092 | 0.0211 | 0.41 | **no fingerprint** | IAM-A_rel 0.8404, Met-A_rel 1.3219, curve B 1.0000, z 3.84 (FINGERPRINT) | share read 2.77 |
| prostate | LNCaP clone FGC vs epithelial cell of prostate | 0.0488 to 0.0803 | 0.0211 | 2.31 | **FINGERPRINT** | IAM-A_rel 0.8596, Met-A_rel 1.3287, curve B 1.0000, z 3.09 (FINGERPRINT) | share read 4.34 |

**Reading.** Arm B, the arm that decided DEV-FINGERPRINT-01 (LNCaP against PrEC, excess +0.1105 to +0.1130), reads **FINGERPRINT for LNCaP
again, in a second laboratory** (ENCODE HAIB, its own normal prostate epithelium): every one of 6 LNCaP files above the bar, the smallest 2.31 times the
healthy donor-to-donor excess. It reads no fingerprint for HepG2 and A549: their cancer excess lies inside the range two healthy donors reach
against each other. The donor range comes from one donor pair and is a lower bound. The run-loss meter detects about 0.023 of
molecules losing every methylated call (planted tests), so these cancer lines show at most that much run loss beyond donor variation.
Arm A is not a verdict (donor null: healthy donors sit up to 0.19 off curve B). The cancer lines read more fully methylated multi-CpG molecules
than their normals (readable share above 2.7 times), recorded without interpretation.
Not read: the breast pairs (MCF-7 held for the author's decision on the breast_mcf10a seal).

## Reviews (one cancer at a time; `score_one.sh` refuses to score while a scored pair has no review)
**Reviewed:** scored_prostate.json. Arm B FINGERPRINT as sealed; LNCaP also gains methylation in normally unmethylated territory (why/both_directions_prostate.csv).
**Reviewed:** scored_liver.json, scored_lung_alveolar.json, scored_lung_bronchial.json. Arm B no fingerprint as sealed. Measured after the fact (why/both_directions_*.csv; post hoc, not a test): HepG2 and A549
carry a large GAIN excess (normally unmethylated molecules made methylated), 10-100x the healthy files; Arm B reads loss only. Their loss excess
sits below Arm B's bar, which is not detected, not absent: whether a loss fingerprint is absent in these lines or below what Arm B can detect is open. These three were scored in one batch;
had liver been reviewed first, A549 would have stayed unread for a sealed gain test. HepG2 and A549 are development evidence for gain, not a test of it.

### breast_hmec (MCF-7 against mammary epithelial cells), scored 2026-10-11 as sealed (rule 02b and Arm G, rule 02c)
**Reviewed:** scored_breast_hmec.json, scored_gain_breast_hmec.json.
- **Arm G (gain): FINGERPRINT.** All 10 MCF-7 files above the sealed bar 0.0048 (gain 0.0454-0.1559). First sealed test of the gain meter on a
  cancer it had not seen: it detects.
- **Arm B (loss): UNDECIDED,** files on both sides of the bar 0.0211 (3 of 10 above). Arm A, reported only: Met-A_rel 1.0667, IAM-A_rel 0.6891
  (IAM-A below 1, as the direction audit expects when gained molecules enter Stage Q).
- **Descriptive (not tested; `data/DEV_FINGERPRINT_02/why/mcf7_experiments.py`, output `why/mcf7_experiments_output.csv`):** the 10 files are 5
  ENCODE experiments, MCF-7 from different stocks (UW, Duke, Stanford; two shRNA lines, one a CTCF knockdown). The two files with the lowest gain
  and the highest loss are both replicates of ENCSR605IIO (the UW stock, 1 h estradiol): stock differences, not noise. Nothing in this pair breaks
  a sealed rule or the scorer, so scoring may continue (breast_mcf10a next).

### breast_mcf10a (MCF-7 against MCF 10A), scored 2026-10-11 as sealed (rule 02b and Arm G, rule 02c)
**Reviewed:** scored_breast_mcf10a.json, scored_gain_breast_mcf10a.json.
- **Arm B (loss): FINGERPRINT,** all 10 MCF-7 files above the bar 0.0211 (excess 0.1038-0.2982).
- **Arm G (gain): FINGERPRINT,** all 10 above the bar 0.0038 (gain 0.0126-0.0536). Arm A, reported only: Met-A_rel 0.8964, IAM-A_rel 0.7618.
- **Descriptive (`why/breast_two_refs.py`, output `why/breast_two_refs_output.csv`):** against the immortalised non-cancer line MCF 10A, MCF-7's
  gain is smaller than against primary mammary epithelial cells (0.0454-0.1559) but still above the bar in every file: culture adds some gain, not
  all of it. Loss depends strongly on the reference (0.0125-0.0429 against mammary epithelial cells, 0.1038-0.2982 against MCF 10A): the choice of
  healthy reference matters for the loss arm. The UW stock (ENCSR605IIO) again reads the most loss and the least gain. No sealed rule or scorer
  problem: scoring may continue (T47D pairs next, when aligned).

### t47d_hmec (T47D against mammary epithelial cells), scored 2026-10-11 as sealed (rule 02b and Arm G, rule 02c)
**Reviewed:** scored_t47d_hmec.json, scored_gain_t47d_hmec.json.
- **Arm B (loss): FINGERPRINT,** all 4 T47D files above the bar 0.0211 (excess 0.0725-0.1180).
- **Arm G (gain): FINGERPRINT,** all 4 above the bar 0.0048 (gain 0.0526-0.0843). Arm A, reported only:
  Met-A_rel 1.2136, IAM-A_rel 0.8287.
- Both breast cancer lines now read FINGERPRINT on the gain meter against primary mammary epithelial cells; T47D also on the loss meter,
  where MCF-7 was undecided. No sealed rule or scorer problem: scoring may continue (t47d_mcf10a next).

### t47d_mcf10a (T47D against MCF 10A), scored 2026-10-11 as sealed (rule 02b and Arm G, rule 02c)
**Reviewed:** scored_t47d_mcf10a.json, scored_gain_t47d_mcf10a.json.
- **Arm B (loss): FINGERPRINT,** all 4 above the bar 0.0211 (excess 0.1888-0.2642).
- **Arm G (gain): FINGERPRINT,** all 4 above the bar 0.0038 (gain 0.0179-0.0259). Arm A, reported only:
  Met-A_rel 1.0007, IAM-A_rel 0.9161.
- As with MCF-7, gain is smaller against MCF 10A than against primary mammary epithelial cells and loss is larger: the same reference pattern.

### Breast, all four sealed pairs scored (2026-10-11)
| pair | Arm B (loss) | Arm G (gain) |
|---|---|---|
| MCF-7 vs mammary epithelial cells | UNDECIDED (3/10) | FINGERPRINT (10/10) |
| MCF-7 vs MCF 10A | FINGERPRINT (10/10) | FINGERPRINT (10/10) |
| T47D vs mammary epithelial cells | FINGERPRINT (4/4) | FINGERPRINT (4/4) |
| T47D vs MCF 10A | FINGERPRINT (4/4) | FINGERPRINT (4/4) |
Development results on cell lines (one laboratory per line, ENCODE HAIB); they extend no commissioned scope.

**Milestone:** listed in `Biological_Physics/README.md`, Advancements, and in `README.md` and `FOLLOW.md` (all generated from `CANON/results_register.json`).

<!-- PLAIN:BEGIN (generated by CANON/results_to_tex.py from CANON/results_register.json; do not edit by hand) -->
**In plain words** (Cancer fingerprint replicated: prostate, a second laboratory, different cells): The same prostate test, repeated on data from a different laboratory, with different normal prostate cells as the comparison. Every cancer sample again carried the fingerprint (whole passages of the switch pattern wiped out), and even the weakest one stood well clear of the difference between two healthy people. A result that repeats in a second laboratory, on different cells, is much harder to explain away.

**In plain words** (Cancer fingerprint in breast cancer cell lines): Two breast cancer cell lines, each compared with two kinds of normal breast cells, one comparison at a time, each sealed before it was read. Cancer can damage the pattern in two directions: switches lost where they should be on, and switches added where they should be off. The meter for added switches found the breast cancer fingerprint in every comparison; the meter for lost switches found it in most, and one was undecided.
<!-- PLAIN:END -->
