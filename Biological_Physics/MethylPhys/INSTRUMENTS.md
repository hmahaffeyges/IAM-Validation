# Candidate instruments, by priority

_Generated from `CANON/instruments.json` by `CANON/instruments_board.py`; do not edit by hand._

Candidate instruments: every reading we could build on the methylation channel, each with the term of the law it reads, its substrate, its blind spot, its advantage, what would break it, and the next concrete step. The point is NOT to pick one - several instruments with different blind spots, agreeing on the same sample, is worth more than one instrument we trust. Priority is by (decides something) x (free or cheap) / (blocked).

## 1. Clustering meter (C-score): finish its commissioning

- **Reads:** whether losses are CLUSTERED (whole stretches) rather than scattered - the shape of the loss, not its size
- **Substrate:** arrays (EPIC v1 now; 450K and EPIC v2 later) · **Cost:** free (a metadata search, then one laptop read)
- **Where it stands:** test (a) MET in two laboratories that had no part in setting the band (34/35 and 22/23 healthy, 95 % bar)
- **Still open:** new SAME-PERSON replicate arrays, from a set that did not help build the band, to confirm the repeatability bar
- **Advantage:** clustering is the cancer fingerprint; ordinary aging and inflammation scatter errors instead. Most specific thing we have
- **Blind spot:** whole blood so far, while the meter it rides on is commissioned on sorted cells - that scope gap must be reconciled before any detection claim; and one array decided the second test (22/23 = 95.65 % against 95 %)
- **What would break it:** healthy replicate pairs from one person differing by more than the band's own tolerance
- **Next step:** search GEO for healthy EPIC v1 series with same-person repeats, excluding every set in the spent register
- **Record:** [CSCORE_COMMISSIONING_PLAN.md](doors/CSCORE_COMMISSIONING_PLAN.md)

## 2. Territory contrast: fidelity in fragile territory over fidelity in stable territory

- **Reads:** WHERE the loss happens first - a sustained energy shortfall should take the late-replicating, solo-WCGW territory before it touches housekeeping promoters
- **Substrate:** arrays (works today, no new data) · **Cost:** free (design, synthetic test, then existing arrays)
- **Where it stands:** not written
- **Still open:** everything: derivation of the expected ordering, site classes fixed before any read, synthetic test, then real arrays
- **Advantage:** SELF-TARING - a ratio inside one array, so plate, batch and much of the composition effect cancel because numerator and denominator see the same cell mixture. Attacks the exact spread problem that blocked the blood answer tonight
- **Blind spot:** says nothing about absolute fidelity; a loss that is uniform across territories is invisible to it
- **What would break it:** the ordering reversing - stable territory losing before fragile territory in a condition known to cost the cell energy
- **Next step:** write the design with its falsifier and the site-class rule BEFORE touching an array

## 3. Met-A on whole blood: measure the real healthy spread

- **Reads:** total disorder above what the estimated cell mixture explains - the existing composition-aware reading, in a tube of blood
- **Substrate:** arrays · **Cost:** free (one laptop read of arrays already on disk)
- **Where it stands:** leverage measured on synthetic blood: a 1 % neutrophil drop moves the reading +2.64 % at a 0.40 share, +3.79 % at 0.55, +5.13 % at 0.70
- **Still open:** the detection limit - the spread between REAL healthy people, which the simulation cannot supply
- **Advantage:** dilution costs much less than expected, and the function already exists in the chain
- **Blind spot:** an array averages molecules before anyone reads them, so one cell type can never be isolated - only the one-directional question is available. And the healthy whole bloods in hand are spent as the clustering meter's test, so this is development use
- **What would break it:** healthy people differing by more than the leverage of the signal we are looking for (then arrays cannot carry blood and sequencing is required)
- **Next step:** run the chain's whole-blood function on the real healthy arrays already calibrated on disk and report the spread
- **Record:** [DEV_BLOOD_SENSITIVITY_01.md](doors/DEV_BLOOD_SENSITIVITY_01.md)

## 4. Distance to the ceiling

- **Reads:** how close a sample sits to the derived entropy ceiling - the right-hand mark of the gauge, past which the reading stops rising with further loss
- **Substrate:** arrays and sequencing · **Cost:** free (arithmetic on readings we already have)
- **Where it stands:** the ceiling is in the foundation as derived. AUDIT DONE 2026-10-13 (DEV-CEILING-AUDIT-01): of 7,681 readings that record the flag, none is past the ceiling; the composition-aware path never recorded it, so the three sealed tissue results cannot be audited from saved files, and that gap could only have cost detections
- **Still open:** re-read the three sealed tissue sets with the flag recorded (free, arrays on disk, reported as an audit that cannot change their sealed verdicts); and the reading itself - distance to the ceiling has still never been reported as a number
- **Advantage:** derived with no free parameter, and it answers the author's own question - how far from the ceiling a tumor arrives
- **Blind spot:** only informative for samples near the ceiling; says nothing about a healthy-range sample
- **What would break it:** a sample reading past the ceiling, which the derivation forbids
- **Next step:** compute it for every sealed cancer reading already in the records and check none is past the turnover
- **Record:** [DEV_CEILING_AUDIT_01.md](doors/DEV_CEILING_AUDIT_01.md)

## 5. Run-length spectrum of losses

- **Reads:** the SHAPE of clustering - exponential against heavy-tailed, which separates a processive mechanism from independent point losses
- **Substrate:** sequencing only (needs several sites on one molecule) · **Cost:** laptop work on reads already in storage; no new sequencing
- **Where it stands:** not written
- **Still open:** the derivation of the expected shape under each mechanism, then a synthetic test
- **Advantage:** same data as the clustering meter, different question; distinguishes mechanism rather than only size
- **Blind spot:** impossible on arrays; needs depth, so shallow libraries are out
- **What would break it:** the measured shape matching neither mechanism's prediction in a case where the mechanism is known
- **Next step:** after the sequencing instrument is released from hold
- **Record:** [DEV_IAMA_CAUSE_01.md](doors/DEV_IAMA_CAUSE_01.md)

## 6. Tail of fragments in plasma (the early-detection instrument)

- **Reads:** the few badly disordered MOLECULES in a plasma sample, not the average - because early cancer is a tiny minority of fragments and an average hides it by construction
- **Substrate:** sequencing of cell-free DNA · **Cost:** box time per sample; the expensive one
- **Where it stands:** not written; gated
- **Still open:** the sequencing instrument is held, the atlas is not commissioned, and a healthy plasma reference does not exist yet
- **Advantage:** the highest reward on the whole board - it is the blood test, and a tail statistic is the right shape for early detection
- **Blind spot:** needs depth and fragment length; a shallow or heavily fragmented sample has no tail to read. Composition of plasma is itself a question
- **What would break it:** the tail in healthy plasma being as heavy as in known cancer plasma
- **Next step:** blocked behind the sequencing instrument's trim rule and the atlas; do not spend box money on it yet
- **Record:** [DEV_IAMA_READACROSS_01.md](doors/DEV_IAMA_READACROSS_01.md)

## Not an instrument, but it releases two of them: the trim rule

The sequencing instrument is held because the same kit read differently in two laboratories, and the cause is largely where in each file we read. Reading more places failed its planted test (pooling dilutes a bias, it cannot remove one). The remedy is to exclude the unsettled head of each run, decided by the sequencer's own per-cycle quality rather than by whether our answer improves. Instruments 5 and 6 are both behind it.

_Several instruments, each with a different blind spot, reading the same sample and agreeing - that is worth far more than one instrument we trust. Disagreement is also informative: a sample that reads normal on total disorder and abnormal on clustering is telling us which channel failed._
