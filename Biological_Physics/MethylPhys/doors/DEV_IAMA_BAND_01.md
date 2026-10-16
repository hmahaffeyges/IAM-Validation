# DEV-IAMA-BAND-01: IAM-A's own healthy band (COM-IAMA-054)

**Sealed 2026-10-16 by the author ("Seal as drafted"), before any Hansen count was read.** Manifest row COM-IAMA-054.

**The reading.** A = the sample's measured held error divided by what its own loose share predicts on the chain's self-reference line
(version 1). A = 1 is on the healthy line; above 1 is more holes than its own line predicts (right on the gauge). Chain Stage Q2 `reading()`.

**The band.** The 2.5th to 97.5th percentile of every healthy reading, both laboratories pooled: Loyfer's 153 samples and each Hansen donor
(GSE49627) that passes the sample-shape check (COM-IAMA-053). A refused donor is not pooled. Computed by chain Stage Q2 `build_band()`
(`data/DEV_IAMA_BAND_01/build_band_v1.py` writes `chain/Runtime Matrices/IAM_A_SelfRef/band_v1.json`, stamped). Nothing is chosen by hand;
the laboratory difference sits inside the band by construction. Prior reading (Loyfer alone, already counted, shown to the author before sealing):
median 0.950, 2.5th to 97.5th percentile 0.785 to 1.055, range 0.720 to 1.068.

**Synthetic test:** DEV_CHAIN_Q2_01/q2_synthetic.py, DEV_CHAIN_Q2_01/q2_synthetic.json

(T7: samples planted exactly on a planted line read A = 1; a planted spread from two laboratories returns its known percentiles. MET.)

**What it would take to succeed:** the band is built by the chain from the stamped counts of both laboratories by the rule above. It is a
measurement, not a pass or fail; it becomes IAM-A's Normal mark when IAM-A is commissioned (COM-IAMA-052).

**In plain words:** the healthy range for IAM-A comes from healthy people measured in two different laboratories, not from a number we picked.
