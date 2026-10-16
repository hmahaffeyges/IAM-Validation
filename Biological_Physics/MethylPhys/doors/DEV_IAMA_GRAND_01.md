# DEV-IAMA-GRAND-01: IAM-A's grand sealed test on healthy CD8 T cells (COM-IAMA-052)

**Status: rule sealed 2026-10-16; alignment running. No CD8 read was downloaded before the seal.**

**The instrument's job:** a healthy CD8 T-cell sample from a laboratory IAM-A has never seen reads Normal on IAM-A's gauge.

**Sources:** GSE144691 (Youngblood laboratory, St. Jude; Abdelsamed et al., J Exp Med 2017, doi 10.1084/jem.20161760; Abdelsamed et al., Nat Immunol 2020, doi 10.1038/s41590-020-0633-5), public on GEO and SRA: raw paired whole-genome bisulfite reads (NovaSeq 6000) of 13 healthy-donor sorted CD8 T-cell samples (naive, stem-cell memory, central memory, effector memory), with each run's sequencer and chemistry measured by the chain. GSE98837's eight samples are the same libraries, so they are not added again. The HIV, type 1 diabetes and mouse samples of the series are not used. The laboratory took no part in IAM-A's line (Loyfer 2023), its band (Loyfer 2023 and the Hansen laboratory, DEV_IAMA_BAND_01.md) or its shape reference.

**Prior reading:** none of the test data. Only GEO and SRA descriptions (titles, cell labels, run sizes, instrument) were read to choose and size the samples.

**The rule (sealed 2026-10-16 by the author, "okay, seal it"; numbers in `data/DEV_IAMA_GRAND_01/grand_rule.json`):**
1. Each sample: its first read pairs up to 40 billion bases (`samples.csv`, spots_cap), aligned as DEV-SECONDLAB-02's Hansen donors were, except that poly-G trimming stays on because these runs are NovaSeq; counted by chain Stage Q2.
2. The sample-shape check (COM-IAMA-053) first, against its Loyfer type (naive -> naive CD8; effector memory -> effector memory CD8; central and stem-cell memory -> CD8). A sample that fails is listed, not counted.
3. Each read sample's IAM-A reading (held error over what its own loose share predicts) against band v1, Normal 0.7787 to 1.0551.

- **S1:** at least 0.75 of the read samples sit inside Normal, with at least 8 samples read; fewer than 8 read gives NOT ASSESSED.
- **Reported, not scored:** every sample's reading, its conversion failure and sequencer error, read length and mapping efficiency.

**Synthetic test:** DEV_CHAIN_Q2_01/q2_synthetic.py, DEV_CHAIN_Q2_01/q2_synthetic.json

**What it would take to succeed:** S1 met. IAM-A is then commissioned on healthy blood CD8 T cells, whole-genome bisulfite or EM-seq reads.

**In plain words:** thirteen healthy immune-cell samples from a laboratory IAM-A has never seen are each read against their own line; most of them must read Normal.

Code: `data/DEV_IAMA_GRAND_01/box/align_grand.sh` (machine), `count_grand.py` (chain count), `score_grand.py` (chain score, run once).
