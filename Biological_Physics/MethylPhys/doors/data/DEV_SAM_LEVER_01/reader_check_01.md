# RRBS reference-free reader check (2026-10-10, on wild-type GSE77079 SRR3111472 only; no knockout read)

| reads | CpG called at a position when | ε |
|---|---|---|
| raw (11.9 M) | ≥ 1 read of the fragment shows C-G (rrbs_iama.py as committed) | 0.307 |
| adapter-trimmed (`rrbs_trim.py`) | ≥ 1 read | 0.117 |
| trimmed, first 3 M reads | ≥ 1 / 2 / 3 / 5 reads | 0.085 / 0.054 / 0.020 / 0.015 |

ε depends on the calling threshold, so the reader does not measure copy error. Without a genome reference, one sequencing error turning T into C
at a TG makes a fake CpG, which then reads 'unmethylated' in every other read of that fragment and counts as an isolated error. The effect grows
with reads per fragment (here ~9). 18.6 % of reads are adapter, whose unconverted CG adds error-free calls. The reader matched a direct count on
constructed reads, which carry no sequencing errors, so that check could not see this.
Consequence: copy error from RRBS is read only after alignment to the species' genome (CpGs from the reference), adapter and fill-in trimming,
then Stage Q's rule, as for human data.
