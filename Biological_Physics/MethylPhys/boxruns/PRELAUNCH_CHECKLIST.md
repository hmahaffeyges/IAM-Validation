# Before every paid launch

Author rule (2026-10-18): *"Ignorance can be forgiven, but ONLY if we didn't already have the information the lesson teaches."*
Read this file before every AWS launch. Every box job in `CANON/box_jobs.json` dated 2026-10-18 or later carries a `prelaunch` field naming
this checklist and the known-facts section that applies; `CANON/box_code_check.py` refuses a job without it. Each item names the lesson it
comes from (`../LESSONS.md`).

## The checklist

1. **The synthetic plants everything the real data holds** for this organism and substrate (the known facts below): unmethylated regions,
   every methylation context, the real length mix, sequencer error, conversion failure. It runs through the exact scoring path the real
   run will use, and passes, before launch. (T6, T15, C23)
2. **The scorer has run on a no-effect case and a planted-effect case.** (T5, R9)
3. **Read length fits the aligner**, and one real file has been downloaded and aligned end to end on a small machine first. (B1, B1b, T3 run 2)
4. **The code is in S3 before the machine boots**: published and verified, then launch. (T3 run 1, 2026-10-17)
5. **Results go to S3 as they are made**, from a role the machine actually has. (C5; atlas v2.1 blocks, 2026-10-16)
6. **The disk is sized** for the largest file times the parallel jobs, plus 50 %; the image's own size is the minimum. (box disk full, 2026-10-17)
7. **A watcher on each machine from minute one**: it flags a stopped machine or 75 minutes without a status line, never only the total. (spot machine B reclaimed and missed for 2 hours, 2026-10-17)
8. **Spot machines are taken back.** Work comes from a shared claim queue so a replacement picks up unfinished claims; the plan says what a reclaim costs. (CD8 grand test; T3)
9. **The machine stops itself** (shutdown timer and terminate on shutdown). (B2)
10. **Cost and finish time are estimated before launch and told to the author**, and re-measured from the real rate within an hour of launch.

- **S3 steps run on the machine's own Python (`~/env/bin/python`), never the chain's clean environment**, which has no boto3 by design. The IAM-A example (2026-10-18) failed its first download this way, though every earlier script already did it right.
- **A rerun on a committed input uses the exact files the committed run stamped, checked by checksum** (reference genome, alignment), never the first match of a file search. The IAM-A example (2026-10-18) first picked Bismark's bisulfite-converted genome.

## Known facts, by data

**Plants (Arabidopsis whole-genome bisulfite).** Methylation in three contexts: CG, CHG and CHH; the IAM-A counter reads CG only. Much of the
genome is unmethylated or only partly methylated, including many genes; transposons and centromeres are heavily methylated; gene-body CG
methylation is moderate. A chloroplast genome, unmethylated, measures conversion failure. Chromosome names are Chr1 to Chr5, ChrC, ChrM.
Pooled plants per sample. A synthetic that plants only methylated molecules cannot test the counter's correction (C23).

**Human whole-genome bisulfite.** Unmethylated CpG islands (about 1 to 2 % of the genome) beside methylated open sea; loose tiles; read
lengths 36 to 150 bases across laboratories; Bismark XM tags or gemBS XB tags (the second-lab undercount, 2026-10-15); conversion is
checked on non-CpG C, not on a plastid.

**Cell-free DNA.** Fragments with a mode near 167 bases and a 10-base periodicity; about 3 % of fragments qualify for the counter
(DEV-CFDNA-SHORT-01); most cfDNA comes from blood cells.

**Arrays.** EPIC v1 only for Met-A; EPIC v2 and 450K are refused. The Illumina manifest must be in place before Stage 1, and a batch runs
its first array alone so the download is not raced.

**Salmonids (planned).** Body temperature is the water's; the Mahaffey number scales by 310.15 K / T. Fin clips; duplication and conversion
failure must be removed before any reading (Part 7, "What it would take").

## Known AWS and transfer caveats

- Spot reclaims, sometimes hours into a run; the reclaimed machine stops billing, but its time is lost.
- ENA downloads fail with "no length" on some passes; the queue keeps the claim and retries every 10 minutes.
- Zenodo is sometimes degraded (5 MB in 38 s on 2026-10-18); test with a small file before a large upload.
- GitHub refuses pushes of several hundred MB in one go; push large files one commit at a time.
- The laptop is not a compute box: no multi-GB transfers or worker pools there.
- A command run on a box must not match its own command line in `pkill` (B3).
