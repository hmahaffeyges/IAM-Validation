# DEV-IAMA-KIT-01 — is the Swift kit's high IAM-A a read-end artefact? (development, 2026-10-09)

**DEVELOPMENT - not commissioned.** Written before the clipped readings below were made.

**Observation (session 2).** Donor 6's neutrophils: TruSeq / HiSeq X IAM-A 1.047 (Normal); Swift / HiSeq X 1.167 and Swift / NovaSeq 1.159.
Swift Accel-NGS libraries add a low-complexity tail at one read end, and the bases next to it read artificially unmethylated. At a truly
methylated site that is an isolated T: exactly what IAM-A counts as a copy error.

**Test.** The two HiSeq X BAMs of donor 6 (SRR9888332 Swift, SRR9888333 TruSeq) through wgbstools 0.1.0 bam2pat with `--clip` 0, 10 and
15 (ignore the first and last N bases of every read; wgbstools' own option for biased read ends), then Stage Q with P v2. The M-bias table
(`--mbias`) is written at clip 0.

**Reading rules (set now).**
1. Read-end artefact if: clipping lowers Swift IAM-A by more than its two halves differ, toward TruSeq, AND changes TruSeq by less than 0.01.
2. If both move together, clipping removes real molecules' ends in both, and P (measured on Loyfer files with no clip) cannot be compared;
   recorded, no conclusion.
3. Whatever the result, no chain setting changes without the author: a clip in the pipeline changes what P was measured on.

---
## Results (2026-10-09, box; nothing above the line changed)

| clip (bp each end) | Swift SRR9888332: IAM-A / ε / opportunities | TruSeq SRR9888333: IAM-A / ε / opportunities | Swift − TruSeq |
|---|---|---|---|
| 0 | 1.167 / 0.04713 / 6.68 M | 1.047 / 0.04077 / 8.55 M | 0.120 |
| 10 | 1.159 / 0.04670 / 5.20 M | 1.028 / 0.03981 / 6.13 M | 0.131 |
| 15 | 1.155 / 0.04644 / 4.47 M | 1.021 / 0.03944 / 5.34 M | 0.134 |

\measured Clipping lowers both kits, TruSeq more (−0.026) than Swift (−0.013). Rule 1 is not met (TruSeq moved by more than 0.01);
under rule 2 no conclusion is drawn from the clipped values. **The 0.12 difference between the kits is not removed by clipping read ends**,
so it is not (only) the read-end tail. \observed The Swift runs' IAM-A C-score is also far higher (306 vs 45 at clip 0), i.e. their extra
copy errors are clustered along the genome, which points to where Swift reads land (coverage of different regions) rather than to read
ends. M-bias tables (clip 0) are on the box at `/mnt/scratch/clip/pat_c0/*.mbias/` (S3 upload of this job failed on a stream error; the
summary above is the job's own output).

**Next (development):** read both kits over the same molecules' regions only — restrict each PAT to the CpG blocks covered ≥ 1× in both
runs — to test whether the difference is where the reads fall.

## Part 2 — same CpG positions (2026-10-09)
On the 2,249,976 positions both runs cover: **Swift 1.159, TruSeq 0.994** (ε 0.04667 vs 0.03806). With each position weighted the same in
both runs: ε 0.0440 vs 0.0390. Swift is higher in every depth quintile. \measured **The gap is not where the reads land**: at the same
positions, the Swift molecules carry more discordant calls.

## Part 3 — which molecules carry the extra calls
Molecules with ≥ 6 calls split by their majority state (a rougher error count than Stage Q's; used only to compare the two runs):
methylated-background discordance 0.100 (Swift) vs 0.085 (TruSeq), +18 %; unmethylated-background 0.103 vs 0.069, +50 %. Share of
unmethylated-background molecules 13 % vs 19 %. \calculated Swift's lower conversion (0.983 vs 0.992, lambda CHH) can add at most ≈ 0.009 to
the unmethylated-background rate: about a quarter of that excess, and none of the methylated-background excess.

## What this means (observed; no chain change)
\observed **Loyfer 2023 also made their libraries with the Swift Accel-NGS kit** (Methods: Zymo EZ-96 conversion, Swift Accel-NGS
Methyl-Seq libraries). So P was measured on Swift libraries, yet this laboratory's Swift libraries read 1.16 and its TruSeq library 1.05
(0.99 on shared positions). The difference is therefore a **laboratory-and-kit effect on IAM-A of up to ~16 % on healthy cells** — the same
kind of offset Met-A had before the same-run tare. Options for the author (in the morning list): (a) IAM-A read against same-run healthy
references, as Met-A is (a tare); (b) P measured per kit and laboratory; (c) a wider healthy band. Donor 7 (four more runs) is being read.
