# DEV-EPIC-V2-01 - EPIC v2 support behind a development flag (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Author decision M.** EPIC v2 is refused at intake now; in development build support: a calibrator that reads EPIC v2, a v2 neutrophil floor,
a v2 replicate test, behind a flag.

**Calibrator.** SeSAMe (Bioconductor) `openSesame(prep = "QCDPB")` (quality mask, channel inference, dye bias, pOOBAH, noob) in its own environment on
the box. EPIC v2 probe names carry a design suffix (cg..._TC21); replicate probes of one CpG are averaged into the EPIC v1 name.

**Search for a v2 floor (2026-10-04, before reading data).** GEO GPL33022 series with neutrophil / sorted / purified / leukocyte: GSE307998 (CLL),
GSE277573 (tumour series). No purified healthy neutrophil EPIC v2 arrays were found. A v2 neutrophil floor cannot be measured from public data now.

**Checks (GSE286313: the same venous bloods on EPIC v1 and EPIC v2).**
1. SeSAMe calibrates every v2 array of GSE286313. Bar: all, 0 crashes.
2. Coverage: of the 6,000 neutrophil identity sites and the 963 composition markers, how many the v2 vector carries (recorded).
3. Same blood, two versions: untared whole-blood A of each v2 array (read on the v1 floor and profiles at the shared sites, labelled cross-version)
   against its v1 pair calibrated by SeSAMe. Development target: SD of the pair differences <= 0.020.
4. Calibrator effect: the v1 arrays by SeSAMe against the same arrays by Stage 1 (methylprep), A difference recorded.
5. v2 replicate test: technical replicates on v2 in the bucket - none listed in the manifest; recorded as not assessable unless found in the series record.

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box jobs 04895900 (SeSAMe installed into `/home/ubuntu/data/sesame_env` with micromamba: r-base 4.3, bioconductor-sesame, sesameData cache; every array
then stopped in the non-linear dye-bias step, preprocessCore "pthread_create() is 22"), b7b2c88e (re-run with the linear dye-bias step in its place:
qualityMask, inferInfiniumIChannel, dyeBiasL, pOOBAH, noob; written per array beside the betas) and 3003767d (pairs). Records: `data/DEV_EPIC_V2_01/`.

1. \measured SeSAMe calibrates **72 of 72** EPIC v2 arrays and 71 of 71 EPIC v1 arrays of GSE286313; 0 crashes.
2. \measured Coverage on v2: identity sites median **5,391 of 6,000** (4,869-5,398), composition markers median 870 of 963 (834-873). Below Stage M's 5,400 on every
   v2 array: no v2 array gets an A by the chain's rule; 17 of 72 also fall below the 867 markers Stage 2 needs.
3. \measured Same blood, two versions, read at the identity sites both arrays measure (median 5,387; below the 90 % rule, development only): 37 pairs read;
   A(v2) - A(v1) mean **-0.034**, SD **0.054** (target 0.020, outside). With the v1 array's composition on both: -0.036, SD 0.053.
4. \measured Calibrator on the same v1 arrays: SeSAMe minus methylprep A -0.026 (SD 0.018, 71 arrays). The
   calibrator alone moves A by about as much as the Normal band's half-width: a v2 floor must be measured with the calibrator that reads v2.
5. \observed No technical replicates on v2 in the bucket (no repeated titles in GSE286313): the v2 replicate test is not assessable.
6. \observed The flag end to end on two v2 IDAT pairs: intake refusal kept, `development.epic_v2` written with status OK; no A (coverage, as in 2).
- \openprob A v2 neutrophil floor needs purified healthy neutrophils on EPIC v2, read through the same calibrator, and its own identity sites chosen among the
  probes v2 carries. None are public (search above).

**Wiring.** `--dev-epic-v2 --sesame-rscript <Rscript>`; EPIC v2 stays refused at intake.
