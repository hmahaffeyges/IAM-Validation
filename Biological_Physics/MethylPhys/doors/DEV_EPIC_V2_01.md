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
