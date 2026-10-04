# DEV-INTAKE-02 - intake changes F, A, B, L and the EPIC v2 refusal (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Changes under test (code, no physics).**
- **F** Age and sex are optional at intake. `--sex` and `--age` are recorded when given and never required. Stage 0.2 no longer stops a
  manifest without them. Stage 0.8 records the sex the array shows; with no declared sex it records `NOT_DECLARED` and does not stop.
- **A** Below 90 % noise-site coverage (fewer than 43,676 of the 48,528 noise sites measured) the noise index N cannot be formed, so the array's
  noise is unknown. The gauge state is then withheld, A is printed as a number, and the report says in plain words how many noise sites were
  measured, how many are needed, why (the gate needs the array's own noise before it can show a state), and what to do (re-hybridise, or tare
  against same-run references and still read the state as withheld until N is measured).
- **B** Identifiers are hashed (sha256, first 32 hex) in the bundle and the evidence ledger, including any path or command argument that
  carries the typed id. The printed report keeps the id the operator typed.
- **L** Intake accepts whole blood and isolated / sorted / purified neutrophils (the specimens that have a reference). It refuses at intake,
  with a report and no reading: PBMC, sorted cell fractions without a reference (B, T, NK, monocytes, basophils, eosinophils, gMDSC), bone
  marrow, cell lines, tissue (placenta and others), and unspecified specimens. Each refusal names the specimen and says it needs its own reference.
  A constructed DNA mixture of blood cells (test material) is read as whole blood with a NOTE.
- **M (refusal)** EPIC v2 stays refused at intake (unchanged from round 1).

**Run.** The 1,569 arrays that stopped at Stage 0 on a missing age or sex in DEV-BASE-CHAIN-01 (manifest `doors/data/DEV_BASE_CHAIN_01/manifest.csv`),
re-run end to end from their IDATs with the round-2 chain: pass 1 for every array, pass 2 (same-run median tare, the DEV-BASE-CHAIN-01 reference
rule) where >= 3 healthy-labelled references of the same series and specimen are read in pass 1. Age and sex are passed when the series record gives them.

**Checks (set now).**
1. No crash: every one of the 1,569 runs end to end, is refused with a report, or stops at Stage 0 with a named reason. Bar: 0 crashes.
2. Specimen rule (L): every array whose recorded specimen is in the refused list carries `SPECIMEN_REFUSED` with the specimen named, has a report
   and a bundle, and no Met-A. Every whole blood and isolated-neutrophil array is not refused for its specimen. Bar: 100 % both ways.
3. Optional age and sex (F): no array stops on a missing age or sex. Bar: 0.
4. Noise coverage (A): every array with fewer than 90 % noise sites measured and an A carries the withheld state and the explanation, and no gauge.
   Bar: 100 % of such arrays; the count is recorded.
5. Hashed ids (B): no bundle and no ledger row of the run contains the typed id (the GSM accession) in clear text; every report carries it in its title.
   Bar: 0 and 100 %.
6. Purified healthy neutrophils that were lost at Stage 0 in round 1 (13 GSE118144 controls, 6 GSE167998) are read: tared A_rel recorded beside the
   DEV-BASE-CHAIN-01 (b) set; the round-1 bar (every array Normal on tared A_rel) is restated over the enlarged set.

---

## Outcome (2026-10-04, read after the checks above were written)

Run: box ssh:methylphys-cpu-01, chain workspace commit 7c33552 on main 185f609, 1,569 arrays pass 1, 268 arrays pass 2 (same-run median tare), 7,804 s.
Records: `doors/data/DEV_INTAKE_02/intake02_readings.csv` (one row per array and pass), `intake02_purified_neutrophils_tared.csv`; scripts `doors/data/DEV_ROUND2_box/r2_idat.py`.

1. \measured No crash: 1,569 of 1,569 ran end to end with a report and a bundle. 0 crashes, 0 Stage 0 stops. One whole-blood array
   (GSE191297 GSM5743153, 591,441 probes) stops with `PLATFORM_REFUSED` and a report (incomplete vector; chain v3 reads EPIC v1 only). Bar met.
2. \measured Specimen rule: 955 arrays refused with `SPECIMEN_REFUSED` (PBMC 235, sorted T 189, cell line 134, placenta 93, sorted monocytes 86,
   sorted B 76, bone marrow 71, unspecified 43, gMDSC 14, sorted basophils 6, sorted NK 4, sorted eosinophils 4); 955 of 955 name the specimen,
   carry a report and a bundle, and have no Met-A. Whole blood 578 and isolated neutrophils 35 are not refused for their specimen (613 of 613). Bar met both ways.
3. \measured Optional age and sex: 0 arrays stop on a missing age or sex. Sex declared on 335 (sex check PASS 335), not declared on 278
   (`NOT_DECLARED`, no stop); age declared on 99. Bar met.
4. \measured Noise coverage: 0 of the 613 read arrays have fewer than 43,676 noise sites measured (lowest 48,127 of 48,528), so the withheld-for-coverage
   branch is not exercised by real data in this run; count 0. It is exercised on a constructed array by release check E8.
   Readings of the 613: A on 550; A withheld on 63 for named reasons (neutrophil fraction < 0.2: 39; < 5,400 identity sites: 21; composition markers short: 3);
   gauge withheld on 213 for noise index > 0.149 with no same-run tare (A printed as a number).
5. \measured Hashed ids: the typed id (GSM accession) appears in 0 of 1,837 bundles and 0 of 1,837 ledgers; no GSM accession of any kind appears in a
   bundle; 1,837 of 1,837 reports carry the typed id in the title and the label DEVELOPMENT - not commissioned. Bar met (0 and 100 %).
6. \measured Purified healthy neutrophils lost at Stage 0 in round 1, now read and tared (same-run median tare, healthy-labelled references of the same series):
   GSE167998 6 of 6 Normal (A_rel 0.991-1.008); GSE118144 controls 8 of 13 Normal (3 below, 2 above; A_rel 0.913-1.065).
   Enlarged (b) set: floor 6 of 6, other laboratories 56 of 68 (round 1: 42 of 49). The round-1 bar (every array Normal on tared A_rel) is not met on the enlarged set.
   Not run here: self-tare II then median tare (DEV-SELFTARE-02) on these 19 arrays.
