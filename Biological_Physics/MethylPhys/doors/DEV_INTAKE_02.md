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
