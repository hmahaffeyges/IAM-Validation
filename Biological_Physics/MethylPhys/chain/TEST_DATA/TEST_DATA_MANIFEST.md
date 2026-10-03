# chain/TEST_DATA — test inputs

Raw public GEO IDAT pairs used as **test inputs** for the chain's own checks (release check E1, E2, E3; operator practice). They are files to
run the chain on, not examples of any reading: nothing in this folder carries or implies a label, a class or an outcome. Calibration (Stage 1)
uses methylprep 1.7.1; `harness/pdshim.py` is a pandas-compatibility shim needed only in a Python 3.12 / pandas 2.x container (the chain env,
Python 3.11 with pandas 1.5.3, runs methylprep natively). The chain logic is identical either way.

| GSM | Series | Array | Specimen | Used by |
|-----|--------|-------|----------|---------|
| GSM1051525 | GSE42861 | 450K | whole blood | 450K input (refused by v3's platform check) |
| GSM1051533 | GSE42861 | 450K | whole blood | 450K input (refused by v3's platform check) |
| GSM2333901 | GSE87571 | 450K | whole blood | release check E2: Stage 0 and Stage 1 run, Stage 0.7b quarantines, exit 2, nothing written |
| GSM2333905 | GSE87571 | 450K | whole blood | 450K input (refused by v3's platform check) |
| GSM2333950 | GSE87571 | 450K | whole blood | 450K input (refused by v3's platform check) |
| GSM8772491 | GSE288652 | EPIC v1 | tissue | release check E1 (Stage 0 PROCEED, Stage 1, platform check refuses the incomplete vector) and E3 (the vector a constructed whole blood is built on) |
| GSM8772492 | GSE288652 | EPIC v1 | tissue | EPIC v1 input |
| GSM5065990 | GSE166212 | EPIC v1 | tissue | EPIC v1 input |
| GSM5065985 | GSE166212 | EPIC v1 | tissue | EPIC v1 input |

Source: NCBI GEO, `https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM<prefix>nnn/<GSM>/suppl/`. A calibrated cache (`betas_cache.pkl`, ~137 MB)
is not stored in git (over GitHub's 100 MB limit); rebuild it from these IDATs with Stage 1.

Single-molecule test data: none is bundled as a file. Release check E4 constructs its per-site table and `.pat` file at run time.

Rewritten 2026-10-03: the earlier text described these files with disease and class-era readings from the retired chain; those are not
test inputs and were removed (the retired text is in the repository history).
