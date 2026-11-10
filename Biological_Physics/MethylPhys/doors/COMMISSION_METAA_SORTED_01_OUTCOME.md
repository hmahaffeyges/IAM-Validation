# COMMISSION-METAA-SORTED-01 — outcome: MET (all three bars)

Sealed in `COMMISSION_METAA_SORTED_01_PREREG.md` (commit de180dba) before any array was read; read by the one command on Google
(`series_google.sh`, tag cs1_sorted, 10 shards); scored by `commission_sorted_01_score.py` on `commission_sorted_01_scored.csv`
(`data/DEV_METAA_HEALTHY_01/commission_sorted_01_summary.txt`, rows `commission_sorted_01_rows.csv`).

**Not a milestone:** the milestone is recorded once, in its commissioning note (`COMMISSIONING_NOTE_METAA_ATLAS_SORTED.md`).

**In plain words.** Healthy sorted white blood cells from five laboratories the atlas never held were read against the atlas alone, and every
one the chain could read came out Normal. **What it would take** to commission the rest of Met-A: the same kind of test passed by lymphocytes, by neutrophils on EPIC, and
by cells inside whole blood.

| Bar | Result | Met |
|---|---|---|
| 1. top cell found is the sorted kind, 90 % of arrays | 73 of 74 (the 74th is an array the chain declined, below) | yes |
| 2. of arrays with a state, 85 % Normal | 66 of 67 | yes |
| 3. each kind with 5+ arrays, 70 % Normal | monocytes 51 of 51, neutrophils 15 of 15 | yes |

**Correction (after reading; the sealed design is unchanged).** GEO lists GSE318669 as 450K, but the chain read 7 of its 8 neutrophil arrays as EPIC v1 from their own IDAT files (one, a smoker's, as 450K). So the scored 450K neutrophils come from one laboratory, GSE110607 (12 of 12 with a state Normal); GSE318669's 3 scored neutrophils are EPIC v1 and read Normal (1.015 to 1.026), outside the commissioned scope and reported as an observation. The three bars were scored over all arrays, as sealed, and are met; Met-A on sorted neutrophils on 450K is commissioned on one new laboratory's arrays.

**Arrays without a state (8 of 74, each with the chain's own reason).** Seven GSE201752 monocyte arrays were declined for array quality: one
detected 41 % of probes, so the cell shares were withheld; six measured fewer than half of the monocyte's 6,000 identity sites (detection
61 % to 86 %), so Met-A was not read. One GSE110607 neutrophil was found at 62.9 %, below the 69 % at which the atlas's neutrophil donors
were measured. No reading carried the atlas's middle-not-at-1 flag.

**Readings by laboratory (Normal arrays, Met-A median and range).** Monocytes (EPIC): GSE201752 0.9884
(0.9567 to 1.0181), GSE189426 0.9875, GSE196071 0.9769.
Neutrophils: GSE110607 (450K) 1.031; GSE318669 (EPIC v1, 3 scored, outside the scope) 1.0159. The 5 smokers (read, not scored) read
Normal, 0.968 to 1.047. Monocytes sit 1 to 2 percent below 1 and neutrophils 2 to 3 percent above, inside each cell's measured width.

**Commissioned:** Met-A on sorted neutrophils (Infinium 450K) and sorted monocytes (Infinium EPIC v1), against IAM-Atlas v2.1 alone (each
cell's healthy profile, Normal as its measured width on the atlas's own donors). This completes the atlas-only re-commissioning plan for
Met-A (the width synthetic, met as test 3; the spent sorted-cell rehearsal, neutrophils and monocytes all Normal; this fresh
test). Neutrophils on EPIC were read Normal in the spent rehearsal only and are not in this scope; lymphocytes read 2 to 5 percent below 1
(rehearsal) and wait on the cross-laboratory work; cells inside whole blood stay development.
