# COMMISSION-METAA-SORTED-01 — pre-registration (written before any of its arrays is read)

**The instrument's job:** a healthy sorted neutrophil or monocyte from a laboratory the atlas never held reads Normal on Met-A for the cell
the IAM-Deconvolver finds most of.

**Reference.** IAM-Atlas v2.1 only: each cell's healthy profile, and Normal as that cell's measured width on the atlas's own donors read
against the atlas (`stage_metaa_atlas.width_state`, `Runtime Matrices/Met_A_Widths/donor_widths_v1.json`, both committed before this test).
Nothing is taken from the samples read here or from any other sample of their laboratories.

**Data (chosen from GEO metadata only; none read before; none an atlas source; none in `CANON/data_spent.csv`).**
`data/DEV_METAA_HEALTHY_01/commission_sorted_01_samples.csv`, 79 arrays, raw IDAT files, read by the one command (default chain):
- monocytes, Infinium EPIC: GSE201752 healthy donors (31), GSE189426 healthy donors, peripheral blood (15), GSE196071 controls on EPIC (12);
- neutrophils, Infinium 450K: GSE318669 CD15 granulocytes (8: 3 non-smokers, 5 smokers), GSE110607 granulocytes of the healthy co-twins (13).

**Smokers.** Smoking changes neutrophil methylation, so the 5 smokers of GSE318669 are read and reported but not scored
(the scorer reads `commission_sorted_01_scored.csv`, the same list without them, 74 arrays). The healthy
co-twins of GSE110607 are healthy people related to a patient; they are scored and reported by series.

**Bars (the rehearsal's, unchanged; `commission_sorted_01_score.py`, a copy of `sorted_rehearsal_score.py`).**
1. The top cell found is the sorted cell's kind (a subtype counts) in at least 90 % of arrays.
2. Of the arrays with a state, at least 85 % read Normal.
3. In each cell kind with 5 or more arrays, at least 70 % read Normal.

**Met means:** Met-A on sorted neutrophils (450K) and sorted monocytes (EPIC) is commissioned against IAM-Atlas v2.1 alone, in that scope.

**Stated limits now.** Sorted cells only, not cells inside whole blood (the in-blood Normal is held until the cross-laboratory work is
done). One platform per cell kind. Arrays with no state are counted and reported.

**Prior reading:** none of these arrays has been read by this project.

**Correction (after reading; the sealed design is unchanged).** GEO lists GSE318669 as 450K, but the chain read 7 of its 8 neutrophil arrays as EPIC v1 from their own IDAT files (one, a smoker's, as 450K). So the scored 450K neutrophils come from one laboratory, GSE110607 (12 of 12 with a state Normal); GSE318669's 3 scored neutrophils are EPIC v1 and read Normal (1.015 to 1.026), outside the commissioned scope and reported as an observation. The three bars were scored over all arrays, as sealed, and are met; Met-A on sorted neutrophils on 450K is commissioned on one new laboratory's arrays.
