# COMMISSION-MONO450-01 — pre-registration (written before any of its arrays is read)

**The instrument's job:** a healthy sorted monocyte on an Infinium 450K array, from a laboratory the atlas never held, reads Normal on Met-A
and sits at or under its C-Score limit, for the cell the IAM-Deconvolver finds most of.

**Reference.** IAM-Atlas v2.1 only: Met-A's Normal is the monocyte's measured width on the atlas's own donors (`donor_widths_v1.json`); the
C-Score's limit is the derived limit times the monocyte's factor measured on the atlas's own donors (`CScore_Limits/cell_factors_v1.json`).
Both committed before this test. Nothing is taken from the samples read here.

**Data (chosen from GEO metadata only; none of these arrays read before; neither series an atlas source).**
`data/DEV_METAA_HEALTHY_01/commission_mono450_01_samples.csv`, 19 arrays, raw IDAT files, read by the one command (default chain):
GSE110607 sorted monocytes of the 13 healthy co-twins (the laboratory's granulocytes were read in COMMISSION-METAA-SORTED-01; its monocytes
were not), and GSE131627 sorted monocytes of the 6 healthy controls. The platform is taken from the chain's own reading of each array.

**Bars (`commission_mono450_01_score.py`).**
1. The top cell found is a monocyte in at least 90 % of arrays.
2. Of the arrays with a Met-A state, at least 85 % read Normal.
3. Of the arrays whose top cell has a C-Score, at least 85 % sit at or under its limit.

**Met means:** Met-A and the C-Score on sorted monocytes (Infinium 450K) are commissioned against IAM-Atlas v2.1 alone, in that scope.

**Stated limits now.** Sorted monocytes on 450K only; two laboratories, one of them already seen for another cell; arrays without a state
are counted and reported; an array the chain reads as EPIC is reported and not scored.

**Prior reading:** GSE110607's granulocytes were read in COMMISSION-METAA-SORTED-01; none of these 19 monocyte arrays has been read.
