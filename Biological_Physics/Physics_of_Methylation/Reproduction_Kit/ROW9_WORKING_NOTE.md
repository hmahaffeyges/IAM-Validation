# Row 9 working note - the report (unsealed build)

## 2026-09-22 - MethylPhys CPG builder (CPG_Engine/MethylPhys_Interface/build_methylphys.py) replaces cpg_report_v3.py
Author's direction: the v3 table was 'very blah'; the report is now the researcher interface - dark, tabbed, printable, Clinician/Researcher toggle, eleven tabs
(Reading, Every cell, Departure, Sky, Healthy reference, Integrity, Chain, Physics, Story, Record, Run). BREACH stays as a gauge reading on the CLASS gauge (a temperature);
no comparison to any prior cohort; no condition named; no years. Vocabulary guard on the four measurement tabs.
- Conductor now passes `cells_all` (all 115 per-cell A) and matches the laboratory key by accession prefix (departure false-alarm rate was printing 'not measured' for Uppsala - fixed).
- First read of GSM2333901 against the spec: (a) stem_adult and progenitor printed the SAME A_mapped 0.9262 - the conductor scores them as one joint haematopoietic-progenitor component
  on shared identity loci; collapsed to one gauge row, labelled as joint. (b) The same lineage appears under many atlas entries with a two-fold spread in per-cell A on one healthy array
  (Neu 1.158, granulocytes 1.123, Neutrophils_EPIC 1.121, Neutrophils_reinius 1.117, Neutro 1.083 ... neutrophil 0.545): on the marker surface, WHICH REFERENCE PANEL DEFINED THE CELL is a
  larger effect than anything biological in a healthy sample. Consequence: the per-cell healthy range must be per atlas ENTRY (its own markers), never per cell name; the report should
  group aliases visibly. (c) Class composition arrives in percent, cell fractions in [0,1] - still an open conductor item; the builder scales.
- Per-cell healthy reference (H_ref(entry), p10-p90 per laboratory) in build from 80 healthy arrays per laboratory (seed 2029; GSE87571/42861/111629/125105), raw IDAT -> Stage 1 noob,
  Stage 1 betas checkpointed per laboratory as artifacts + manifest the moment they exist (author: 'I wish you would have saved the betas'). Until then the cell table prints 'pending'.
- Substrate x specimen grid (5 substrates x whole blood / plasma cfDNA / tissue / CSF / urine / stool) reserved: every cell prints its commissioning status. Not yet in the builder.
