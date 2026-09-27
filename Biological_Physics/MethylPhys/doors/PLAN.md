# PLAN — what we do next, in order

One line per item. No history, no strikes: when an item is done it comes OFF this page; its record is the register
([`CHAIN_COMMISSIONING.md`](CHAIN_COMMISSIONING.md)), the ledger ([`ENHANCEMENTS.md`](ENHANCEMENTS.md)) and the procedure's own outcome file.
Order rule: chain before documents, documents before launcher, and nothing edits the chain while a procedure is scoring.

## Housekeeping first (today showed both are needed)

1. **Runner convention** — every long runner takes its own output dir, log and results name per launch and polls a STOP file; the sandbox cannot kill a process from a later cell, so "kill and relaunch" is never done again.
2. **Bundle at the end of every session** — chain files changed that day, the generators, and any new data, as three zips with SHA256SUMS.

## Chain — one at a time

3. **Sky — decided 2026-09-27 (author): drawn as z on the atlas-posterior σ**; the off-identity-loci offset is drawn and labelled, never re-centred. Remaining: the label on the plate, and the offset's cause (items 7, 20).
4. **FOREIGNSCORE-01 diagnostics** — sealed NOT ADOPTED 2026-09-27 (no scoring floor; detected foreign cells print fraction and no A). Three cheap diagnostics before any re-run: the inversion arithmetic (non-monotonic in f), the neural twin label at f ≥ 10 %, and B5 — the blood cells' own A moves up to 0.031 with foreign material present; per blood cell, per foreign cell, which shared identity loci; correct or print.
6. **Twin/family thresholds as a runtime matrix** — no constants in code (same move as the intake thresholds).
7. **Chip term** — the SNP tare did not remove it (TARE-01 B6); the control-probe model (0.002–0.018 on the clean laboratories) is the recorded route. Pre-register; run on the 768 calibrated arrays.
8. **Serial mode** — `run_sample.py --prior <bundle>`: same patient, per-cell ΔA, Δfraction, difference sky; change floor pre-registered. This is what the sky is for.
8a. **PROC-BLOODCANCER-01** (author 2026-09-27: 'this one sounds like our specialty right now' — moves ahead of DIRECTION-01) — the leukocyte as the diseased cell (PHYSICS_LEUKOCYTE_GAUGE.md row 1): public 450K whole-blood/PBMC AML and CLL cohorts; pre-registered per-cell A on the affected lineage against the others (CLL: B cells move, neutrophils read 1.00). The one test the fraction confound cannot fake.
8b. **PROC-DIRECTION-01** — settle the 'bidirectional immune cell' question from the AD/PSP era (GIFT AD d +0.68, PSP −0.38 were pooled-class, case-minus-control numbers): re-read the AD and PSP cohorts on disk through the current chain, per cell, against 1.00; record for each present immune cell which side of 1.00 it sits on, and whether T and B lineages sit on opposite sides in the same specimens (the AD-era record: per-CpG bidirectional drift nulled the pooled entropy; CPG-VAL-008 and the covariance PC2 put T cells DOWN in AD while healthy ageing reads UP). Falsifiable form: CD4/CD8 below 1.00 with neutrophils/monocytes at or above it in the same AD specimens. Direction is a property of a cell, never a net over a class or a sign relative to a control group.
9. **Fraction confound, second step** — minority cells at 5–10 % read the majority cell's β on their identity loci (NK ~0.946, CD8 ~0.961, CD4 ~1.031 on a perfect specimen). Needs a solver precise to < 0.005 on minority blood cells; first candidate is atlas v2's covariance (item 20). Until then RC2 reports it as open.

## The face of the chain

10. **Launcher** — local `run.py`: verify files by hash, take the IDAT pair, run stage by stage with live pass/fail, open the report; replication menu for the sealed procedures.

## Procedures — on the commissioned chain

11. **PROC-BRAIN-01 redo** — clean single-provenance CSF run.
12. **Commission a solid-tissue laboratory** — pipeline map for its pipeline; gates the glioma and progression re-tests.
13. **Gastric** — six stomach entries, two families, never tested.
14. **Breast shedding in real patient blood** — the question that started the detection work.
15. **Re-run the webpage-draft claims (CRC, breast, HCC) as pre-registered procedures** — education re-tested on the commissioned chain.

## Later chain work

16. CD4/CD8 separation — needs loci this block lacks (second blood block).
17. Second-block solve for the nine finer blood subsets.
18. EPIC platform block (865k loci).
19. Coverage floor on any class-selection rule.
20. **Atlas v2 — joint hierarchical build** ([`ATLAS_V2_SPEC.md`](ATLAS_V2_SPEC.md): model, inputs, acceptance tests A1–A9, dry run, compute) (November machine or a rented 128 GB box, ~1–2 days). One MCMC over all cells with a shared per-locus probe effect, a per-source effect and a per-cell effect; undefined loci imputed from class and nearest cells with wide SD; cross-cell covariance kept; H_min untouched. Acceptance written before the run: PROC-COV-01 misfit (0.067) below the NORMAL tolerance; twins collapse to one cell effect; a 252-locus cell's imputed loci carry SD > 0.1. Adding a cell from a new source (Loyfer 2023, the 476-methylome set, the July 2026 single-cell atlas) becomes one script: scale map → coverage family → twin test → identity loci → reads 1.000 on itself. Its covariance is the noise model the matched filter lacked and the solver item 9 needs.
21. **Ledger re-plan** — once 1–10 are done, rewrite ENHANCEMENTS as one ordered plan for what remains.

21b. **RETEST set — pre-registered now, run on the rented machine** (author 2026-09-27: "get a bunch of the tests ready to run"). One PROC per old VAL worth re-reading on the commissioned chain, bars fixed before the machine exists: EPIC-Italy pre-dx breast (GSE51032, per cell, per year-to-diagnosis), AD / PSP direction (DIRECTION-01), plasma sepsis (cfDNA specimen - composition + A per present cell), CRC and gastric (the untested stomach cells), the 200-array healthy panels per laboratory. Each gets: cohort on disk or a fetch script, a runner that writes one shard per array, bars, and a dispatch line — so the whole set is one submission and the results come back to the same outcome format.

## Reach (after the human chain is stable; author 2026-09-27: "I just don't want to limit the reach of this")

The argument is **one instrument, one fixed point, many species**: the same A against the same class floors, no population, read in
a human leukaemia, a dog lymphoma, an ageing Swede's neutrophils, a hatchery steelhead's red cells. Order by readiness: human blood
cancer (BLOODCANCER-01, public 450K, the chain runs today) → ageing track (SATSA, running) → dogs (mammalian array on GEO; canine H_min
already in the OM; dog lymphoma / osteosarcoma sets exist) → livestock (cattle, horse, pig in the mammalian consortium data: ageing and
stress more than cancer) → salmon (RRBS; new locus set; the largest build and the nearest audience). One pre-registration template for all.

22. **Mammalian array** — the Mammalian Methylation Consortium's conserved-CpG array (348 species, ~15k samples, much on GEO): the same identity-loci + H_min construction per species; dogs (Dog Aging Project) as the first non-human adopter — M_dog = ΔG_ATP/(RT) at 38.5 °C = 20.84 (human 20.94; the temperature, nothing else); the dog age chart from the the methylation report web source and `papers/iam_vertebrate_lifespan.tex` are the existing pieces. **A canine atlas is required** (mean/SD per CpG per cell type on the ~37k conserved array loci) — the same v2 build, hours not days; floors, identity-loci rule, gauge and chain unchanged. The limit is references: public canine array data is whole blood and tissue from the clock studies, not sorted leukocytes, so version 1 reads tissue-level classes (ageing track; lymphoma, where the tumour cell is in the blood) and not yet a per-cell composition. Inventory 2026-09-27: NO sorted-cell canine reference exists anywhere (a Dec-2025 canine lymphoma blood study: 'due to the lack of reference data on DNA methylation patterns of each cell type in dogs, we could not assess cell composition'). Public: 16-tissue MRE-seq reference (32 samples; needs a scale map), whole-blood WGBS across breeds (BMC Genomics 2024), human EPIC on dog blood (~34.6k probes hybridise; lymphoma 2019, mammary 2024, GI lymphoma 2025 - the cancer side). The sorted-leukocyte reference must be MEASURED: ~10 healthy dogs x 6 sorted populations x arrays (~60 arrays) with a vet school - a first-of-its-kind canine cell-type reference, publishable before any reading. Candidate route with no new samples (author 2026-09-27, 'filter the full genome that exists'): factorise the breed whole-blood WGBS (dozens of dogs, each a different leukocyte mixture) into shared source profiles, anchored by the human sorted-cell profiles lifted through orthologous CpGs - recovers per-cell means where cells differ (the identity loci and markers), not SDs; a DERIVED reference, labelled as such on every reading until the sorted arrays confirm it, which then become the confirmation study rather than the prerequisite. A cross-species test of the physics at the cellular scale.
23. **Salmonid chain** — RRBS, not arrays. Public: Methow River steelhead hatchery-vs-wild RBC and sperm RRBS (G3 2018; 85 RBC DMRs, 108 sperm DMRs); coho hatchery-vs-wild muscle (PNAS 2017); steelhead liver hatchery-vs-stream. Fish blood is nucleated red cells — one cell type, no composition problem. Needs: a salmonid locus set from RRBS coverage, reference methylomes per tissue, H_min per class fitted as G-002 did. First pre-registration: hatchery vs wild RBC per fish against the fixed point, on the Methow — the question Chelan PUD's biologists already ask in methylation terms.
24. **Open human data beyond GEO** — CALERIE, TRIIM/TRIIM-X and other academic intervention trials (public or on request); Framingham / WHI / Lothian / Generation Scotland / Dunedin serial methylation under controlled access (dbGaP / EGA) — apply once a pre-registered method is published; TCGA-LAML and GEO CLL/MDS for BLOODCANCER-01.

## Standing (not tasks)

- Healthy is A = 1.00; the tier scale is the tolerance. No population defines any number on a cell. MEASURE, DON'T COMPARE.
- A pre-registration is written before data is read; a failed bar is investigated before it is accepted — once is a result, twice is interesting, every time is a signal.
- Verify by reading the render; write the outcome after.
- Every push carries a copy of the changed files. Data worth keeping is bundled before it can be lost.
- A README describes a folder as it is. It is not a log.
