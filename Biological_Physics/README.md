# Biological Physics — Physics of Methylation: Landauer Metrology

**What this is.** An instrument that reads, for every cell type a specimen is found to contain, where that cell's write process
sits on its own gauge:

    A = H(mean β over the cell's identity loci) / H_min of its architecture class

`H` is binary Shannon entropy; `H_min` is the minimum entropy a cell of that class must hold to keep its identity — a frozen
physical constant, one per class, fitted once by MCMC on published reference cell methylomes (G-002, April 2026) and never
touched by a specimen. **Healthy is A = 1.00 by the physics.** The tier scale about it is the tolerance (NORMAL [0.95, 1.05),
ELEVATED [1.05, 1.07), the Warburg line at 1.07, BREACH at 1.10; SUPPRESSED below 0.95). Nothing about any other person enters a
reading. MEASURE, DON'T COMPARE.

Prior art: Sanchez & Mackenzie (2016) established that cytosine methylation obeys Landauer's bound and used it as a filter for
thermal background. This work uses the same bound as a ruler: the floor is the zero of the instrument, not the noise to remove.

**New here? Read [`MethylPhys/START_HERE.md`](MethylPhys/START_HERE.md).** It says which document is canonical for what and gives the
one command that regenerates and gates all of them.

## Two rules

1. **Measure, don't compare.** A cell is read against its own class floor, never against a population. No cohort, no healthy
   band, no age curve, no laboratory zero, no reference range enters a reading. Where people of an age sit on the gauge is an
   observation about people and is reported as such — never applied to a cell. A render-time vocabulary guard fails any report
   tab that names a population; the same guard runs over the SOP, the Operations Manual and the papers on every build.
2. **No foregrounds subtracted.** The chain subtracts no age / sex / smoking / batch term. Intake facts are annotations on the
   report, never operands in the score.

## Layout

| folder | what it is |
|---|---|
| [`MethylPhys/`](MethylPhys/) | **the instrument** — chain, kit, SOP, Operations Manual, report interface, papers, runtime constants |
| [`MethylPhys/chain/`](MethylPhys/chain/) | the running code: Stage 0 intake, Stage 1 calibration, the pipeline map, composition, per-cell A, tiers, the sky, the report. [`chain/README.md`](MethylPhys/chain/README.md) |
| [`MethylPhys/atlas/`](MethylPhys/atlas/) | IAMAtlasREBUILD — 483,092 CpGs × 115 cell types, per-locus posterior mean and sd, and the scripts that built it. [`atlas/README.md`](MethylPhys/atlas/README.md) |
| [`MethylPhys/kit/`](MethylPhys/kit/) | conformance tests, the release check, the sealed procedures (`PROC_*.py`) and their results, the document generators |
| [`MethylPhys/sop/`](MethylPhys/sop/), [`MethylPhys/manual/`](MethylPhys/manual/) | the SOP (mirrors the code: a STATUS banner on every section from a code-derived table) and the Operations Manual (PDF, built from the runtime files) |
| [`MethylPhys/doors/`](MethylPhys/doors/) | pre-registrations, outcomes, findings, the plan and the ledger, the runbook, the generated inventories |
| [`MethylPhys/papers/`](MethylPhys/papers/) | the methods paper (*Physics of Methylation: Landauer Metrology*) and the programme document; sources and the author's compiled PDFs |
| [`MethylPhys/reference_data/`](MethylPhys/reference_data/) | the Stage-1 calibrated arrays the instrument constants were measured on, shipped so a reader can recompute them |
| [`Record/`](Record/) | **the pre-chain record**: the April–June validation runs (VAL-001 … VAL-141), cohort manifests and extraction scripts. Education, not evidence: nothing in it is defended, and nothing in the live chain reads it. [`Record/README.md`](Record/README.md) |
| [`RETIRED_2026-09/`](MethylPhys/RETIRED_2026-09/) | material removed from the chain in September 2026, kept for the record with a README per folder saying what it was and why it was taken out. Nothing live reads it. |

## Verify, then run

```
cd Biological_Physics/MethylPhys
python3 kit/release_check.py                       # every guard, one command
python3 chain/build_all.py                          # regenerate every derived document and run the three gates
python3 chain/MethylPhys_Interface/run_sample.py --grn X_Grn.idat.gz --red X_Red.idat.gz --age 58 --sex F --lab MYLAB --out report.html
```

Every push runs `chain/guarded_push.sh`, which runs `build_all.py` and the propagate gate first: a hand edit to a generated
document is overwritten before it can be committed. Never edit a generated document; fix the source it names.

## Where the history is

What was built, measured, removed and why is in [`MethylPhys/doors/`](MethylPhys/doors/) — one pre-registration and one outcome
per procedure, [`CHAIN_COMMISSIONING.md`](MethylPhys/doors/CHAIN_COMMISSIONING.md) as the register, [`ENHANCEMENTS.md`](MethylPhys/doors/ENHANCEMENTS.md)
as the ledger, [`PLAN.md`](MethylPhys/doors/PLAN.md) as what is next — and in the git log. This README describes the tree; it is
not a log.

*Research and development stage. Nothing in this tree is clinical validation.*
