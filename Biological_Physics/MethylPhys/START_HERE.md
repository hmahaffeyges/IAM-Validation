# START HERE - MethylPhys CPG

**What this is.** An instrument that reads, for every cell type a specimen is found to contain, where that cell sits on its own
gauge: A = H(mean beta over the cell's identity loci) / H_min of its architecture class. Healthy is A = 1.00 by the physics; the
tier scale about it is the tolerance. Nothing about any other person enters a reading. MEASURE, DON'T COMPARE.

**Which document is canonical for what**

| question | document | how it is kept current |
|---|---|---|
| what does the chain do, step by step, today | [`sop/MethylPhys_CPG_SOP.md`](sop/MethylPhys_CPG_SOP.md) Part II-A + the STATUS banner on every section | generated from `chain/chain_sequence.json` by `sop/build_sop_mirror.py` |
| the physics, the record, every ruling and procedure | [`manual/MethylPhys_CPG_Operations_Manual.pdf`](manual/MethylPhys_CPG_Operations_Manual.pdf) | built by `manual/build_om.sh` from `manual/om_data.py` + the runtime files |
| what a report says and why | the report itself (17 tabs), described tab by tab in [`doors/REPORT_TAB_REFERENCE.md`](doors/REPORT_TAB_REFERENCE.md) | rendered fresh from one repository-shipped array on every build |
| what is in the tree | [`doors/REPO_INVENTORY.md`](doors/REPO_INVENTORY.md), [`doors/COMPONENT_MAP.md`](doors/COMPONENT_MAP.md) | measured from `git ls-files` and the chain inventory |
| how to run it | [`doors/RUNBOOK.md`](doors/RUNBOOK.md) | the chain table inside it is generated |
| what is next | [`doors/PLAN.md`](doors/PLAN.md) (one line per item); [`doors/ENHANCEMENTS.md`](doors/ENHANCEMENTS.md) is the ledger | by hand |
| the methods paper | [`papers/Physics_of_Methylation__Landauer_Metrology.pdf`](papers/Physics_of_Methylation__Landauer_Metrology.pdf) | author's Overleaf build of `papers/Landauer_Metrology_of_the_Methylome.tex` |

**The one command.** `python3 chain/build_all.py` regenerates every document above from the chain files and refuses (exit 1) if the
report's vocabulary guard, the SOP-mirror reconciliation or the link check fails. `chain/guarded_push.sh` runs it before every
push. Never edit a generated document; fix the source it names.

**To verify the chain before trusting it:** `cd kit && python3 release_check.py` (every guard, one command).

**To read one specimen:** `python3 chain/MethylPhys_Interface/run_sample.py --grn X_Grn.idat.gz --red X_Red.idat.gz --age 58 --sex F --lab MYLAB --out report.html`
