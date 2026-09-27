# PLAN — what we do next, in order

One line per item. No history, no strikes: when an item is done it comes OFF this page; its record is the register
([`CHAIN_COMMISSIONING.md`](CHAIN_COMMISSIONING.md)), the ledger ([`ENHANCEMENTS.md`](ENHANCEMENTS.md)) and the procedure's own outcome file.
Order rule: chain before documents, documents before launcher, and nothing edits the chain while a procedure is scoring.

## Housekeeping first (today showed both are needed)

1. **Runner convention** — every long runner takes its own output dir, log and results name per launch and polls a STOP file; the sandbox cannot kill a process from a later cell, so "kill and relaunch" is never done again.
2. **Bundle at the end of every session** — chain files changed that day, the generators, and any new data, as three zips with SHA256SUMS.

## Chain — one at a time

3. **PROC-SKY-01 follow-up** — re-run the sky script with Munich refused at intake. The sky is currently DRAWN on the no-population σ (atlas posterior + this array's SNP noise) — my call under the development-stage ruling, not the author's explicit pick. **Author's open decision**: keep it drawn as z on that σ, or draw the raw residual in β units with no σ at all (plus the serial difference map, item 8).
4. **Stage 2d rebuilt** — FINDING_DETECTION_PANEL_HELDOUT.md (732 arrays: 48 % false alarms, 24 % 'substrate mismatch', σ 2.6× too small). Pre-register: per-array common-mode removal; lines re-measured on the admitted set with a stated quantile — **author's decision: is a noise floor measured on arrays known to lack the cell acceptable, or must the line come from the array itself (SNP noise)?**; thin-source cells print NOT DETECTABLE until atlas v2. Until then the detector is not trusted on any specimen and the report says so.
5. **Stage 2d kit test** — contract from the finding: per-cell FP ≤ 1 % on OK arrays, UNSPECIFIC ≤ 1 % of healthy arrays, thin-source cells 'not detectable'.
6. **Twin/family thresholds as a runtime matrix** — no constants in code (same move as the intake thresholds).
7. **Chip term** — the SNP tare did not remove it (TARE-01 B6); the control-probe model (0.002–0.018 on the clean laboratories) is the recorded route. Pre-register; run on the 768 calibrated arrays.
8. **Serial mode** — `run_sample.py --prior <bundle>`: same patient, per-cell ΔA, Δfraction, difference sky; change floor pre-registered. This is what the sky is for.
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
20. **Atlas v2 — joint hierarchical build** (November machine or a rented 128 GB box, ~1–2 days). One MCMC over all cells with a shared per-locus probe effect, a per-source effect and a per-cell effect; undefined loci imputed from class and nearest cells with wide SD; cross-cell covariance kept; H_min untouched. Acceptance written before the run: PROC-COV-01 misfit (0.067) below the NORMAL tolerance; twins collapse to one cell effect; a 252-locus cell's imputed loci carry SD > 0.1. Adding a cell from a new source (Loyfer 2023, the 476-methylome set, the July 2026 single-cell atlas) becomes one script: scale map → coverage family → twin test → identity loci → reads 1.000 on itself. Its covariance is the noise model the matched filter lacked and the solver item 9 needs.
21. **Ledger re-plan** — once 1–10 are done, rewrite ENHANCEMENTS as one ordered plan for what remains.

## Standing (not tasks)

- Healthy is A = 1.00; the tier scale is the tolerance. No population defines any number on a cell. MEASURE, DON'T COMPARE.
- A pre-registration is written before data is read; a failed bar is investigated before it is accepted — once is a result, twice is interesting, every time is a signal.
- Verify by reading the render; write the outcome after.
- Every push carries a copy of the changed files. Data worth keeping is bundled before it can be lost.
- A README describes a folder as it is. It is not a log.
