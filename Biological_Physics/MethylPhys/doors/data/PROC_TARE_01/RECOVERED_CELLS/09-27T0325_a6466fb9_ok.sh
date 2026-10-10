cd iamrepo/Biological_Physics/MethylPhys && cat > doors/PLAN.md <<'EOF'
# PLAN — what we do next, in order

One line per item. No history, no strikes. When an item is done it comes OFF this page and its record goes to
`ENHANCEMENTS.md` (the long ledger) and the register. Rule for the order: chain before documents, documents before
launcher, and nothing edits the chain while a procedure is scoring.

## Now (in flight)

1. **Report audit, round 3** — 16-tab verdict table; pushed once the release check passes at this commit.
2. **PROC-TARE-01** — SNP-probe tare, 768 arrays; sealed as measured when it lands (B2 likely fails as written).

## Chain — after TARE-01 finishes, one at a time

3. **Deferred chain patch** — four cohort stages out of `run_full`; `Patient_CMB` onto the search path.
4. **PROC-UNMIX-01** — re-zero the identity loci so the standard reads 1.000, then invert the dilution line per present cell; six bars pre-registered.
5. **Sky zero and spread from the atlas posterior** — a constructed atlas specimen must read quiet; the four panels must stay at 2.6-3.2 %.
6. **Held-out Stage 2d** — per-array shards, run alone, then register row B-12 says verified or not.
7. **Stage 2d kit test** — commissioned panels never fire more than 1-in-n; no panel → "not commissioned", nothing else.
8. **Twin/family thresholds as a runtime matrix** — no constants in code.
9. **Chip term** — read TARE-01 B6; if the tare does not remove it, an on-chip reference (control probes) does.
10. **Serial mode** — `run_sample.py --prior <bundle>`: same patient, per-cell ΔA, Δfraction, difference sky; change floor pre-registered.

## Documents — written once, after the chain above is still

11. **SOP** — LESSON-DECON-01, Stage 2d and its commissioning rule, TARE/UNMIX outcomes, pre-registration conventions; every runtime file named from the tree.
12. **Operations Manual** — engine section generated from `chain_sequence.json`; then every rendered page read against a checklist; per-tab figures regenerated from the audited report.
13. **The cells** — an OM section, one entry per scoreable cell, rewritten from the webpage drafts onto the physics (no cohort range, no wellness framing); welcome-page explanations folded into OM front matter and the Physics/How-to tabs where they add something.
14. **Documentation catch-up** — manifest, component map, runbook, chain sequence, inventory naming the per-cell surface, the reference folder, the solve block and twin rules (mostly regenerates; read anyway).
15. **START_HERE.md** — what this is, which document is canonical for what, the one command that verifies the chain.

## The face of the chain

16. **Launcher** — local `run.py`: verify files by hash, take the IDAT pair, run stage by stage with live pass/fail, open the report; replication menu for sealed PROCs.

## Procedures — once the chain is fully commissioned

17. **PROC-BRAIN-01 redo** — clean single-provenance CSF run.
18. **Commission a solid-tissue laboratory** — pipeline map and floors; gates the glioma and progression re-tests.
19. **Gastric** — six stomach entries, two families, never tested.
20. **Breast shedding in real patient blood** — the question that started the detection work.
21. **Re-run the webpage-draft VALs (CRC, breast, HCC) as pre-registered PROCs** — education re-tested on the commissioned chain.

## Later chain work

22. CD4/CD8 separation — needs loci this block lacks (second blood block).
23. Second-block solve for the nine finer blood subsets.
24. EPIC platform block (865k loci).
25. Coverage floor on any class-selection rule.
26. **ENHANCEMENTS re-plan** — once 1-16 are done, rewrite the ledger as one ordered plan for the remaining CMB borrowings, atlas duplicates and the EPIC block.

## Standing (not tasks)

- Every push carries a copy of the changed files. Data worth keeping is bundled before it can be lost.
- Healthy is A = 1.00; the tier scale is the tolerance. No cohort defines any number on a cell.
- A pre-registration is written before data is read and does not move afterwards.
- Verify by reading the render; write the verdict after.
EOF
python3 - <<'PY'
p="doors/ENHANCEMENTS.md"; s=open(p,encoding="utf-8").read()
if "PLAN.md" not in s[:2000]:
    i=s.find("## Standing to-do, 2026-09-26")
    s=s[:i]+"> **The clean ordered list is [`PLAN.md`](PLAN.md).** This file is the ledger — what was found, what was decided, what was closed and why. Read PLAN.md for what comes next.\n\n"+s[i:]
    open(p,"w",encoding="utf-8").write(s); print("ENHANCEMENTS points at PLAN.md")
PY
cp doors/PLAN.md ../../../PLAN.md