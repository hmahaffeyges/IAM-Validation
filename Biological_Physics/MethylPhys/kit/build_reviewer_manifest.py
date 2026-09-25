#!/usr/bin/env python3
"""Regenerate doors/REVIEWER_MANIFEST.md from the tree.

The manifest is the page whose whole purpose is that a reviewer can find every file without hunting, so a
stale one is worse than none. It was written by hand on 2026-09-22 and by the next day it listed none of the
Stage 0 files - the module that decodes the intake QC inputs, the three procedure documents, the per-array
evidence, the step-order derivation. Hand-written pages drift; this one is generated.

Rules that make it trustworthy:
  * every path is resolved by basename from the live tree, never typed - a guessed path was wrong six times
    out of six the first time this page was built
  * a file the group names but the tree does not carry is listed as ABSENT, not silently dropped
  * the procedure counts are computed from one enumeration and asserted against the rows actually written,
    so the headline cannot disagree with the body (it did, on 2026-09-22)

Usage: python3 build_reviewer_manifest.py   [from kit/]
"""
import collections
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True).stdout.strip()
B = "Biological_Physics/MethylPhys"
OUT = os.path.join(ROOT, B, "doors/REVIEWER_MANIFEST.md")

GROUPS = [
 ("The instrument a reviewer would run", [
  ("run_sample.py", "one sample end to end: the intake steps, calibration, the eleven conductor stages, the report"),
   ("cmb_tools.py", "the register of every method borrowed from CMB analysis, with a check per tool "
    "that returns its state on a finished bundle"),
   ("walther_clinical.py", "the batch path: its own stage functions, used for a cohort "
    "rather than a single specimen"),
  ("cpg_conductor.py", "the orchestrator - every stage in call order"),
  ("stage_0_intake.py", "the intake steps (SOP 11-19) and the decision that stops the chain"),
  ("stage_0_1_qc_handoff.py", "decodes the control probes, negative controls, bead counts and chrX/chrY the intake QC steps read"),
  ("stage_1_idat_calibration.py", "a raw IDAT pair to noob-calibrated betas; the array type is read from the header"),
  ("walther_iam_deconvolver.py", "composition and presence against the atlas"),
  ("nilc_celltype_deconvolver.py", "the second, independent separation used as a class-level agreement check"),
  ("iamatlas_a_scoring.py", "the gauge itself: measured entropy over the calibrated floor"),
  ("stage_4_6_patient_cmb.py", "the residual sky for one specimen"),
  ("build_methylphys.py", "the report: every tab, including Troubleshooting"),
  ("build_chain_sequence.py", "derives the step order from the code - run it if you move a file"),
  ("chain_sequence.json", "that derivation, machine-readable; the report and the SOP read it"),
  ("run_batch.py", "the second interface, a folder at a time"),
 ]),
 ("The constants it divides by, and the MCMC that produced them", [
  ("gape_mcmc_g002.py", "the sampler that fitted the eight class floors"),
  ("g003_mcmc_framework.py", "the framework G-003b runs on"),
  ("gape_mcmc_g003b.py", "the floor posterior used by the chain"),
  ("gape_mcmc_g008.py", "the substrate sampler"),
  ("gape_bootstrap_comparison.py", "the bootstrap cross-check, and what it does not cover"),
  ("reference_cells_37.csv", "the 37 reference cells the floors were fitted on"),
  ("REPRODUCTION.md", "re-run it and compare: every floor lands inside its own posterior SD"),
  ("reproduce.sh", "the runner"),
  ("requirements.txt", "the pinned environment"),
 ]),
 ("The runtime matrices every reading is corrected by", [
  ("iamatlas_gauge_identity_loci_v1_0.json", "the identity loci per class, with the floor each divides by"),
  ("beta_scale_maps_v1.json", "one affine map per pipeline; without it a reading is UNMAPPED and not reportable"),
  ("reference_age_curve_v1.json", "the age reference every reading is corrected against"),
  ("tier_breakpoints.json", "the tier boundaries"),
  ("identity_band_v3.json", "the healthy band per class, with each laboratory's own false-alarm rate"),
  ("percell_reference_v0_3.json", "the per-entry reference, with which entries are resolvable"),
 ]),
 ("The atlas", [
  ("IAMAtlasREBUILD.csv.xz", "the atlas itself, compressed"),
  ("IAMAtlasREBUILD_celltype_to_class.json", "cell type to class"),
  ("IAMAtlasREBUILD_provenance.json", "where every entry came from"),
 ]),
 ("Stage 0: the gate before the measurement", [
  ("PROC_STAGE0_02_PREREG.md", "pre-registered before any array was read: the bars, the decision rule, the limitation"),
  ("PROC_STAGE0_02_OUTCOME.md", "the sealed answer over 732 arrays, with every gate's distribution"),
  ("PROC_STAGE0_02.json", "per-array: every status, every flag, every metric"),
  ("PROC_STAGE0_02_betamean_GSE87571.json", "the calibrated per-array means the sealed set was drawn from"),
  ("PROC_STAGE0_02_arrival.py", "the arrival and manifest pass over all 732 pairs"),
  ("PROC_STAGE0_02_sweep.py", "the QC decode sweep, cached per array so it resumes"),
  ("PROC_STAGE0_02_seal.py", "the gates run on the cached metrics, and the answer computed"),
  ("PROC_STAGE0_04_PREREG.md", "the bisulfite threshold, left to the author with the healthy distribution published"),
  ("MethylPhys_GSM2333905_with_intake.html", "one real run end to end, intake record included"),
 ]),
 ("The procedure, and the documents that keep it honest", [
  ("CPG_Chain_of_Custody_SOP_v2_0_0.md", "the procedure: every stage and step links its code, and the intake steps carry their thresholds, refusal strings and healthy distributions"),
  ("sop_stage_links.py", "writes the implemented-in line under every stage and step section"),
  ("sop_step_detail.py", "writes the operational detail under each intake step"),
  ("sop_repoint.py", "regenerates the SOP and resolves paths from earlier layouts"),
  ("CHAIN_SEQUENCE.md", "the live step order, generated - if a document disagrees with it, the document is wrong"),
  ("RUNBOOK.md", "how to run one sample"),
  ("CHAIN_COMMISSIONING.md", "the register: every row, its status, and what is still open"),
  ("link_check.py", "every relative path in every live document resolves"),
  ("add_doc_links.py", "turns filenames written in prose into links, once per section"),
  ("release_check.py", "the guards, each with its result; a guard that could not run prints NOT RUN"),
 ]),
 ("The record, including what failed", [
  ("DETAILED_VALIDATION_RECORD.md", "the long-form record"),
  ("VAL_INDEX.csv", "every validation row and its status"),
  ("chain_inventory_v1.json", "every chain file with its role"),
  ("ROW9_WORKING_NOTE.md", "the working note: what was tried, what failed, what was withdrawn"),
 ]),
]

NOT_PUBLISHED = [
 ("Raw MCMC posterior chains",
  "they do not exist: the samplers hold their samples in memory and print summaries, so no run wrote an "
  ".h5 or .npy. What exists instead is a re-run that returns every floor inside its own posterior SD "
  "(10.5281/zenodo.22905819)"),
 ("Controlled-access cohort data",
  "named as requirements, not held: the fragmentomics cohorts are controlled access and the chain's "
  "predictions for them are filed against data nobody here has"),
 ("Disease evidence from the commissioned chain",
  "it does not exist yet - Issue 004, after sealed runs. The April 2026 evidence database is published "
  "unedited but measures a different surface and must not be quoted beside Issue 003"),
 ("Per-cell reporting below class level",
  "withheld by the instrument. Per-entry resolvability is published so a reader sees which entries are "
  "affected and why"),
 ("The bisulfite threshold",
  "not set: every healthy array in the reference cohort sits below the SOP's 0.95, so the gate reports its "
  "measured value and does not refuse on it until the threshold is calibrated (PROC-STAGE0-04)"),
]


def index():
    tracked = [f for f in subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True,
                                         text=True).stdout.split("\n") if f]
    untracked = [f for f in subprocess.run(["git", "-C", ROOT, "ls-files", "--others",
                                            "--exclude-standard"], capture_output=True,
                                           text=True).stdout.split("\n") if f]
    by = collections.defaultdict(list)
    for f in tracked + untracked:
        if "RETIRED" in f or "author_copies" in f:
            continue
        by[os.path.basename(f)].append(f)
    return by


def rel(path):
    return os.path.relpath(os.path.join(ROOT, path), os.path.join(ROOT, B, "doors")).replace(" ", "%20")


def main():
    by = index()
    L = ["# What a reviewer can download, and what we have not published", "",
         "Every path below was resolved by looking the name up in the tree, not typed - which is why it has no "
         "dead links. This page is generated by "
         "[`build_reviewer_manifest.py`](../kit/build_reviewer_manifest.py); run it after adding a file that a "
         "reviewer would want, and [`link_check.py`](../kit/link_check.py) will tell you if anything here has "
         "moved.", ""]
    absent, listed = [], 0
    for title, items in GROUPS:
        L += ["## " + title, ""]
        for name, what in items:
            hits = by.get(name, [])
            if not hits:
                absent.append((title, name))
                L.append("- `" + name + "` \u2014 " + what + " **(ABSENT from the tree)**")
                continue
            hits.sort(key=lambda p: (0 if "/MethylPhys/" in p else 1, len(p)))
            L.append("- [`" + name + "`](" + rel(hits[0]) + ") \u2014 " + what)
            listed += 1
        L.append("")

    procs = collections.defaultdict(list)
    for base, paths in by.items():
        for p in paths:
            m = re.search(r"PROC[-_]([A-Z0-9]+)[-_](\d+)", p)
            if m:
                procs["PROC-%s-%s" % (m.group(1), m.group(2))].append(p)
    nfiles = sum(len(v) for v in procs.values())
    L += ["## Every sealed procedure \u2014 %d procedures, %d files" % (len(procs), nfiles), "",
          "Pre-registration, outcome and evidence for each. The failures are here too: a procedure that "
          "closed NOT COMMISSIONED is as much a result as one that passed.", ""]
    rows = 0
    for k in sorted(procs):
        files = sorted(set(procs[k]))
        L.append("- **" + k + "** \u2014 " + ", ".join(
            "[`" + os.path.basename(f) + "`](" + rel(f) + ")" for f in files))
        rows += 1
    assert rows == len(procs), "the headline count must equal the rows written: %d vs %d" % (len(procs), rows)
    L.append("")

    L += ["## What is NOT published, and why", "",
          "A reviewer should learn this from the page, not by hunting.", "",
          "| not published | why |", "|---|---|"]
    for a, b in NOT_PUBLISHED:
        L.append("| " + a + " | " + b + " |")
    L += ["",
          "## Two scope limits to read before citing anything", "",
          "1. **The bootstrap comparison does not cover methylation.** Its 32 rows are eight classes by four "
          "non-methylation substrates. The eight methylation floors the chain divides by rest on the "
          "sampler's own convergence and its 37-cell reference database.",
          "2. **The April 2026 evidence database is a different surface.** Marker-union, not identity loci - "
          "and the two move in opposite directions with age. Its numbers must never be quoted beside Issue "
          "003's.", "",
          "## The order of steps", "",
          "[`CHAIN_SEQUENCE.md`](CHAIN_SEQUENCE.md) is generated from the code and lists both interfaces, "
          "every live step in order, and anything documented as a chain step that nothing calls. If a "
          "document disagrees with it, the document is wrong.", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print("manifest: %d files listed, %d procedures (%d files), %d absent"
          % (listed, len(procs), nfiles, len(absent)))
    for t, n in absent:
        print("   ABSENT:", t, "->", n)


if __name__ == "__main__":
    main()
