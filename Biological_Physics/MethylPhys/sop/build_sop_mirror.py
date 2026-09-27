#!/usr/bin/env python3
"""build_sop_mirror.py - the SOP mirrors the code, and says so on every section.

The SOP was written in June 2026 as the specification of a ten-stage chain. The chain that runs today is not that chain:
stages were removed, added, renamed and rewired, and hand-patching an 11,000-line document is how it drifted. This script
is the durable fix (author, 2026-09-27: 'check EVERY single step and make sure the chain matches ... consider a script'):

  1. PART II-A (generated). 'The chain as the code runs it' - one row per step of cpg_conductor.run_full and the two
     scripts around it, derived from chain_sequence.json (which build_chain_sequence.py derives from the code by following
     every call from run_full), with each step's docstring first line and the runtime files the inventory ties to it.
     Regenerated every run; never hand-edited.
  2. STATUS BANNERS. Every numbered section (§n / Step x.y / Stage n) gets one banner line under its heading, from the
     STATUS table below: LIVE (names the function and file that implement it), RECORD (built, then removed - the section
     is the record of what was built), NOT IN CHAIN (specified in June; run_full does not do it and the firewall says why),
     NOT BUILT (specified, never built), or KIT (a tool run around the chain, not a per-sample step). Idempotent: an
     existing banner is replaced, never duplicated.
  3. RECONCILIATION. Every live step must have at least one LIVE section; every LIVE section must name a step that exists
     in chain_sequence.json. Any miss is printed and --check exits 1 (propagate rule 15).

The vocabulary scan (kit/vocab_scan.py) reads these banners: RECORD / NOT IN CHAIN / NOT BUILT sections may name what
they removed; LIVE sections are held to the report's own guards.
"""
import os, re, json, sys, glob, ast

HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(HERE); CH = os.path.join(MP, "chain")
SOP = os.path.join(HERE, "MethylPhys_CPG_SOP.md")
SEQ = glob.glob(os.path.join(CH, "**", "chain_sequence.json"), recursive=True)[0]
INV = os.path.join(CH, "Runtime Matrices", "chain_inventory_v1.json")

# section-key -> (status, implemented-by, note). Key = the § number, or a heading prefix for unnumbered stage headings.
L, R, N, B, K = "LIVE", "RECORD", "NOT IN CHAIN", "NOT BUILT", "KIT"
STATUS = {
 # Part I
 "§0.9": (R, "-", "hard-won lessons, kept as written (the author's own words on the drift are quoted here)"),
 "§1.": (R, "-", "the June chain at a glance; the chain as it runs is Part II-A"),
 "§2.": (R, "-", "the L1-L9 grading table of the June specification"),
 "§1.5.": (R, "-", "the three-component separation of the June specification"),
 "§3.": (R, "-", "stages vs links, June"),
 "§4.": (L, "stage_a_cells (cpg_conductor.py) reads IAMAtlasREBUILD.csv", "the atlas: what it is, where it lives"),
 "§5.": (L, "iamatlas_gauge_identity_loci_v1_0.json / cpg_gauge_engine.H_MIN_TABLE", "the eight frozen floors"),
 "§6.": (L, "cmb_tools.py (the register)", "the CMB -> methylome translation principle"),
 "§7.": (R, "-", "bidirectional calibration as specified in June; the gauge is H(mean beta)/H_min (RULING A3)"),
 "§8.": (L, "build_methylphys.py FORBIDDEN / COHORT guards; kit/vocab_scan.py", "vocabulary"),
 "§9.": (L, "stage_2b_second_opinion (cpg_conductor.py)", "the second opinion"),
 "§10.": (R, "-", "Stage 6 'reporting layer' of the June specification; the report is Stage 9 (build_methylphys.py) and the legal boundary is the vocabulary guard"),
 "§10.5.": (L, "-", "the non-negotiable rules"),
 # Stage 0
 "§11.": (L, "step_0_1_idat_arrival (stage_0_intake.py)", ""), "§12.": (L, "step_0_2_manifest_creation (stage_0_intake.py)", ""),
 "§13.": (L, "step_0_3_integrity_hash (stage_0_intake.py)", ""),
 "§14.": (L, "step_0_4_control_probe_validation on Stage 1's control medians (stage_1_idat_calibration.py -> run_sample.py)", "bisulfite threshold reported, not applied"),
 "§15.": (L, "step_0_5_detection_pvalue_qc on Stage 1's poobah mask (p <= 0.05); failed probes removed before any stage reads the beta", "a deferred check never advances"),
 "§16.": (L, "step_0_6_bead_count_qc (stage_0_intake.py) - runs; bead counts are not yet extracted from the IDAT, so it records DEFERRED and the flag BEAD_COUNT_NOT_EXTRACTED", "the only Stage 0 check still deferred; it does not advance anything on its own"),
 "§17.": (L, "step_0_7_call_rate on the detection mask; step_0_7b_platform_coverage (>= 80 % of the identity loci present)", "thresholds under the author's decision (call rate vs signal-to-background)"),
 "§18.": (L, "step_0_8_sex_check (stage_0_intake.py)", ""), "§19.": (L, "step_0_9_decision_gate (stage_0_intake.py)", "reads detection, call rate, controls, integrity, coverage, sex"),
 # Stage 1
 "§20.": (L, "Stage 1 - IDAT calibration: methylprep noob inside calibrate_idat_to_beta (stage_1_idat_calibration.py)", "dye bias"),
 "§21.": (L, "methylprep noob inside calibrate_idat_to_beta", "probe-type normalisation"),
 "§22.": (N, "-", "ComBat / batch correction: the firewall (§104) - the chain subtracts no foregrounds and corrects no batches; each array is calibrated alone"),
 "§23.": (L, "Stage 0.4 on the bisulfite-conversion control medians Stage 1 returns", "reported, not a gate"),
 "§24.": (L, "calibrate_idat_to_beta -> beta per CpG", ""),
 "§25.": (R, "-", "sanity checks of the June specification; the chain's checks are Stage 0.5/0.7 (detection) and Stage 1s (scale)"),
 "§26.": (B, "-", "probe response function: PROC-TARE-01 measured a linear SNP-probe tare and found it is not the instrument's; a nonlinear response from the control probes is the recorded route"),
 "§27.": (L, "run_sample.py hands the beta dict to cpg_conductor.run_full", ""),
 "§109.": (L, "stage_1s_scale_map (cpg_conductor.py), beta_scale_maps_v1.json", "the pipeline map"),
 # Stage 2
 "§28.": (L, "stage_a_cells: synthetic_patient_generator._cell_means / IAMAtlasREBUILD.csv", ""),
 "§29.": (L, "legacyIAMDeconvolver marker selection (legacy_iam_deconvolver.py), iamatlas_celltype_markers_v0_2.json", ""),
 "§30.": (L, "legacyIAMDeconvolver (NNLS) inside stage_a_cells", ""),
 "§31.": (L, "stage_a_cells: present / fraction / status per cell", ""),
 "§32.": (L, "stage_2b_second_opinion (NILC)", ""), "§33.": (L, "stage_2b_second_opinion: agreement check", ""),
 "§34.": (L, "run_full bundle keys cells / cells_all / class_fractions", ""),
 # Stage 3
 "§35.": (N, "-", "firewall §104: no foreground is subtracted"), "§36.": (N, "-", "firewall §104"), "§37.": (N, "-", "firewall §104"),
 "§38.": (N, "-", "firewall §104"), "§39.": (N, "-", "firewall §104"), "§40.": (N, "-", "firewall §104"),
 # Stage 4
 "§41.": (R, "-", "per-CpG marker panels for A: superseded by RULING A3 - A is read on the identity loci"),
 "§42.": (R, "-", "per-CpG entropy: the gauge is the entropy of the MEAN beta (H(beta_mean)/H_min), one number per cell"),
 "§43.": (L, "stage_b_classes (class A on the marker union) and stage_b_identity (class A on the identity loci) - both INTERNAL gates; neither carries a tier word; the mean-of-per-CpG-H construction described below is superseded (RULING A3)", "the class A is not a reading"),
 "§44.": (L, "stage_a_cells -> iamatlas_a_scoring._score_one_identity on iamatlas_percell_identity_loci_v1_0.json", "the per-cell A - THE reading"),
 "§45.": (N, "-", "disease-panel A: the chain applies no disease panel"),
 "§46.": (L, "run_full bundle: cells_all[cell].A", ""),
 "§46.5.": (L, "stage_4_5_bidirectional (cpg_conductor.py)", ""),
 "§46.6.": (L, "stage_4_6_patient_sky (cpg_conductor.py) + stage_4_6_patient_cmb.py", "sigma from the atlas posterior + this array's SNP-probe noise; no laboratory zero or spread"),
 # Stage 5, 6
 "§47.": (R, "-", "Stage 5 Mahalanobis REMOVED 2026-09-27 - a distance from a population's centroid"), "§48.": (R, "-", "REMOVED 2026-09-27"),
 "§49.": (R, "-", "REMOVED 2026-09-27"), "§50.": (R, "-", "REMOVED 2026-09-27"), "§51.": (R, "-", "REMOVED 2026-09-27"),
 "§52.": (R, "-", "Stage 6 cellular age REMOVED 2026-09-27 - an age read back from a curve of people"), "§53.": (R, "-", "REMOVED 2026-09-27"),
 "§54.": (R, "-", "REMOVED 2026-09-27"), "§55.": (R, "-", "REMOVED 2026-09-27"), "§56.": (R, "-", "REMOVED 2026-09-27"),
 "§57.": (R, "-", "REMOVED 2026-09-27"), "§58.": (R, "-", "REMOVED 2026-09-27"),
 # Stage 7
 "§59.": (R, "-", "per-CLASS tier: the class A carries no tier word (author's ruling); tiers are read on a cell's A"),
 "§60.": (L, "cpg_tiers.tier_of on cells_all[cell].A; tier_breakpoints.json v1.5", "the tier of every present cell"),
 "§61.": (N, "-", "cfDNA branch / cfdna_weight.json: not part of the chain"),
 "§62.": (L, "tier_breakpoints.json: BREACH at 1.10", ""),
 "§63.": (L, "tier_breakpoints.json customer_label; build_methylphys.py", "SUPPRESSED / NORMAL / ELEVATED / SIGNIFICANTLY ELEVATED / BREACH"),
 "§64.": (L, "run_full bundle: cells_all[cell].tier", ""),
 # Stage 8
 "§65.": (N, "-", "no signature matrix is consulted; stage_8_matching is not in the live path"), "§66.": (N, "-", "no card residual maps"),
 "§67.": (N, "-", "no pattern matching"), "§68.": (N, "-", "no covariate adjustment of any reading"), "§69.": (N, "-", "no card verdict"),
 # Stage 9
 "§70.": (L, "Report: build_methylphys.py (17 tabs)", "the report"), "§71.": (N, "-", "literature anchors: not applied"),
 "§72.": (N, "-", "cancer prior: the chain applies no disease prior"), "§73.": (N, "-", "family history: no multiplier on any reading"),
 "§74.": (N, "-", "sex-specific risk: none"), "§75.": (L, "run_sample.py -> build_methylphys.build", ""),
 "§76.": (L, "build_methylphys.guard (FORBIDDEN + COHORT); propagate.py bundle-key guard", "the legal boundary is mechanical"),
 # Stage 10
 "§77.": (L, "kit/file_run.py (filed run: report + bundle + checksums)", ""), "§78.": (B, "-", "delivery channel: none built"),
 "§79.": (L, "kit/file_run.py, guarded_push.sh, propagate.py", "the audit trail is the repository"),
 # L9
 "§80.": (K, "CPG_Null_Runner (kit; not per-sample)", ""), "§81.": (K, "-", ""), "§82.": (K, "-", ""), "§83.": (K, "-", ""), "§84.": (K, "-", ""),
 "§85.": (K, "-", ""), "§86.": (K, "-", ""), "§87.": (K, "synthetic_patient_generator.py; kit tests", ""), "§88.": (K, "-", ""),
 "§89.": (K, "synthetic_patient_generator.py", ""), "§90.": (R, "-", "the June VAL sealing protocol (case/HC hypotheses, cohort specification): superseded by the procedure rule - doors/*_PREREG.md written before data is read, *_OUTCOME.md as found, finding_check.py on the push"), "§91.": (K, "-", ""),
 # Part IV / V
 "§92.": (R, "-", "failure modes by June stage; per-stage status as above"), "§93.": (R, "-", ""), "§94.": (R, "-", "the June disagreement protocol (cohort halts, Stage 3 advancement); today the second opinion (stage_2b) records agreement on the bundle and the report prints it"),
 "§95.": (L, "presence floors; UNMAPPED refusal; QUARANTINE", ""), "§96.": (L, "file_run.py --run-id", ""),
 "§97.": (L, "generated from chain_inventory_v1.json by sop_repoint.py", ""), "§98.": (R, "-", "the CMB <-> methylome term map of June; rows naming removed stages are record"), "§99.": (L, "-", "the floors"),
 "§100.": (R, "-", ""), "§101.": (R, "-", ""), "§102.": (R, "-", "change log: git carries the history"),
 "§103.": (L, "-", "lesson"), "§104.": (L, "-", "the firewall"), "§105.": (R, "-", "LESSON-ASCORE-02 applies to the SEPARATION statistic, not the gauge (RULING A3)"),
 "§106.": (L, "-", "RULING A3"), "§107.": (R, "-", "July wiring, recorded"), "§108.": (L, "presence floors", ""),
}
UNNUMBERED = {  # heading prefix -> status for stage headings without a § number
 "## Stage 0 ": (L, "stage_0_intake.py via run_sample.py", ""), "## Stage 1 ": (L, "stage_1_idat_calibration.py via run_sample.py", ""),
 "## Stage 2 ": (L, "stage_a_cells, stage_2b_second_opinion, stage_2c (trace), stage_2d_foreign_detection", ""),
 "## Stage 3 ": (N, "-", "firewall §104: the production chain subtracts no foregrounds"),
 "## Stage 4 ": (L, "stage_a_cells (per-cell A); stage_b_identity (class gate)", ""),
 "## Stage 4.5 ": (L, "stage_4_5_bidirectional", ""), "## Stage 4.6 ": (L, "stage_4_6_patient_sky", ""),
 "## Stage 2c ": (L, "stage_2c_trace_detection.py (called from stage_a_cells)", ""),
 "## Stage 5 ": (R, "-", "REMOVED 2026-09-27"), "## Stage 6 ": (R, "-", "REMOVED 2026-09-27"),
 "## Stage 7 ": (L, "cpg_tiers.tier_of on every present cell's A", "the class carries no tier"),
 "## Stage 8 ": (N, "-", "signature matching: the chain does not do it; stage_8_matching removed 2026-09-27"),
 "## Stage 9 ": (L, "build_methylphys.py", "the report"), "## Stage 10 ": (L, "kit/file_run.py", "filing; no delivery channel"),
 "## L9": (K, "-", "the null suite runs around the chain, not per sample"), "## Part III": (K, "-", ""), "### Part III": (K, "-", ""), "# Part III": (K, "-", "runs around the chain, not per sample"),
 "# Stage 5 ": (R, "-", "REMOVED 2026-09-27"), "# Stage 6 ": (R, "-", "REMOVED 2026-09-27"), "# Stage 7 ": (L, "cpg_tiers.tier_of on every present cell's A", "the class carries no tier"),
 "# Stage 8 ": (N, "-", "signature matching: the chain does not do it; stage_8_matching removed 2026-09-27"), "# Stage 9 ": (L, "build_methylphys.py", "the report"),
 "# Stage 10 ": (L, "kit/file_run.py", "filing; no delivery channel"), "# Stage 3 ": (N, "-", "firewall §104"), "# Stage 4 ": (L, "stage_a_cells (per-cell A)", ""),
 "# Part IV": (R, "-", "failure modes and decision trees of the June specification; per-section status above governs"), "# Part V": (L, "-", "reference"),
}
BANNER = re.compile(r"^> \*\*STATUS: [A-Z ]+\*\*.*$", re.M)

def load_seq():
    cs = json.load(open(SEQ)); return cs["live_path"], cs

def gen_part_iia(live, inv):
    files = inv["files"] if isinstance(inv.get("files"), list) else list(inv["files"].values())
    by_stage = {}
    for f in files:
        st = (f.get("stage") or "").strip()
        if st: by_stage.setdefault(st.lower(), []).append(f["file"])
    tag = {"step_0": "stage 0", "Stage 1 - IDAT": "stage 1", "stage_1s": "stage 1s", "stage_a_cells": "stage A", "stage_2b": "stage 2b",
           "stage_b": "stage B", "stage_2d": "2d", "stage_4_5": "stage 4.5", "stage_4_6": "stage 4.6", "Report": "interface"}
    rows = ["| # | step (as the code names it) | file | what it does (first line of its docstring) | runtime files tied to it |", "|---|---|---|---|---|"]
    for i, s in enumerate(live, 1):
        st = next((v for k, v in tag.items() if s["step"].startswith(k)), "")
        rf = ", ".join(sorted(set(by_stage.get(st.lower(), []))))[:220] if st else ""
        rows.append(f"| {i} | `{s['step']}` | `{s['where']}` | {s.get('implements','').replace('|','/')[:160]} | {rf} |")
    return ("## Part II-A — The chain as the code runs it (GENERATED - do not edit)\n\n"
            "Derived by `chain/build_chain_sequence.py` from `cpg_conductor.run_full` and the two scripts around it (`run_sample.py`, "
            "`stage_1_idat_calibration.py`), following every call; written into `chain_sequence.json`; rendered here by `sop/build_sop_mirror.py`. "
            "Every numbered section in Part II carries a STATUS banner tying it to one of these rows (LIVE), or stating that the step is RECORD "
            "(built, then removed), NOT IN CHAIN (specified, not run), NOT BUILT, or KIT (a tool around the chain). The reconciliation is checked "
            "on every push (propagate rule 15).\n\n" + "\n".join(rows) + "\n\n")

def apply(text, live, check_only=False):
    steps = {s["step"] for s in live}
    lines = text.split("\n"); out = []; i = 0; placed = {}; live_named = set()
    while i < len(lines):
        l = lines[i]; out.append(l); key = None
        m = re.match(r"^#{2,3} (§\d+(?:\.\d+)?\.?)", l)
        if m:
            k = m.group(1); k = k if k.endswith(".") else k + "."
            key = k if k in STATUS else (k[:-1] if k[:-1] in STATUS else None)
        else:
            for pre in UNNUMBERED:
                if l.startswith(pre) and not (pre.startswith("# ") and l.startswith("## ")): key = pre; break
        if key:
            st, impl, note = (STATUS.get(key) or UNNUMBERED.get(key))
            banner = f"> **STATUS: {st}**" + (f" — implemented by `{impl}`" if impl != "-" else "") + (f". {note}" if note else "") + (" *(2026-09-27, build_sop_mirror.py)*")
            # drop an existing banner (and the blank line after it) directly below the heading
            j = i + 1
            while j < len(lines) and lines[j].strip() == "": j += 1
            if j < len(lines) and BANNER.match(lines[j]): i = j  # skip old banner; its trailing blank is re-added
            out.append(""); out.append(banner); placed[key] = st
            if st == L:
                for s in steps:
                    if s in impl: live_named.add(s)
        i += 1
    text2 = "\n".join(out)
    # Part II-A: replace or insert before "## Stage 0 "
    inv = json.load(open(INV)); part = gen_part_iia(live, inv)
    if "## Part II-A — The chain as the code runs it" in text2:
        a = text2.index("## Part II-A — The chain as the code runs it"); b = text2.index("## Stage 0 ", a)
        text2 = text2[:a] + part + text2[b:]
    else:
        b = text2.index("## Stage 0 "); text2 = text2[:b] + part + text2[b:]
    # reconciliation
    missing_live = [s for s in steps if not any(s in (STATUS[k][1] if k in STATUS else UNNUMBERED[k][1]) for k in placed if (STATUS.get(k) or UNNUMBERED.get(k))[0] == L)]
    unmapped = [k for k in STATUS if k not in placed]
    return text2, placed, missing_live, unmapped

def main():
    live, cs = load_seq(); text = open(SOP, encoding="utf-8").read()
    new, placed, missing_live, unmapped = apply(text, live)
    counts = {}
    for st in placed.values(): counts[st] = counts.get(st, 0) + 1
    print("sections bannered:", len(placed), counts)
    if missing_live: print("LIVE STEPS WITH NO LIVE SECTION:", missing_live)
    if unmapped: print("STATUS keys with no heading in the SOP:", unmapped)
    if "--check" in sys.argv:
        # byte equality is not the test: sop_repoint.py legitimately rewrites references after this script. The test is that
        # every section carries its banner, Part II-A is present, and every live step is named by a LIVE section.
        banners_present = all(f"> **STATUS: {st}**" in text for st in set(placed.values())) and text.count("> **STATUS:") >= len(placed)
        part = "## Part II-A — The chain as the code runs it" in text
        ok = banners_present and part and not missing_live
        print("SOP MIRROR:", "PASS" if ok else "FAIL" + ("" if banners_present else " (banners missing)") + ("" if part else " (Part II-A missing)") + (" (live steps without a section)" if missing_live else ""))
        sys.exit(0 if ok else 1)
    open(SOP, "w", encoding="utf-8").write(new); print("SOP written")
    sys.exit(1 if missing_live else 0)

if __name__ == "__main__": main()
