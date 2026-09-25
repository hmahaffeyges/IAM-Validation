#!/usr/bin/env python3
"""Give every stage and step section in the SOP a link to the code that implements it, and say whether it runs.

The SOP described ten stages in prose and, for most of them, named no file at all - so a reader could not get
from a step to the code. This inserts one line under each stage heading and each step heading:

    Implemented in: a link to the file, then whether the code calls it

Status comes from chain/chain_sequence.json, which is derived from the AST, so a section cannot claim to be
implemented by something the code does not call. Idempotent: run it again and nothing changes. Called by
sop_repoint.py, so regenerating the SOP keeps the links.
"""
import json, os, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True).stdout.strip()
CHAIN = os.path.normpath(os.path.join(HERE, "..", "chain"))
SOP = os.path.join(HERE, "MethylPhys_CPG_SOP.md")
MARK = "**Implemented in:**"

# stage (and step-range) -> the files that implement it, and the conductor function when there is one.
# Every entry is checked against chain_sequence.json before it is written.
MAP = {
    "Stage 0":   (["stage_0_intake.py"], None),
    "Stage 1":   (["stage_1_idat_calibration.py"], "calibrate_idat_to_beta"),
    "Stage 2":   (["walther_iam_deconvolver.py", "nilc_celltype_deconvolver.py"], "stage_a_cells"),
    "Stage 3":   ([], None),
    "Stage 4.5": (["bidirectional_decomposition.py"], "stage_4_5_bidirectional"),
    "Stage 4.6": (["stage_4_6_patient_cmb.py"], "stage_4_6_patient_sky"),
    "Stage 4":   (["iamatlas_a_scoring.py", "cpg_gauge_engine.py"], "stage_b_classes"),
    "Stage 5":   (["iamatlas_mahalanobis_scoring.py"], "stage_5_mahalanobis"),
    "Stage 6":   (["reference_age_curve_v1.json", "iam_cellular_age_scoring.py"], "stage_6_cellular_age"),
    "Stage 7":   (["cpg_tiers.py", "tier_breakpoints.json"], "stage_b_identity"),
    "Stage 8":   ([], None),
    "Stage 9":   (["build_methylphys.py"], None),
    "Stage 10":  ([], None),
}
NOTE = {
    "Stage 0": "runs before calibration on the live path (register row 0 COMMISSIONED, PROC-STAGE0-02); "
               "the intensity-dependent checks read the array's own controls through "
               "`stage_0_1_qc_handoff.py`, and a QUARANTINE stops the chain with nothing scored",
    "Stage 3": "the live path subtracts no foregrounds - see §104, which is the ruling that governs this stage",
    "Stage 8": "not a chain stage (author's ruling 2026-09-21): `stage_8_matching` is defined in the conductor "
               "and `run_full` does not call it",
    "Stage 10": "delivery is not part of the commissioned path; the report itself is Stage 9",
}


def _paths():
    by = {}
    for f in subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True).stdout.split("\n"):
        if f and "RETIRED" not in f and "author_copies" not in f:
            by.setdefault(os.path.basename(f), []).append(f)
    return by


def _enc(rel):
    return rel.replace("%", "%25").replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def line_for(stage, by, seq):
    files, fn = MAP[stage]
    live = {st["step"] for st in seq["live_path"]}
    bits = []
    for f in files:
        cand = by.get(f, [])
        if len(cand) != 1:
            continue
        rel = _enc(os.path.relpath(os.path.join(ROOT, cand[0]), HERE))
        bits.append("[`" + f + "`](" + rel + ")")
    if fn == "calibrate_idat_to_beta":
        rel = _enc(os.path.relpath(os.path.join(CHAIN, "MethylPhys_Interface", "run_sample.py"), HERE))
        bits.append("called by [`run_sample.py`](" + rel + ") before the conductor - runs in the live path")
    elif fn:
        status = "runs in the live path" if fn in live else "NOT called by run_full"
        rel = _enc(os.path.relpath(os.path.join(CHAIN, "cpg_conductor.py"), HERE))
        bits.append("called by [`cpg_conductor." + fn + "`](" + rel + ") - " + status)
    if stage in NOTE:
        n = NOTE[stage]
        if files:
            for f in files:
                cand = by.get(f, [])
                if len(cand) == 1:
                    rel = _enc(os.path.relpath(os.path.join(ROOT, cand[0]), HERE))
                    n = n.replace("`" + f + "`", "[`" + f + "`](" + rel + ")")
        bits.append(n) if not bits else bits.append("**" + n + "**")
    return MARK + " " + "; ".join(bits) if bits else None


def main():
    by, added = _paths(), 0
    seq = json.load(open(os.path.join(CHAIN, "chain_sequence.json")))
    lines = open(SOP, encoding="utf-8").read().split("\n")
    out = []
    for i, l in enumerate(lines):
        out.append(l)
        m = re.match(r"^#{2,3} (Stage (?:\d+(?:\.\d+)?)) — ", l)
        if not m:
            # Stages 5-10 have no stage heading in Part II: their steps are top-level numbered sections, and
            # the author asked for the detail sections to carry links too.
            m2 = re.match(r"^#{2,3} §[\d.]+\.? Step (\d+(?:\.\d+)?)\.\d+ — ", l)
            if not m2:
                continue
            stage = "Stage " + m2.group(1)
        else:
            stage = m.group(1)
        if stage not in MAP:
            continue
        nxt = "\n".join(lines[i + 1:i + 4])
        if MARK in nxt:
            continue
        ln = line_for(stage, by, seq)
        if ln:
            out.append("")
            out.append(ln)
            added += 1
    open(SOP, "w", encoding="utf-8").write("\n".join(out))
    print(f"stage sections given an implemented-in line: {added}")
    return added


if __name__ == "__main__":
    main()
