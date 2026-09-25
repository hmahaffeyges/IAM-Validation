#!/usr/bin/env python3
"""sop_repoint.py - bring the SOP's file references onto the files that exist, and generate its reference table.

The SOP is the procedure the chain implements. An audit on 2026-09-22 found that of the 108 file names it cited,
55 were live, 30 existed only under RETIRED/, and 23 existed nowhere; and 8 of the 18 files the chain actually
resolves were not named in it at all - including Stage 1 calibration and the sky stage.

This script is the durable fix rather than a one-off edit:
  1. the change history at the top (six changelog blocks and a 167-line supersession ledger) is replaced by ONE
     statement of what the current version is. Git carries the history; a procedure states what to do now.
  2. every dead file reference is repointed by an explicit map, each entry carrying WHY:
       RENAME       - the same thing under its current name
       RECORD_SIDE  - the file exists and is real, but run_full() does not call it
       NOT_IN_CHAIN - the step it belonged to is not part of the chain
       DROP         - the name never named a file in this repository
  3. the reference section is GENERATED from chain_inventory_v1.json, so it cannot drift again. That inventory is
     itself generated from the tree.
Run it after any file move; it is idempotent.
"""
import os, re, json, sys, hashlib, time

HERE=os.path.dirname(os.path.abspath(__file__)); PM=os.path.dirname(HERE); BIO=os.path.dirname(PM)
SOP=os.path.join(HERE,"MethylPhys_CPG_SOP.md")
INV=os.path.join(BIO,"MethylPhys/chain","Runtime Matrices","chain_inventory_v1.json")

# name -> (kind, replacement_or_None, note)
MAP={
 # --- the same thing under its current name ---
 "iamatlas_a_score_loci_v1_0.json":("RENAME","iamatlas_gauge_identity_loci_v1_0.json","the identity-loci gauge file"),
 "nilc_deconvolver.py":("RENAME","nilc_celltype_deconvolver.py","restored into the engine as the second opinion"),
 "patient_brightness_comparison.py":("RENAME","stage_4_6_patient_cmb.py","the sky stage; it weights by the sample's own composition, which a precomputed brightness file cannot"),
 "tier_breakpoints_v0_4tier_statistical.json":("RENAME","tier_breakpoints.json","the single tier definition, read by cpg_tiers.py"),
 "mahalanobis_healthy_reference_v0_3.json":("RENAME","mahalanobis_healthy_reference_v2_0_age_matched_derived.json","the departure reference in the chain"),
 "mahalanobis_healthy_reference_v0_5.json":("RENAME","mahalanobis_healthy_reference_v2_0_age_matched_derived.json","the departure reference in the chain"),
 "SYSTEM_INVENTORY.md":("RENAME","chain_inventory_v1.json","generated from the tree by build_chain_inventory.py"),
 "walther_patient_report_builder.py":("RENAME","cpg_report_v3.py","the report the chain renders"),
 "walther_report_builder.py":("RENAME","cpg_report_v3.py","the report the chain renders"),
 "provenance.json":("RENAME","IAMAtlasREBUILD_provenance.json","the atlas build record"),
 "1_README.md":("RENAME","README.md",None),
 "per_sample.json":("RENAME","per_sample.csv",None),
 "age_reference_matrix.py":("RENAME","reference_age_curve_v1.json","the age term the chain subtracts, measured leave-one-laboratory-out"),
 "IAMAtlas_age_layer.csv":("RENAME","reference_age_curve_v1.json","age enters as a decade term subtracted before placement, not as a foreground layer"),
 "_stage_4_5_bidirectional_decomposition.json":("RENAME","directional_panels_v1_0.json","the panel bidirectional_decomposition.py reads"),
 "N1_permutation_distribution.json":("RENAME","cpg_null_runner.py","the null suite that produces the permutation distributions"),
 "N2_within_decade_distribution.json":("RENAME","cpg_null_runner.py","the null suite that produces the within-decade distributions"),
 # --- real files, but the chain does not call them ---
 "disease_matching_BUILD_SPEC_v1_3.md":("RECORD_SIDE","disease_matching.py (v1 conductor retired 2026-09-25)","the pre-conductor monolith; kept for provenance"),
 "nilc_fractions_all.csv":("RECORD_SIDE",None,"output of an earlier deconvolver run"),
 "nilc_fractions_v2_departure.csv":("RECORD_SIDE",None,"output of an earlier deconvolver run"),
 "nilc_walther_crosscheck.json":("RECORD_SIDE",None,"superseded by the second-opinion comparison inside run_full"),
 "nilc_walther_crosscheck_v2.json":("RECORD_SIDE",None,"superseded by the second-opinion comparison inside run_full"),
 "mahalanobis_per_patient_breast_predx_validation.csv":("RECORD_SIDE",None,"a validation output, not an input"),
 "cellular_ages_v4_epic_italy_validation.csv":("RECORD_SIDE",None,"a validation output of the retired cellular-age layer"),
 "Phase_B2_FINDING.md":("RECORD_SIDE",None,"a pre-build finding, kept in the record"),
 "Phase_B2_1_FINDING.md":("RECORD_SIDE",None,"a pre-build finding, kept in the record"),
 "Phase_B3_FINDING.md":("RECORD_SIDE",None,"a pre-build finding, kept in the record"),
 "headline_results.json":("RECORD_SIDE",None,"a results summary, not a chain input"),
 # --- the step is not in the chain ---
 "IAMAtlas_sex_layer.csv":("NOT_IN_CHAIN",None,"no foreground layer is subtracted; sex is recorded, not adjusted"),
 "IAMAtlas_smoking_layer.csv":("NOT_IN_CHAIN",None,"no foreground layer is subtracted; smoking is measured as a null on this gauge"),
 "age_axis_foreground.py":("NOT_IN_CHAIN",None,"superseded by the measured age curve"),
 "sex_axis_foreground.py":("NOT_IN_CHAIN",None,"no foreground subtraction"),
 "smoking_axis_foreground.py":("NOT_IN_CHAIN",None,"no foreground subtraction"),
 "age_layer_diagnostics.json":("NOT_IN_CHAIN",None,"diagnostics of the retired foreground layer"),
 "foreground_registry.py":("NOT_IN_CHAIN",None,"no foreground registry exists; the decision was not to subtract"),
 "cancer_prior.json":("NOT_IN_CHAIN",None,"the chain applies no disease prior"),
 "family_history_multiplier.json":("NOT_IN_CHAIN",None,"the chain applies no risk multiplier"),
 "cfdna_weight.json":("NOT_IN_CHAIN",None,"a placeholder that was never derived"),
 "disease_cell_signature_matrix_v1_5.csv":("NOT_IN_CHAIN","disease_cell_signature_matrix_v1_13.csv","disease matching is not part of the chain; the matrix is record-side"),
 "disease_signature_matrix_README.md":("NOT_IN_CHAIN",None,"disease matching is not part of the chain"),
 "breast-epic_card_v2_3.json":("NOT_IN_CHAIN","breast-epic_card_v3_1.json","the cards are record-side"),
 "breast_epic_residual_map_v0_1.csv":("NOT_IN_CHAIN","breast_epic_residual_map_chr_annotated.csv","the maps are record-side"),
 "README_immune_atlas_residual_maps.md":("NOT_IN_CHAIN",None,"the residual maps are record-side"),
 "GAPE_WEB_v13.py":("NOT_IN_CHAIN",None,"a report generator from the preliminary era"),
 "CPG_Chain_of_Custody_SOP_v1_PART_I.md":("NOT_IN_CHAIN",None,"this document is the current SOP; earlier parts are in git history"),
 # --- never named a file here ---
 "dye_bias.py":("DROP",None,None), "sex_check.py":("DROP",None,None), "null_invocation_log.json":("DROP",None,None),
 "web.commercial.py":("DROP",None,None), "v1_CPG_Recipe.md":("DROP",None,None), "tar.xz":("DROP",None,None),
 "_brightness.csv":("DROP",None,None), "prereg.json":("DROP",None,None),
}

CURRENT_STATE = """## What this version is

This is the operating procedure for the chain as it stands at the commit named below. It describes what the chain
does now. It is not a history of what it used to do: every earlier version of this document is in the repository's
git history, and every finding that changed the chain is in `Record/PROC_data/` with its pre-registration,
its outcome and its seal.

| | |
|---|---|
| engine commit | `{commit}` |
| stages the conductor runs | Stage 0 intake, Stage 1 calibration, Stage 2 composition (with a second opinion), Stage B identity gauge, Stage 4.5 directional decomposition, Stage 4.6 the patient's sky, Stage 5 departure, Stage 7 tiers, Stage 9 the report |
| what it reports | whether this sample's cellular write process is operating within the healthy range for its age, by architecture class, against a fixed physical zero |
| the healthy reference | three measured layers: the frozen class floor (physics, universal), the pipeline scale map (one per processing pipeline), and the laboratory zero (40 healthy arrays of that laboratory, read against the age curve) |
| what it does not do | it names no condition, matches no pattern to any signature, and states no age in years |
| file inventory | generated, not hand-listed: `MethylPhys/chain/Runtime Matrices/chain_inventory_v1.json`, built from the tree by `build_chain_inventory.py`. The reference table at the end of this document is generated from it |
| the rule for a finding | seal the procedure before running it, register the outcome as found, close it in code, teach every door, rebuild, read, push. `MethylPhys/kit/finding_check.py` gates the push |

"""

def commit():
    try:
        import subprocess
        return subprocess.run(["git","-C",os.path.dirname(BIO),"rev-parse","--short","HEAD"],
                              capture_output=True,text=True).stdout.strip() or "unknown"
    except Exception: return "unknown"

HEADER_FIXES = [
    # The title pinned an engine commit (66f37fe, July). A procedure that names a commit is stale the moment the
    # engine moves; the same defect was removed from the manual's page one. State the version, point at git.
    (re.compile(r"^# CPG Chain-of-Custody Standard Operating Procedure \\(SOP\\) — v2\\.0\\.0[^\\n]*$", re.M),
     "# MethylPhys CPG SOP\n\n## Standard Operating Procedure: Cellular Performance Gauge Chain of Custody"),
    (re.compile(r"^\\*\\*Document version:\\*\\* v2\\.0\\.0[^\\n]*$", re.M),
     "**Document version:** v2.0.0, matched to the engine in this repository at the commit this file was last "
     "regenerated from (see `git log -1 -- Biological_Physics/MethylPhys/chain`). Earlier versions are in git "
     "history; this document states the current procedure."),
]

HEADER_TITLE = "# CPG Chain-of-Custody Standard Operating Procedure (SOP) \u2014 v2.0.0"
HEADER_VERSION = ("**Document version:** v2.0.0, matched to the engine in this repository at the commit this file "
                  "was last regenerated from (`git log -1 -- Biological_Physics/MethylPhys/chain`). Earlier versions "
                  "are in git history; this document states the current procedure.")

PATH_FIXES = [
    # 2026-09-25: the v1 conductor and its batch driver were retired; the SOP still cites them
    # as history, so the citation points at where they actually are.
    ("../chain/walther_clinical.py", "../../RETIRED_2026-09/v1_conductor_2026-09/walther_clinical.py"),
    ("../chain/run_batch.py", "../../RETIRED_2026-09/v1_conductor_2026-09/dependents/run_batch.py"),('Runtime Matrices/Bidirectional_Decomposition/bidirectional_decomposition.py', 'Runtime Matrices/Directional Panel/bidirectional_decomposition.py'), ('Biological_Physics/RETIRED_2026-09/PostBuild_atlas_vault_snapshot_2026-06/Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py', 'Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py')] + [('pipeline_runtime_matrices/iamatlas_a_scoring.py', 'Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py'), ('Biological_Physics/RETIRED_2026-09/PostBuild_atlas_vault_snapshot_2026-06/chain_inventory_v1.json', 'Runtime Matrices/chain_inventory_v1.json'), ('Runtime Matrices/Brightness_Comparison/stage_4_6_patient_cmb.py', 'stage_4_6_patient_cmb.py'), ('Biological_Physics/chain_of_custody/L9_null_suite/synthetic_patient_generator.py', 'Synthetic_Patient_Generator/synthetic_patient_generator.py')]   # directory-qualified references that named an old or wrong location


def _repoint_stale_dirs(text):
    """Directory-qualified references written against the June/July layout.

    Added 2026-09-22 after link_check.py found eight of them. The rule is general and idempotent rather than a
    list of special cases: for a reference of the form `dir/.../name.ext` whose path does not resolve, if exactly
    one tracked file in the live tree has that basename, repoint to it; if none or several do, leave the text and
    mark it as a historical path so a reader does not try to follow it.
    """
    import subprocess, collections
    root = os.path.dirname(BIO)   # the repository root: Biological_Physics' parent
    tracked = [f for f in subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True).stdout.split("\n")
               if f and "RETIRED" not in f]
    by_base = collections.defaultdict(list)
    for f in tracked:
        by_base[f.split("/")[-1]].append(f)
    pat = re.compile(r"(?<![\w/])([A-Za-z0-9_][A-Za-z0-9_.\- ]*(?:/[A-Za-z0-9_.\- ]+)+\.(?:py|json|csv|md|npz|npy|tsv))")
    fixed = 0
    marked = 0
    out = []
    pos = 0
    for m in pat.finditer(text):
        ref = m.group(1)
        if os.path.exists(os.path.join(root, ref)) or os.path.exists(os.path.join(root, "Biological_Physics", ref)):
            continue
        base = ref.split("/")[-1]
        cands = by_base.get(base, [])
        out.append((m.start(), m.end(), ref, cands))
    for start, end, ref, cands in reversed(out):
        if len(cands) == 1:
            new = cands[0].replace("Biological_Physics/", "")
            text = text[:start] + new + text[end:]
            fixed += 1
        elif " (historical path" not in text[end:end + 20]:
            text = text[:end] + " (historical path)" + text[end:]
            marked += 1
    print("  stale dirs: %d repointed, %d marked historical" % (fixed, marked))
    return text


def _report_tab_section(text):
    """The report, tab by tab, generated from manual/report_tabs.json (author 2026-09-25: "it should have
    every tab described and explained and the CMB pass/fails etc ... its literally the operating manual").

    Generated rather than written: a hand-written tab list went stale twice in September. The source is the
    JSON that kit/build_report_tab_reference.py writes from a real report, so this section cannot describe a
    tab the report does not have, or miss one it does.
    """
    import json as _json
    jp = os.path.join(BIO, "MethylPhys", "manual", "report_tabs.json")
    if not os.path.exists(jp):
        return text
    d = _json.load(open(jp, encoding="utf-8"))
    m, tabs, reg = d["_meta"], d["tabs"], d["cmb_registry"]
    L = ["", "## The report this chain produces, tab by tab", "",
         "_Generated from `manual/report_tabs.json` by `kit/build_report_tab_reference.py`, read off a real "
         "report (`%s`, commit `%s`). Re-run it after any change to the report builder._" %
         (m["generated_from"], m["commit"]), "",
         "One run writes **one self-contained HTML file of %s MB with %d tabs** - %d carry this specimen's "
         "own measurements and %d carry reference material identical in every report. A reference tab tells "
         "you how the instrument works; only a specimen tab tells you anything about the patient." %
         (m["report_mb"], m["n_tabs"], m["n_specimen"], m["n_reference"]), "",
         "| tab | kind | what it carries | size |", "|---|---|---|---|"]
    for t in tabs:
        L.append("| **%s** (`%s`) | %s | %s | %d KB, %d tables |" %
                 (t["label"], t["tab"], t["kind"], t["purpose"].replace("|", "/"), t["kb"], t["tables"]))
    L += ["", "Figures of every tab, with the sections each one contains, are in "
              "[`REPORT_TAB_REFERENCE.md`](../doors/REPORT_TAB_REFERENCE.md).", ""]
    if reg:
        npass = sum(1 for c in reg if c["state"] == "PASS")
        nb = sum(1 for c in reg if c["state"] == "NOT_BUILT")
        L += ["### The CMB tool register, and what a FAIL does", "",
              "The Safeguards tab carries every method borrowed from CMB analysis with a check that runs on "
              "the finished bundle: **%d methods, %d PASS on the commissioning specimen, %d NOT_BUILT**. A "
              "FAIL is also emitted to the Red flags tab as `CMB_TOOL_FAIL`, so a borrowed method that "
              "stopped working cannot be missed in the middle of a long tab. NOT_BUILT entries are listed on "
              "purpose - the shelf is part of the record." % (len(reg), npass, nb), "",
              "| state | method |", "|---|---|"]
        for c in sorted(reg, key=lambda c: (c["state"] != "FAIL", c["state"], c["tool"])):
            L.append("| `%s` | %s |" % (c["state"], c["tool"]))
        L += ["", "**To add a tool:** append one entry to `TOOLS` in `chain/cmb_tools.py` with a "
                  "`check(bundle) -> (status, evidence)`. The table, the counts and the red-flag routing all "
                  "follow from it; nothing else needs editing.", ""]
    L += ["### The three gates that keep this document true", "",
          "| gate | what it refuses |", "|---|---|",
          "| `chain/propagate.py` | regenerates every derived document, then checks the rules a human wrote "
          "(every live module named here and in the reviewer manifest, every sealed procedure in the "
          "commissioning table, every relative reference resolving). Exits non-zero on drift. |",
          "| `kit/link_check.py` | every relative path in the live documentation set must resolve - a path in "
          "a document is a claim like any other. |",
          "| `chain/guarded_push.sh` | runs `propagate.py` **without a pipe** and refuses to commit or push "
          "if it fails. A pipe hands the shell the pipe's exit status, not the gate's, which is how a push "
          "once proceeded over a printed failure. |", "",
          "Every report prints the first gate's verdict on its own Run tab, so a reading whose documents had "
          "drifted says so on the page.", ""]
    marker = "\n## The report this chain produces, tab by tab\n"
    if marker in text:
        i = text.index(marker)
        j = text.find("\n## ", i + len(marker))
        text = text[:i] + "\n".join(L) + (text[j:] if j > 0 else "")
    else:
        text = text.rstrip() + "\n" + "\n".join(L)
    print("  report tab section: %d tabs, %d CMB tools" % (len(tabs), len(reg)))
    return text

def _fix_header(text):
    """The title used to pin an engine commit (66f37fe, July). A procedure that names a commit is stale the
    moment the engine moves - the same defect was removed from the manual's page one. State the version and
    point at git instead."""
    lines = text.split("\n")
    if lines and lines[0].startswith("# CPG Chain-of-Custody"): lines[0] = HEADER_TITLE
    for i in range(1, min(10, len(lines))):
        if lines[i].startswith("**Document version:**"): lines[i] = HEADER_VERSION; break
    return "\n".join(lines)

def main():
    s=open(SOP,encoding="utf-8").read(); orig=s
    # 1a. the header: one line for what this is, not five for what it replaced
    s=re.sub(r"^\*\*(Previous|Supersedes):\*\*[^\n]*\n","",s,flags=re.M)
    s=re.sub(r"^\*\*Document version:\*\*[^\n]*\n",
             "**Document version:** v2.0.0. Earlier versions are in git history; this document states the current procedure.\n",s,flags=re.M)
    # 1b. one current-state block in place of the changelogs and the supersession ledger
    i=s.find("## Changelog"); j=s.find("## How to use this document")
    if i>0 and j>i:
        s=s[:i]+CURRENT_STATE.format(commit=commit())+s[j:]
    k=s.find("## SUPERSESSION LEDGER")
    if k>0:
        e=s.find("\n# Part I", k)
        if e>k: s=s[:k]+s[e+1:]
    # 2. repoint
    counts={}
    # Substitute on WORD BOUNDARIES. A plain str.replace turned "provenance.json" into
    # "IAMAtlasREBUILD_IAMAtlasREBUILD_provenance.json" because the short name is a substring of the long one.
    # The lookbehind refuses a match that is already part of a longer file name, and the annotation branch is
    # idempotent so re-running the script does not nest annotations.
    for name,(kind,repl,note) in sorted(MAP.items(), key=lambda kv:-len(kv[0])):
        # exclude only word characters and hyphen before the name: a path prefix like "runtime/" or a directory
        # component ending in "/" is exactly where these names appear, and excluding "/" and "." refused them all.
        pat=re.compile(r"(?<![\w-])"+re.escape(name)+r"(?![\w-])")
        n=len(pat.findall(s))
        if not n: continue
        counts[name]=(kind,n)
        if kind=="RENAME" and repl:
            s=pat.sub(repl.replace("\\","\\\\"),s)
        else:
            tag={"RECORD_SIDE":"not called by run_full","NOT_IN_CHAIN":"not part of the chain","DROP":"no such file in this repository"}[kind]
            ann=f"{name} ({tag}"+(f"; use {repl}" if repl else "")+(f" - {note}" if note else "")+")"
            if f"{name} ({tag}" in s: continue
            s=pat.sub(ann.replace("\\","\\\\"),s)
    # 3. generated reference table
    inv=json.load(open(INV)); rows=[r for r in inv["files"] if r["role"] in ("chain","reference","interface")]
    tbl=["\n\n# Reference - the files of the chain\n",
         f"_Generated from `chain_inventory_v1.json` (built {inv['_meta']['built']}, commit `{inv['_meta']['commit']}`) "
         "by `sop_repoint.py`. Do not hand-edit: re-run the script after any file move._\n",
         "\n| file | role | stage | what it is |\n|---|---|---|---|\n"]
    for r in sorted(rows,key=lambda r:(r["role"]!="chain",str(r["stage"]),r["file"])):
        tbl.append(f"| `{r['file']}` | {r['role']} | {r['stage']} | {r['description'][:190]} |\n")
    m=s.find("\n# Reference - the files of the chain")
    if m>0: s=s[:m]
    s=s.rstrip()+"".join(tbl)
    for pat,rep in HEADER_FIXES:
        s=pat.sub(rep,s,count=1)
    for _a,_b in PATH_FIXES: s=s.replace(_a,_b)
    s=_report_tab_section(s)
    s=_repoint_stale_dirs(s)
    s=_fix_header(s)
    open(SOP,"w",encoding="utf-8").write(s)
    print(f"SOP: {orig.count(chr(10))} -> {s.count(chr(10))} lines")
    for k2 in ("RENAME","RECORD_SIDE","NOT_IN_CHAIN","DROP"):
        got=[(n,c) for n,(kk,c) in counts.items() if kk==k2]
        print(f"  {k2:<13} {len(got):>2} names, {sum(c for _,c in got):>3} occurrences")
    # 2026-09-25: a runtime file that nothing reads must still be NAMED, with its status, or a reader
    # who finds it assumes it is live. propagate.py fails if one is missing from this table.
    import glob as _glob, os as _os
    _known = " ".join(r if isinstance(r, str) else " ".join(map(str, r)) for r in rows)
    _extra = []
    for _f in sorted(_glob.glob(_os.path.join(BIO, "MethylPhys", "chain", "Runtime Matrices", "**", "*.json"), recursive=True)):
        _b = _os.path.basename(_f)
        if _b not in _known and _b not in s:
            _extra.append(_b)
    if _extra:
        s += ("\n\n## Runtime files present but NOT READ BY THE CHAIN\n\n"
              "Generated. Named so a reader who finds them knows their status rather than assuming\n"
              "they are live; nothing in the live path resolves them.\n\n")
        for _b in _extra:
            _why = ("a TRIAL panel: the adoption and anchor re-seal it needs have not been done, so the "
                    "chain still reads v0_2" if "TRIAL" in _b else
                    "superseded by identity_band_v3.json, the commissioned band; kept because sealed "
                    "records cite it" if "PROVISIONAL" in _b else "present in the tree, read by nothing")
            s += "- `%s` - %s\n" % (_b, _why)
        open(SOP, "w", encoding="utf-8").write(s)
        print("  runtime files named but not read: %d" % len(_extra))
    print(f"  reference table: {len(rows)} files")
if __name__=="__main__": main()
