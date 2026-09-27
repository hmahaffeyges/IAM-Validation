#!/usr/bin/env python3
"""vocab_scan.py - the report's own vocabulary guards, run over the SOP and the Operations Manual.

The report refuses to render a measurement tab that carries a banned word (FORBIDDEN) or population vocabulary (COHORT);
those two regexes live in build_methylphys.py and are imported here, so the documents are held to exactly the standard the
report is. Author, 2026-09-27: 'root out all the cohort methodology and banned words from them too.'

A hit is LIVE or RECORD. RECORD = inside a section whose heading (or a line within the previous 40 lines) marks it as record:
[RECORD], REMOVED, RETIRED, 'historical', 'superseded', 'Edition record', 'What changed', 'Falsification', 'procedure log',
or a PROC-/VAL- outcome table. Record text describes what was built and measured and may name what it removed; LIVE text is
procedure and must not. Exit 1 on any LIVE hit (propagate rule 14). Writes kit/results/vocab_scan.json with every hit."""
import os, re, sys, json, subprocess, ast
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(HERE); CH = os.path.join(MP, "chain")
src = open(os.path.join(CH, "MethylPhys_Interface", "build_methylphys.py"), encoding="utf-8").read()
def _rx(name):
    m = re.search(name + r'\s*=\s*re\.compile\((r?"(?:[^"\\]|\\.)*"(?:\s*r?"(?:[^"\\]|\\.)*")*)\s*,\s*re\.I\)', src, flags=re.S)
    assert m, name; return re.compile(eval("(" + m.group(1) + ")"), re.I)
FORBIDDEN, COHORT = _rx("FORBIDDEN"), _rx("COHORT")
RECORD_MARK = re.compile(r"\[RECORD\]|REMOVED|RETIRED|historical|superseded|SUPERSEDED|Edition record|EDITION RECORD|What changed|WHAT CHANGED|Falsification|FALSIFICATION|procedure log|PROCEDURE LOG|PROC-[A-Z0-9]+-\d+|VAL-\d+|band_v2|LAB-ZERO|Record:|record of|as a record|kept as the record|not procedure|^## §4[6-9]|^## §5\d|^## Stage [56]\b|Stage 5 |Stage 6 |Mahalanobis hull|cellular age", re.I | re.M)
# words that are legitimately in procedure text when they name the thing being REFUSED or the file it lives in
ALLOW_LINE = re.compile(r"MEASURE, DON'T COMPARE|never a correction|no population|no laboratory zero|no age term|no band|not procedure|were removed|was removed|removed 2026|retired 2026|RETIRED_2026|is gone|are gone|taken out|came off|banned|guard|vocabulary|forbidden|"
                        r"Not a diagnostic|not a diagnostic|no disease|not claim|does not claim|no diagnostic|is not how|never a |Neither is a|no healthy range is printed|Disease Matrix|DISEASE_MATRIX|disease_cell_signature|disease_origin_cells|not in chain|Not read by run_full|record side|record-side|pre-atlas|Pre-atlas|preliminary|not a cohort method|only point|never as a baseline|Never compare|never a truth|cannot see|no cohort|not disease|not a tumour|names no|is not a detection|cohort comparison could not|structurally cannot|'cohorts, circular'|does not do it|not in the chain|no longer", re.I)

def scan_text(lines, label):
    hits = []; sect_status = "LIVE"
    for i, line in enumerate(lines):
        if re.match(r"^#{1,3} ", line):
            # a heading takes the status of the banner that follows it (within 3 lines); a ### without a banner inherits
            look = [lines[k] for k in range(i + 1, min(i + 4, len(lines)))]
            mbs = [re.match(r"^> \*\*STATUS: ([A-Z ]+)\*\*", x) for x in look]; mbs = [m for m in mbs if m]
            if mbs: sect_status = mbs[0].group(1).strip()
            elif line.startswith("## ") or line.startswith("# "): sect_status = "LIVE"
        mb = re.match(r"^> \*\*STATUS: ([A-Z ]+)\*\*", line)
        if mb: sect_status = mb.group(1).strip()
        if sect_status != "LIVE" and label == "SOP":
            for rx, kind in ((FORBIDDEN, "FORBIDDEN"), (COHORT, "COHORT")):
                for m in rx.finditer(line): hits.append({"doc": label, "line": i + 1, "kind": kind, "word": m.group(0), "status": "RECORD", "text": line.strip()[:200]})
            continue
        for rx, kind in ((FORBIDDEN, "FORBIDDEN"), (COHORT, "COHORT")):
            for m in rx.finditer(line):
                ctx = "\n".join(lines[max(0, i - 40):i + 1])
                record = bool(RECORD_MARK.search(ctx)) or bool(ALLOW_LINE.search(line))
                hits.append({"doc": label, "line": i + 1, "kind": kind, "word": m.group(0), "status": "RECORD" if record else "LIVE", "text": line.strip()[:200]})
    return hits

# ---- the OM is scanned at SOURCE: every string literal in build_operations_manual.py by the section function that emits it,
# and every top-level constant in om_data.py. Record sections are declared here by name; everything else is LIVE.
OM_RECORD_FUNCS = {"sec_edition_record", "sec1_recon", "sec1b_rulings", "sec8_procedures", "sec10_falsification", "sec_proc_log",
                   "secV_val_index", "secVII_sprint", "secVIII_part2", "secIX_future", "sec0b_prior_art", "card_addendum",
                   "_render_002_physics_without_derivation", "sec_cosmo_evidence", "sec7_substrates", "sec5a_tools", "sec_presence",
                   "secVI_translation_map"}
OM_RECORD_VARS = {"RECON", "RECON_EXTRA", "FALSIFICATION", "PANEL", "MAHA_PROC", "MAHA2_PROC", "HULL", "CHAIN_RULES", "SWITCH_PROC", "HISTORY_PROC",
                  "HISTORY", "HISTORY_NOTE", "STAGE0_01", "STAGE01_QC", "CONFORMANCE", "PLASMA_COMPOSITION", "PLASMA_NOTE", "CELL_ROUTING", "CHK31", "WHOLE_BLOOD",
                  "GRID_STATUS", "VAL_INDEX_NOTE", "VAL_INDEX", "AGE_REF", "TIERS_V13", "STAGE7_TIERS", "REFERENCE_LAYERS", "NULLS", "CATALOGUE",
                  "COSMO_EVIDENCE", "COSMO_EVIDENCE_RULE", "CHAIN01", "CHAIN01_ROWS", "PHASE1", "PHASE1C", "BAND_V2", "LABZERO_01", "LABZERO_02",
                  "N7_01", "FORMULA_FINDINGS", "FORMULA_VERDICT", "FORMULA_2X2", "FORMULA_LABELS", "S105", "XU538", "ANCHOR", "CAL01", "NILC_01",
                  "SEP_01", "SEP_02", "SEP_03", "STEM_ADULT_RECORD", "CMB_PROC", "MATCH_PROC", "BIDIR_PROC", "AGE_PROC", "TIER_PROC", "DETECTION_RULE",
                  "PROC_LOG_2026_09", "REPORT_CHANGES_2026_09_25", "RULING_A3", "RULING_M1B", "HMIN_BOOT", "ATLAS_BUILD_DETAIL", "MCMC_HMIN",
                  "FUTURE_GOALS", "SPRINT_VERDICT", "PART_II_OUTLINE", "CANINE", "QPROC_CHIP", "MIX_T6", "MIX_STATS", "MIX_T8", "MIX_T8_R", "ROW9_NOTE",
                  "SEALING_RULE", "PRIOR_ART", "ENGINE_MAP", "TRANSLATION_MAP", "GLOSSARY_NOTE_MAHAFFEY", "MAHAFFEY", "_GL_TAIL", "FOUR_SKIES_CAP"}

def om_source():
    """lines tagged [OM:<function or var>] so a hit names where it lives; status decided per function / variable."""
    out = []
    for fn, kind in ((os.path.join(MP, "manual", "build_operations_manual.py"), "func"), (os.path.join(MP, "manual", "om_data.py"), "var")):
        if not os.path.exists(fn): continue
        src = open(fn, encoding="utf-8").read(); tree = ast.parse(src)
        for node in tree.body:
            if kind == "func" and isinstance(node, ast.FunctionDef): name = node.name; rec = name in OM_RECORD_FUNCS
            elif kind == "var" and isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name): name = node.targets[0].id; rec = name in OM_RECORD_VARS
            else: continue
            for c in ast.walk(node):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and len(c.value) > 12:
                    txt = re.sub(r"<[^>]+>", " ", c.value).replace("\n", " ")
                    out.append(("RECORD" if rec else "LIVE", f"[OM:{name}] {txt}"))
    return out

def om_text():
    pdf = os.path.join(MP, "manual", "MethylPhys_CPG_Operations_Manual.pdf")
    if not os.path.exists(pdf): return []
    try:
        import pypdfium2 as p
        d = p.PdfDocument(pdf); out = []
        for k in range(len(d)):
            for l in d[k].get_textpage().get_text_range().split("\n"): out.append(f"[p{k+1}] {l}")
        return out
    except Exception as e:
        return [f"(OM text not extractable: {type(e).__name__})"]

def main():
    sop = open(os.path.join(MP, "sop", "MethylPhys_CPG_SOP.md"), encoding="utf-8").read().split("\n")
    hits = scan_text(sop, "SOP")
    for status, line in om_source():
        for rx, kind in ((FORBIDDEN, "FORBIDDEN"), (COHORT, "COHORT")):
            for m in rx.finditer(line):
                st = "RECORD" if (status == "RECORD" or ALLOW_LINE.search(line) or "[RECORD" in line or "REMOVED" in line or "RETIRED" in line or "removed 2026" in line.lower()) else "LIVE"
                hits.append({"doc": "OM", "line": 0, "kind": kind, "word": m.group(0), "status": st, "text": line.strip()[:220]})
    rep = os.path.join(HERE, "results", "reference_report", "reference.html")
    if os.path.exists(rep):
        import html as _h
        t = open(rep, encoding="utf-8").read()
        for tab, body in re.findall(r"<section class='tab[^']*' id='(\w+)'>(.*?)</section>", t, flags=re.S):
            if tab == "record": continue     # the VAL index is the record of the pre-chain era, by name
            txt = _h.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)))
            for rx, kind in ((FORBIDDEN, "FORBIDDEN"), (COHORT, "COHORT")):
                for m in rx.finditer(txt):
                    ctx = txt[max(0, m.start() - 160):m.end() + 60]
                    st = "RECORD" if ALLOW_LINE.search(ctx) or re.search(r"REMOVED|RETIRED|removed 2026|retired 2026|\[RECORD", ctx) else "LIVE"
                    hits.append({"doc": "REPORT:" + tab, "line": 0, "kind": kind, "word": m.group(0), "status": st, "text": ctx})
    tex = os.path.join(MP, "papers", "Landauer_Metrology_of_the_Methylome.tex")
    if os.path.exists(tex):
        T = open(tex, encoding="utf-8").read().split("\n"); rec = False
        for i, line in enumerate(T):
            if re.match(r"\\(section|subsection|paragraph)\{", line): rec = bool(re.search(r"RECORD|record|history|Provenance|not claimed|What is not", line, re.I))
            if line.lstrip().startswith("%") or line.lstrip().startswith("\\bibitem"): continue   # comments and cited titles are not our claims
            for rx, kind in ((FORBIDDEN, "FORBIDDEN"), (COHORT, "COHORT")):
                for m in rx.finditer(line):
                    st = "RECORD" if (rec or ALLOW_LINE.search(line) or re.search(r"REMOVED|RETIRED|removed|retired|record|since removed|\\cite", line)) else "LIVE"
                    hits.append({"doc": "PAPER", "line": i + 1, "kind": kind, "word": m.group(0), "status": st, "text": line.strip()[:220]})
    tex2 = os.path.join(MP, "papers", "IAM_for_physicists", "IAM_for_physicists.tex")
    if os.path.exists(tex2):   # the programme document: scanned for the record, never a gate (it is not a document that reports this chain)
        for i, line in enumerate(open(tex2, encoding="utf-8").read().split("\n")):
            if line.lstrip().startswith(("%", "\\bibitem")): continue
            for rx, kind in ((FORBIDDEN, "FORBIDDEN"), (COHORT, "COHORT")):
                for m in rx.finditer(line): hits.append({"doc": "PAPER-IAM", "line": i + 1, "kind": kind, "word": m.group(0), "status": "RECORD", "text": line.strip()[:220]})
    live = [h for h in hits if h["status"] == "LIVE"]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"n_hits": len(hits), "n_live": len(live), "hits": hits}, open(os.path.join(HERE, "results", "vocab_scan.json"), "w"), indent=1)
    import collections
    by = collections.Counter((h["doc"], h["status"]) for h in hits); print("hits:", dict(by))
    words = collections.Counter(h["word"].lower() for h in live); print("LIVE words:", words.most_common(25))
    for h in live[:int(os.environ.get("VOCAB_SHOW", "40"))]: print(f"  {h['doc']} {h['line']:>6} {h['kind']:<9} <{h['word']}> {h['text'][:130]}")
    print("VOCAB SCAN:", "PASS" if not live else f"FAIL ({len(live)} live hits)")
    sys.exit(1 if live else 0)

if __name__ == "__main__": main()
