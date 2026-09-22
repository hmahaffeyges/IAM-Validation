#!/usr/bin/env python3
"""finding_check.py — RUNBOOK §14 steps 2, 3, 5, 6 as assertions.
Usage: python3 finding_check.py <ID> [--retires "phrase" ...] [--registers RECON FALSIFICATION COSMO COMMISSIONING SWITCHING FUTURE]
                                      [--doors] [--pdf path]
Exit 0 only if: the ID appears in every named register; (with --doors) the ID appears in every door; no retired phrase
renders in the PDF outside a sentence containing WITHDRAWN/CORRECTED/SUPERSEDED/RETIRED."""
import sys, os, re, argparse, glob
HERE = os.path.dirname(os.path.abspath(__file__)); BP = os.path.abspath(os.path.join(HERE, "..", ".."))
I3 = os.path.join(BP, "MethylPhys", "Issue003")
REG = {"RECON": ("data003.py", r"RECON\s*(\+=|=)"), "FALSIFICATION": ("data003.py", r"FALSIFICATION\s*(\+=|=)"), "COSMO": ("data003.py", r"COSMO_EVIDENCE"),
       "COMMISSIONING": (os.path.join(BP, "MethylPhys", "CHAIN_COMMISSIONING.md"), None), "SWITCHING": ("switching_order.py", None), "FUTURE": ("data003.py", r"FUTURE_GOALS")}
DOORS = [os.path.join(BP, "HANDOFF.md"), os.path.join(BP, "README.md"), os.path.join(BP, "MethylPhys/chain", "README.md"), os.path.join(BP, "Record", "README.md"),
         os.path.join(BP, "MethylPhys/atlas", "README.md"), os.path.join(HERE, "RUNBOOK.md"), os.path.join(BP, "MethylPhys/chain", "CPG_Lessons_Learned_2026-06-29.md"),
         os.path.join(BP, "MethylPhys/chain", "README's", "README_FOR_FUTURE_AI.md")] + glob.glob(os.path.join(BP, "MethylPhys", "SOP", "CPG_Chain_of_Custody_SOP_v2*.md"))
def read(p): return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("--retires", nargs="*", default=[]); ap.add_argument("--registers", nargs="*", default=[]); ap.add_argument("--doors", action="store_true"); ap.add_argument("--pdf", default=os.path.join(I3, "IAMPerformance_GAPEIssue003_RC1.pdf"))
    a = ap.parse_args(); fails = []
    VAR = {"RECON": "RECON", "FALSIFICATION": "FALSIFICATION", "COSMO": "COSMO_EVIDENCE", "FUTURE": "FUTURE_GOALS"}
    def scoped_blocks(txt, name):
        """Every `NAME = [...]`, `NAME += [...]`, `NAME.insert(...)`, `NAME.append(...)`, `NAME[i] = (...)` statement, bracket-matched."""
        out = ""
        for mm in re.finditer(rf"^{name}\s*(?:=|\+=)\s*\[|^{name}\.(?:insert|append)\(|^{name}\[\d+\]\s*=\s*\(", txt, re.M):
            j = mm.end() - 1; open_, close = txt[j], {"[": "]", "(": ")"}[txt[j]]; depth = 0
            while j < len(txt):
                depth += (txt[j] == open_) - (txt[j] == close); j += 1
                if depth == 0: break
            out += txt[mm.start():j] + "\n"
        return out
    for r in a.registers:
        f, _ = REG[r]; p = f if os.path.isabs(f) else os.path.join(I3, f); txt = read(p)
        if r in VAR: txt = scoped_blocks(txt, VAR[r])
        if a.id not in txt: fails.append(f"register {r}: '{a.id}' not in the {r} block(s) of {os.path.relpath(p, BP)}")
    if a.doors:
        for d in DOORS:
            if a.id not in read(d): fails.append(f"door: '{a.id}' not in {os.path.relpath(d, BP)}")
    if a.retires:
        try:
            import pypdfium2 as pdfium; doc = pdfium.PdfDocument(a.pdf); pages = [doc[i].get_textpage().get_text_range().replace("\r", " ").replace("\n", " ") for i in range(len(doc))]
        except Exception as e: fails.append(f"pdf unreadable: {e}"); pages = []
        for ph in a.retires:
            for i, t in enumerate(pages):
                for m in re.finditer(re.escape(ph), t):
                    ctx = t[max(0, m.start() - 220): m.end() + 220]
                    if not re.search(r"WITHDRAWN|CORRECTED|SUPERSEDED|RETIRED|retired|withdrawn|superseded|corrected", ctx):
                        fails.append(f"retired phrase live on p{i+1}: '{ph}' … {ctx[:120]!r}")
    if fails:
        print(f"FINDING CHECK {a.id}: FAIL"); [print("  -", f) for f in fails]; sys.exit(1)
    print(f"FINDING CHECK {a.id}: PASS ({len(a.registers)} registers, {'doors, ' if a.doors else ''}{len(a.retires)} retired phrases)")
# ---- detection-language scan (author's rule 2026-09-21) ----
def detection_scan(root):
    import re, os
    files=["MethylPhys/manual/data003.py","MethylPhys/manual/build_gape_issue003.py","HANDOFF.md","README.md","MethylPhys/papers/Landauer_Metrology_of_the_Methylome.tex"]
    pat=re.compile(r"[^.\n]{0,100}\b(cannot (?:resolve|detect|tell|distinguish|determine|name|read)|no [a-z-]* ?claim is supported|only (?:breast|immune) (?:has|carries)|never (?:carries|shows|contains)|\bdetects\b|validated (?:for|on|in) (?!patient care))\b[^.\n]{0,100}",re.I)
    ok=re.compile(r"not yet tested|PROC-[A-Z]+-\d+|prior art|MethylIT|Sanchez|Planck|cohort comparison|cohort-only|cohort-relative|Cohort validation|A cohort cannot|by construction|retired|withdrawn|superseded|Convergence diagnostics|C1 |DETECTION_RULE|no definitive statement|run on that question|calibrated on cohorts|the absolute reading is not yet|sealed procedure that ran the chain",re.I)
    hits=[]
    for f in files:
        fp=os.path.join(root,f)
        if not os.path.exists(fp): continue
        for m in pat.finditer(open(fp,encoding="utf-8",errors="replace").read()):
            if not ok.search(m.group(0)): hits.append((f,m.group(0).strip()[:200]))
    for f,h in hits: print(f"  DETECTION-LANGUAGE {f}: {h}")
    print(f"detection scan: {len(hits)} unqualified detection statements -> {'PASS' if not hits else 'FAIL'}"); return not hits
if __name__=="__main__" and "--detection-scan" in sys.argv:
    import os; sys.exit(0 if detection_scan(os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))) else 1)

if __name__ == "__main__": main()


