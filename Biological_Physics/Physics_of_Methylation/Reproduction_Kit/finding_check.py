#!/usr/bin/env python3
"""finding_check.py — RUNBOOK §14 steps 2, 3, 5, 6 as assertions.
Usage: python3 finding_check.py <ID> [--retires "phrase" ...] [--registers RECON FALSIFICATION COSMO COMMISSIONING SWITCHING FUTURE]
                                      [--doors] [--pdf path]
Exit 0 only if: the ID appears in every named register; (with --doors) the ID appears in every door; no retired phrase
renders in the PDF outside a sentence containing WITHDRAWN/CORRECTED/SUPERSEDED/RETIRED."""
import sys, os, re, argparse, glob
HERE = os.path.dirname(os.path.abspath(__file__)); BP = os.path.abspath(os.path.join(HERE, "..", ".."))
I3 = os.path.join(BP, "Physics_of_Methylation", "Issue003")
REG = {"RECON": ("data003.py", r"RECON\s*(\+=|=)"), "FALSIFICATION": ("data003.py", r"FALSIFICATION\s*(\+=|=)"), "COSMO": ("data003.py", r"COSMO_EVIDENCE"),
       "COMMISSIONING": (os.path.join(BP, "Physics_of_Methylation", "CHAIN_COMMISSIONING.md"), None), "SWITCHING": ("switching_order.py", None), "FUTURE": ("data003.py", r"FUTURE_GOALS")}
DOORS = [os.path.join(BP, "HANDOFF.md"), os.path.join(BP, "README.md"), os.path.join(BP, "CPG_Engine", "README.md"), os.path.join(BP, "Testing_and_Code", "README.md"),
         os.path.join(BP, "IAM_Atlas", "README.md"), os.path.join(HERE, "RUNBOOK.md"), os.path.join(BP, "CPG_Engine", "CPG_Lessons_Learned_2026-06-29.md"),
         os.path.join(BP, "CPG_Engine", "README's", "README_FOR_FUTURE_AI.md")] + glob.glob(os.path.join(BP, "Physics_of_Methylation", "SOP", "CPG_Chain_of_Custody_SOP_v2*.md"))
def read(p): return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("--retires", nargs="*", default=[]); ap.add_argument("--registers", nargs="*", default=[]); ap.add_argument("--doors", action="store_true"); ap.add_argument("--pdf", default=os.path.join(I3, "IAMPerformance_GAPEIssue003_RC1.pdf"))
    a = ap.parse_args(); fails = []
    for r in a.registers:
        f, _ = REG[r]; p = f if os.path.isabs(f) else os.path.join(I3, f); txt = read(p)
        if a.id not in txt: fails.append(f"register {r}: '{a.id}' not in {os.path.relpath(p, BP)}")
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
if __name__ == "__main__": main()
