"""Every sequencing instrument's full record, printed the same way in the SOP, the Decoder and the sample report.
Source: manual/instrument_records/ (one file per instrument), generated from the instruments database (never edited by hand). Each instrument: what it is in the
field's terms, how it measures, where IAM's Law enters, the cosmology tool it borrows, credit, every rule's full record (how it works, why,
physics, data, numbers, method and standard, known-answer test, limits, where the result goes, code, how to rebuild it, history), the
questions a researcher will ask with answers and evidence, every input and every output with all of their fields, and the code files,
linked in the repository at the chain version given (main when none)."""
import os, json
H = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(H, "instrument_records")
REPO = "https://github.com/hmahaffeyges/IAM-Validation/blob/{ref}/{path}"
from urllib.parse import quote as _q
def load():
    import glob
    return {"instruments": {json.load(open(f, encoding="utf-8"))["instrument"]: json.load(open(f, encoding="utf-8"))["record"] for f in sorted(glob.glob(os.path.join(SRC, "*.json")))}}
def _e(t): return str(t or "").replace("|", "/").strip()
def markdown(ref="main", level=2, only=None, io=True):
    """Lines of markdown. level: heading level of each instrument; only: a set of instrument ids; io: print every input and output."""
    J = load()["instruments"]; h = "#" * level; L = []
    for lid in sorted(J, key=lambda k: [int(x) if x.isdigit() else x for x in k.replace("seq.", "").replace("b", ".5").split(".")] if k.startswith("seq.") else [k]):
        if only and lid not in only: continue
        x = J[lid]; L += [f"{h} {_e(x.get('link'))}", ""]
        for k, lab in (("field_term", "In the field's terms"), ("reads_from", "Reads from"), ("field_how", "What it measures and how"), ("law", "Where IAM's Law enters"), ("cosmo_tool", "Tool borrowed from cosmology"), ("credit", "Credit")):
            if x.get(k): L.append(f"- **{lab}:** {_e(x[k])}")
        if x.get("tools"): L.append("- **Methods it uses (established, not IAM's Law):** " + "; ".join(_e(t if isinstance(t, str) else t.get("tool", "") + " (" + t.get("credit", "") + ")") for t in x["tools"]))
        L.append("")
        for r in x.get("records", []):
            L += [f"{h}# {_e(r['name'])}", "", f"*Status:* {_e(r.get('status'))}", ""]
            for lab, v in r["fields"]: L.append(f"- **{lab}:** {_e(v) or 'open: not written yet'}")
            if r.get("paths"): L.append("- **Files in the repository:** " + ", ".join(f"[{os.path.basename(p)}]({REPO.format(ref=ref, path=_q(p))})" for p in r["paths"]))
            if r.get("questions"):
                L += ["", "*Questions a researcher will ask:*", ""]
                for q, a, ev in r["questions"]: L += [f"- **{_e(q)}** {_e(a)} *(Evidence: {_e(ev)})*"]
            L.append("")
        if io:
            for kind, lab in (("inputs", "Input"), ("outputs", "Output")):
                for r in x.get(kind, []):
                    L += [f"{h}# {lab}: {_e(r['name'])}", ""] + [f"- **{a}:** {_e(b)}" for a, b in r["fields"]] + [""]
    return L
if __name__ == "__main__":
    import sys; t = "\n".join(markdown(sys.argv[1] if len(sys.argv) > 1 else "main")); print(len(t.splitlines()), "lines")
