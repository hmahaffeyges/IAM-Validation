"""CANON/undate.py - the one rule every generator applies to canon text before printing it in a public document (2026-10-23, plan 39.15;
author: dates belong in the logs, not in prose). The canon keeps its dates as data; a page never prints them. Removes a date with the words
that only carry it ("ADDED 2026-10-14:", "superseded 2026-10-19", "(author, 2026-10-15: ...)" -> "(author: ...)", "(2026-10-14)", "on
2026-10-15") and keeps everything else. A year alone is kept (a citation). Used by catalogue.py, book_toolbox.py, four_domains.py, the
README and status-line generators and the website builder; checked by CANON/no_dates.py on what they write."""
import re
M = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
D = rf"(?:20\d\d-[01]\d-[0-3]\d|{M}\s+[0-3]?\d(?:st|nd|rd|th)?(?:,?\s+20\d\d)?|[0-3]?\d\s+{M}(?:\s+20\d\d)?)"
RULES = [
    (rf"(?i)\badded\s+{D}\s+from\s+[^.]*\.\s*", ""),                     # an edit note: "Added 2026-10-15 from the whiteboard."
    (rf"\b([A-Z]{{3,}}(?:\s[A-Z]{{3,}})?)\s+{D}\s*:", r"\1:"),             # "DONE 2026-10-15:" -> "DONE:"
    (rf"\(\s*(?:[0-3]?\d\s+{M}|{M}\s+[0-3]?\d,?)\s+(20\d\d)\s*\)", r"(\1)"),   # a citation keeps its year
    (rf"(?i)\b(added|corrected|tested|updated|superseded|retired|literature checked|checked)\s+(on\s+)?{D}(\s*\([^)]*\))?\s*[:,]?\s*",
     lambda m: {"corrected": "Corrected: ", "tested": "Tested: ", "superseded": "superseded" + ("," if m.group(0).rstrip().endswith(",") else "") + " ", "retired": "retired" + ("," if m.group(0).rstrip().endswith(",") else "") + " "}.get(m.group(1).lower(), "")),
    (rf"(?i)\((author|author's rule|the author)\s*,?\s*{D}\s*:\s*", r"(\1: "),
    (rf"(?i)\((author|the author)\s*,?\s*{D}\s*\)", r"(\1)"),
    (rf"(?i)\b(author)\s+{D}\b\s*", r"\1 "),
    (rf"\s*\(\s*{D}\s*\)", ""),
    (rf",\s*{D}\b", ""),
    (rf"(?i)\blast run\s+{D}\s*:\s*", "last run: "),
    (rf"(?i)\b(on|of|since|from|until|by|as of)\s+{D}\b\s*", ""),
    (rf"\b{D}\s*[:,;-]?\s*", ""),
]
KEEP = re.compile(r"\]\([^)]*\)|`[^`\n]*`|https?://\S+|\b[\w./-]*20\d\d-[01]\d-[0-3]\d[\w./-]*\.(?:json|md|txt|csv|png|html)\b|\\(?:href|url|input|include)\{[^}]*\}")
def undate(t):
    """Canon text or a whole page with its dates removed; link targets, code, URLs and file names that contain a date are kept."""
    if not isinstance(t, str): return t
    keep = []
    def hold(m): keep.append(m.group(0)); return f"\x00{len(keep) - 1}\x00"
    u = KEEP.sub(hold, t); cap = u[:1].isupper()
    for p, r in RULES: u = re.sub(p, r, u)
    u = re.sub(r"\(\s*\)", "", re.sub(r"(?<=\S)[ \t]{2,}", " ", u)).replace(" ,", ",").replace(" )", ")")
    u = re.sub(r" \.(?=\s|$)", ".", u); u = re.sub(r"(?<!\.)\.\.(?!\.)", ".", u)
    u = re.sub(r"(^|(?<=[.!?]\s))([a-z])", lambda m: m.group(1) + m.group(2).upper(), u) if cap else u
    return re.sub("\x00(\\d+)\x00", lambda m: keep[int(m.group(1))], u)
