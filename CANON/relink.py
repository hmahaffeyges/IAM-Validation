"""Markdown links as GitHub resolves them: relative to the document's own folder (2026-10-19, when the MethylPhys documents moved from the
repository root into Biological_Physics/MethylPhys). fix(md, out) turns a link written from the repository root into one that resolves from
out's folder; broken(path) lists the links in a document that resolve to nothing. Used by the generators that write documents and by
checked_push.sh (strict: no second base is tried)."""
import os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LINK = re.compile(r"(\]\()([^)\s#]+)((?:#[^)\s]*)?\))")
def _ext(t): return re.match(r"^(?:[a-z]+:|/|#|mailto:)", t) is not None
def fix(md, out):
    d = os.path.dirname(os.path.abspath(out))
    def r(m):
        t = m.group(2)
        if _ext(t) or os.path.exists(os.path.join(d, t)): return m.group(0)
        if os.path.exists(os.path.join(ROOT, t)): return m.group(1) + os.path.relpath(os.path.join(ROOT, t), d).replace(os.sep, "/") + m.group(3)
        u = re.sub(r"^(\.\./)+", "", t)   # a link written for the document's old folder, one or more levels up
        if u != t and os.path.exists(os.path.join(ROOT, u)): return m.group(1) + os.path.relpath(os.path.join(ROOT, u), d).replace(os.sep, "/") + m.group(3)
        return m.group(0)
    return LINK.sub(r, md)
def broken(path):
    d = os.path.dirname(os.path.abspath(path)); s = open(path, encoding="utf-8").read()
    s = re.sub(r"```.*?```|`[^`]*`", "", s, flags=re.S)
    return [m.group(2) for m in LINK.finditer(s) if not _ext(m.group(2)) and not os.path.exists(os.path.join(d, m.group(2).replace("%20", " ")))]
if __name__ == "__main__":
    import sys
    if sys.argv[1] == "fix":
        for p in sys.argv[2:]:
            s = open(p, encoding="utf-8").read(); n = fix(s, p)
            if n != s: open(p, "w", encoding="utf-8").write(n); print("relinked", p)
    else:
        bad = {p: broken(p) for p in sys.argv[2:]}; bad = {p: b for p, b in bad.items() if b}
        for p, b in bad.items(): print(f"relink: {p}: {len(b)} broken link(s): {b[:4]}")
        print("relink: " + ("ok" if not bad else "REFUSED")); sys.exit(1 if bad else 0)
