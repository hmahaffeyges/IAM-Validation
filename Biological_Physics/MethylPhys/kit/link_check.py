#!/usr/bin/env python3
"""link_check.py - every relative path a document points at must exist.

Written 2026-09-22 after the tree move left RUNBOOK.md in kit/ while both READMEs pointed at doors/RUNBOOK.md.
The author found it by clicking a link. A path in a document is a claim; this checks the claims.

    python3 link_check.py          the live documentation set - what a reader is handed (exit 1 if broken)
    python3 link_check.py --all    every tracked document, including the pre-build card folders whose
                                   internal links have been broken since long before this tree

Three bases are tried for each reference: relative to the document, to the repository root, and to
Biological_Physics - the corpus writes repo-relative paths both with and without that prefix.
"""
import os
from urllib.parse import unquote
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _root():
    d = HERE
    for _ in range(6):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return d


R = _root()
EXT = ("md", "py", "html", "tex", "txt")
SKIP = re.compile(r"RETIRED|__pycache__|/\.git")

# The live documentation set: the instrument's own documents plus the two front pages. The pre-build
# disease-card folders carry their own long-broken internal links, which are not part of what the chain
# hands a reader; --all includes them.
LIVE = re.compile(
    r"^(README\.md"
    r"|docs/README\.md"
    r"|Biological_Physics/README\.md"
    r"|Biological_Physics/MethylPhys/README\.md"
    r"|Biological_Physics/MethylPhys/(doors|kit|sop|manual|papers)/"
    r"|Biological_Physics/MethylPhys/atlas/[^/]*\.md"
    r"|Biological_Physics/MethylPhys/chain/[^/]*\.md)"
)

LINK = re.compile(r"\]\(([^)\s#]+)\)")
CODEPATH = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./\- ]*\.(?:md|py|json|csv|npz|npy|pdf|xz|tsv))`")
BARE_OK = ("py", "json", "csv", "npz", "npy", "xz", "tsv")


def references(text):
    out = set()
    for m in LINK.finditer(text):
        out.add(m.group(1))
    for m in CODEPATH.finditer(text):
        out.add(m.group(1))
    return out



def _self_reference(docs, R):
    """A line may not say that a file replaces, supersedes or stands instead of itself.

    Added 2026-09-22: a blanket rename of one filename to another turned "X (replaces Y)" into
    "X (replaces X)" in a README table, and the checks in place at the time - leftover old names, unresolved
    filenames - all passed on it. A rename can corrupt a sentence without leaving a bad path behind.
    """
    import re as _re
    pat = _re.compile(r"`([^`]+)`.{0,60}?\b(?:replaces|supersedes|instead of|rather than)\b.{0,60}?`\1`")
    bad = []
    for f in docs:
        try:
            for i, line in enumerate(open(os.path.join(R, f), encoding="utf-8"), 1):
                if pat.search(line):
                    bad.append((f, i, line.strip()[:120]))
        except OSError:
            continue
    return bad

def main():
    show_all = "--all" in sys.argv
    # Tracked AND untracked-but-present files. A new document is exactly where broken links live, and until
    # 2026-09-22 this checked only tracked paths - so REVIEWER_MANIFEST.md passed with a deliberately broken
    # link in it, because it had not been committed yet.
    tracked = subprocess.run(["git", "-C", R, "ls-files"], capture_output=True, text=True).stdout.split("\n")
    untracked = subprocess.run(["git", "-C", R, "ls-files", "--others", "--exclude-standard"],
                               capture_output=True, text=True).stdout.split("\n")
    tracked = tracked + [f for f in untracked if f]
    basenames = {f.split("/")[-1] for f in tracked if f}
    docs = [f for f in tracked
            if f and not SKIP.search(f) and f.split(".")[-1] in EXT and (show_all or LIVE.match(f))]
    bad = []
    checked = 0
    for f in sorted(docs):
        path = os.path.join(R, f)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        base = os.path.dirname(path)
        for ref in references(text):
            # a regex character class or quantifier inside parentheses is a pattern in source code, not
            # a path: build_reviewer_manifest.py's own PROC-id regex read as two broken links (2026-09-23)
            if any(c in ref for c in "[]\\+*?^") and not ref.endswith((".md", ".py", ".json", ".csv")):
                continue
            if ref.startswith(("http", "mailto:", "#", "/")):
                continue
            # a shell command inside link parentheses is not a path
            if ref.split(" ")[0] in ("python", "python3", "bash", "sh", "cd", "export"):
                continue
            # a bare filename with no directory is a file being named, not a link to follow
            if "/" not in ref and ref.split(".")[-1] in BARE_OK:
                continue
            checked += 1
            # Prose in this corpus writes tree-relative paths from several natural roots: the document itself,
            # the repository, Biological_Physics, and the instrument's own directories (a note written while
            # working in chain/ says "Runtime Matrices/x.json"). All are legitimate; try each.
            BASES = (base, R, os.path.join(R, "Biological_Physics"),
                     os.path.join(R, "Biological_Physics", "MethylPhys"),
                     os.path.join(R, "Biological_Physics", "MethylPhys", "chain"),
                     os.path.join(R, "Biological_Physics", "MethylPhys", "atlas"),
                     os.path.join(R, "Biological_Physics", "MethylPhys", "doors"),
                     os.path.join(R, "Biological_Physics", "MethylPhys", "kit"),
                     os.path.join(R, "Biological_Physics", "MethylPhys", "manual"))
            # a reference with no directory is a file being named in prose, not a path to follow: accept it if
            # a file of that name exists anywhere in the tree
            if "/" not in ref and ref in basenames:
                continue
            ref = unquote(ref)   # %20 is how a space reaches a markdown link
            candidates = tuple(os.path.join(b, ref) for b in BASES)
            if not any(os.path.exists(c) for c in candidates):
                bad.append((f, ref))
    selfref = _self_reference(docs, R)
    for f, i, line in selfref:
        bad.append((f, "line %d says a file replaces itself: %s" % (i, line)))
    scope = "all documents" if show_all else "live documentation set"
    print("link_check (%s): %d relative references checked in %d documents" % (scope, checked, len(docs)))
    if bad:
        print("BROKEN (%d):" % len(bad))
        for f, ref in bad[:40]:
            print("   %s -> %s" % (f, ref))
        sys.exit(1)
    print("link_check: PASS - every relative reference resolves")


if __name__ == "__main__":
    main()
