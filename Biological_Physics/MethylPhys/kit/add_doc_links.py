#!/usr/bin/env python3
"""Turn filenames written in prose into links, and link the interface and the manual where they are named.

A document that names `cpg_conductor.py` in backticks tells a reader the file exists; a link takes them to it.
The SOP named 200 files and linked none of them.

Rules, deliberately conservative:
  * a name is linked only when exactly ONE tracked file in the live tree carries that basename - no guessing
  * the FIRST occurrence in each section (heading-delimited) is linked, not all of them, so a section that
    mentions one file twenty times does not become twenty links
  * a name followed by an annotation in the same backtick span - "(not part of the chain" - is left alone
  * already-linked names are skipped, so the script is idempotent and safe to run from a generator
Run: python3 add_doc_links.py [file ...]   (default: the live documentation set)
"""
import os, re, subprocess, sys, collections

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
# not already a link: in a linked name the opening backtick is preceded by "[", so skip that case
NAME = re.compile(r"(?<!\]\()(?<!\[)`([A-Za-z0-9_][A-Za-z0-9_.\-]*\.(?:py|json|csv|tsv|md|npy|xz))`")
HEAD = re.compile(r"^#{1,6} |^\*\*Stage ", re.M)
SKIP_IF_FOLLOWED = ("(not part of the chain", "(not in this repository", "(historical path")


def _tracked():
    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True).stdout.split("\n")
    by = collections.defaultdict(list)
    for f in out:
        if f and "RETIRED" not in f and "author_copies" not in f:
            by[os.path.basename(f)].append(f)
    return by


def _enc(rel):
    """Percent-encode what markdown cannot carry in a link target.

    Several runtime files live under `Runtime Matrices/`, and a space in a link target breaks the link on
    GitHub even though the path resolves on disk. Parentheses break it too.
    """
    return rel.replace("%", "%25").replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def link_file(path, by_base, extra=()):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    src = open(full, encoding="utf-8").read()
    doc_dir = os.path.dirname(full)
    # section boundaries
    bounds = [m.start() for m in HEAD.finditer(src)] + [len(src)]
    if not bounds or bounds[0] != 0:
        bounds = [0] + bounds
    out, n = [], 0
    for a, b in zip(bounds, bounds[1:]):
        seg = src[a:b]
        # names already linked in this section count as seen, so a second run does not go on to link the
        # section's next occurrence - the pass must be idempotent to be safe inside a generator
        seen = set(re.findall(r"\[`([^`]+)`\]\(", seg))

        def sub(m):
            nonlocal n
            name = m.group(1)
            tail = seg[m.end():m.end() + 40]
            if any(tail.lstrip().startswith(s) for s in SKIP_IF_FOLLOWED):
                return m.group(0)
            cand = by_base.get(name, [])
            if len(cand) != 1 or name in seen:
                return m.group(0)
            seen.add(name)
            rel = os.path.relpath(os.path.join(ROOT, cand[0]), doc_dir)
            n += 1
            return "[`" + name + "`](" + _enc(rel) + ")"

        out.append(NAME.sub(sub, seg))
    new = "".join(out)
    # the interface and the manual, by name, first mention in the document
    for phrase, target in extra:
        rel = _enc(os.path.relpath(os.path.join(ROOT, target), doc_dir))
        # compare against the RELATIVE target actually written, or every run adds another link
        if phrase in new and "](" + rel not in new:
            new = new.replace(phrase, "[" + phrase + "](" + rel + ")", 1)
            n += 1
    if new != src:
        open(full, "w", encoding="utf-8").write(new)
    return n


def main(paths=None):
    by = _tracked()
    MANUAL = "Biological_Physics/MethylPhys/manual/MethylPhys_CPG_Operations_Manual.pdf"
    IFACE = "Biological_Physics/MethylPhys/chain/MethylPhys_Interface/build_methylphys.py"
    extra = (("Issue 003", MANUAL), ("MethylPhys report", IFACE))
    if not paths:
        paths = [p for p in subprocess.run(["git", "-C", ROOT, "ls-files", "*.md"], capture_output=True,
                                           text=True).stdout.split("\n")
                 if p and "RETIRED" not in p and "author_copies" not in p and "/MethylPhys/" in p]
    total = 0
    for p in paths:
        k = link_file(p, by, extra)
        if k:
            print(f"  {k:>4} links  {p}")
        total += k
    print(f"links added: {total}")
    return total


if __name__ == "__main__":
    main(sys.argv[1:] or None)
