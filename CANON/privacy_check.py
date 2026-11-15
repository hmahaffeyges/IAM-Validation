#!/usr/bin/env python3
"""privacy_check.py (2026-11-14, the author: private product names 'are not supposed to appear anywhere public at all'; only minimal chip
and qubit material, none new). Part of the gate (checked_push.sh). Reads the lines this push ADDS against origin/main, in every file, and
refuses:
  (1) always: a private name (the list in PRIVATE_NAMES below is generic on purpose; the specific private names are matched as hashes so
      this public file never spells them out);
  (2) new chip or qubit material: an added line naming a chip or qubit term, anywhere outside the book's existing Part 5 chapters and their
      verification files (which stay as published). Lines already on main are never flagged; an edited line counts only its NET new
      chip or qubit terms (2026-11-15), so a public row label rewritten unchanged is not new material.
'controls': planted cases (one per rule, and one clean line) must give the expected verdicts. Exit 1 on any finding."""
import hashlib, re, subprocess, sys

# sha256 of the lower-cased private names; never write the names themselves in a public file
PRIVATE_HASHES = set()  # the hashes live in CANON/privacy_hashes.txt
CHIP_QUBIT = re.compile(r"\b(qubits?|transmons?|quasiparticles?|x_qp|xqp|dram|sram|mram|cmos|wafers?|dennard|josephson|gate errors?)\b", re.I)
EXEMPT = ("docs/book/part5/", "docs/verification/particle/", "docs/book/figscripts/fig_p3", "CANON/privacy_check.py")

def words(line):
    return set(re.findall(r"[a-z0-9_]+", line.lower()))

def findings(path, line, hashes):
    f = []
    for w in words(line):
        if hashlib.sha256(w.encode()).hexdigest() in hashes: f.append(f"{path}: a private name is added")
    if not path.startswith(EXEMPT) and CHIP_QUBIT.search(line):
        f.append(f"{path}: new chip or qubit material added ('{CHIP_QUBIT.search(line).group(0)}'); keep it private")
    return f

def changed_lines():
    """Added and removed lines of this push against origin/main, by file."""
    base = subprocess.run(["git", "merge-base", "HEAD", "origin/main"], capture_output=True, text=True).stdout.strip() or "origin/main"
    add, rem, path, old = [], [], None, None
    for l in subprocess.run(["git", "diff", "-U0", base, "HEAD"], capture_output=True, text=True).stdout.split("\n"):
        if l.startswith("--- "): old = l[6:] if l.startswith("--- a/") else None
        elif l.startswith("+++ "): path = l[6:] if l.startswith("+++ b/") else None
        elif l.startswith("+") and path: add.append((path, l[1:]))
        elif l.startswith("-") and old: rem.append((old, l[1:]))
    return add, rem

def added_lines():
    return changed_lines()[0]

def net_new_terms(add, rem):
    """2026-11-15 (a generated public table re-wrote a line whose row label was already public): a chip or qubit term counts as NEW only when
    a file gains more occurrences of it than the push removes from that file. Editing a line that already carried the term (the same
    number of occurrences out and back in) adds no chip or qubit material; any extra occurrence is still refused."""
    import collections
    cnt = collections.Counter()
    for p, l in add:
        if not p.startswith(EXEMPT):
            for m in CHIP_QUBIT.finditer(l): cnt[(p, m.group(0).lower())] += 1
    for p, l in rem:
        for m in CHIP_QUBIT.finditer(l): cnt[(p, m.group(0).lower())] -= 1
    return [f"{p}: new chip or qubit material added ('{t}'); keep it private" for (p, t), n in sorted(cnt.items()) if n > 0]

def private_name_findings(add, hashes):
    return [f"{p}: a private name is added" for p, l in add for w in words(l) if hashlib.sha256(w.encode()).hexdigest() in hashes]

def load_hashes():
    try: return set(open("CANON/privacy_hashes.txt").read().split())
    except FileNotFoundError: return set(PRIVATE_HASHES)

if __name__ == "__main__":
    H = load_hashes()
    if len(sys.argv) > 1 and sys.argv[1] == "controls":
        h = hashlib.sha256(b"plantedname").hexdigest()
        cases = [("Biological_Physics/x.md", "a plantedname appears", {h}, True), ("Biological_Physics/x.md", "a transmon line", set(), True),
                 ("docs/book/part5/p5_02_xqp.tex", "a transmon line", set(), False), ("Biological_Physics/x.md", "held error 0.045", set(), False),
                 ("Biological_Physics/x.md", "the qubit simulator", set(), True), ("Biological_Physics/x.md", "ChIP-seq peaks", set(), False)]
        ok = all(bool(findings(p, l, hs)) == want for p, l, hs, want in cases)
        # net-new controls (2026-11-15): an edited line keeping its existing term passes; an extra term, or a term added with nothing removed, is refused
        X = "FOUR_DOMAINS.md"
        nets = [([(X, "| Gate error | writer 0.0143 |")], [(X, "| Gate error | writer |")], False),
                ([(X, "| Gate error | writer, gate error |")], [(X, "| Gate error | writer |")], True),
                ([(X, "the qubit row")], [], True), ([(X, "held error 0.045")], [], False)]
        ok = ok and all(bool(net_new_terms(ad, rm)) == want for ad, rm, want in nets)
        print("privacy_check controls:", "ok" if ok else "FAILED"); sys.exit(0 if ok else 1)
    _add, _rem = changed_lines()
    F = private_name_findings(_add, H) + net_new_terms(_add, _rem)
    if F:
        print("privacy_check: REFUSED (%d)" % len(F)); [print("  " + x) for x in sorted(set(F))[:20]]; sys.exit(1)
    print("privacy_check: ok (no private name, no new chip or qubit material)")
