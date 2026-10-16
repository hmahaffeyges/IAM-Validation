#!/usr/bin/env python3
"""Chain gate (rule 23b, author 2026-10-15): "you can use your side script for temporary exploration but for any synthetic test or ANY test that
will be logged and pushed, it MUST run through the chain." Applies to every doors note to which a push ADDS a synthetic-test line, a bar, a
seal or an outcome (older notes are not re-judged until such a line is added to them):
 a. every script on its '**Synthetic test:**' line lives under doors/data/ and imports the chain; the old '# NO CHAIN:' escape is not accepted
 b. every .json / .csv this push changes in that note's data folder carries a valid chain stamp (in-file for .json, <file>.stamp.json beside
    others): payload hash recomputed here, chain_sha listed in CANON/chain_versions.json (a committed chain). Declared inputs are listed in the
    folder's INPUTS.txt and are exempt.
 c. no record cites a scratch/ path (scratch/ is for exploration and is never evidence)
e. a line added to a commissioning note may cite only chain code (CANON/ scripts excepted)
 d. no changed script outside chain/ types the self-reference line's numbers (they live only in chain Stage Q2's runtime matrix)
Usage: chain_gate.py register (add the working chain's sha; checkpoint.sh runs it) | check [base] (default origin/main)"""
import os, sys, re, json, glob, hashlib, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CHAIN = "Biological_Physics/MethylPhys/chain"; DOORS = "Biological_Physics/MethylPhys/doors/"; REG = "CANON/chain_versions.json"
ADDS = re.compile(r"(?i)^\*\*Synthetic test:\*\*|^\*\*Bars?\b|^## Outcome|^\*\*Status: rule|^\*\*Sealed|^## Bars?\b")
LINE_NUMS = re.compile(r"(?<![\d.])(0\.0298|0\.0555|0\.74128\d*|0\.0402|0\.0347)(?!\d)")
CHAIN_IMPORT = re.compile(r"(?m)^\s*(from|import)\s+(conductor_v3|stage_\w+|dev_stages|cmb_tools|sky_statistics)\b")
def chain_sha(root=ROOT):
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(root, CHAIN, "*.py"))): h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    return h.hexdigest()
def registry(root=ROOT):
    p = os.path.join(root, REG); return {v["chain_sha"] for v in json.load(open(p))["versions"]} if os.path.exists(p) else set()
def _canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"), default=float).encode()
def verify_stamp(path, known):
    if path.endswith(".json"):
        try: o = json.load(open(path, encoding="utf-8"))
        except Exception as e: return f"{path}: not readable JSON ({e})"
        s = o.get("chain_stamp") if isinstance(o, dict) else None
        if not s: return f"{path}: no chain stamp - a logged result must be written by the chain (rule 23b)"
        if hashlib.sha256(_canon({k: v for k, v in o.items() if k != "chain_stamp"})).hexdigest() != s.get("payload_sha256"):
            return f"{path}: edited after the chain wrote it (payload hash differs) (rule 23b)"
    else:
        sp = path + ".stamp.json"
        if not os.path.exists(sp): return f"{path}: no chain stamp ({os.path.basename(sp)} missing) (rule 23b)"
        s = json.load(open(sp))["chain_stamp"]
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if h != s.get("payload_sha256"): return f"{path}: edited after the chain wrote it (file hash differs) (rule 23b)"
    if s.get("chain_sha") not in known: return f"{path}: written by a chain version that was never committed ({str(s.get('chain_sha'))[:12]}) - rerun on the committed chain (rule 23b)"
    return None
def check(changed, added, root=ROOT, known=None):
    """changed: repo-relative paths changed by the push; added: {path: [added lines]}. Returns a list of refusals."""
    known = registry(root) if known is None else known; F = []
    for f in changed:
        if f.endswith(".py") and not f.startswith(CHAIN + "/") and f not in ("CANON/chain_gate_controls.py",) and os.path.isfile(os.path.join(root, f)):   # the controls must contain the numbers they test
            hits = set(LINE_NUMS.findall(open(os.path.join(root, f), encoding="utf-8", errors="replace").read()))
            if len(hits) >= 2: F.append(f"{f}: types the self-reference line's numbers {sorted(hits)} outside the chain; ask chain Stage Q2 (rule 23b-d)")
        if f.startswith(DOORS + "COMMISSIONING_NOTE_") and f.endswith(".md"):   # rule e: a commissioning note may add only chain code as evidence
            for l in added.get(f, []):
                for x in re.findall(r"[A-Za-z0-9_/.-]+\.(?:py|sh|R)\b", l):
                    if not x.startswith(("chain/", CHAIN, "CANON/")): F.append(f"{f}: adds {x} as evidence; a commissioning note may cite only chain code (rule 23b-e)")
        if not (f.startswith(DOORS) and f.endswith(".md") and os.path.isfile(os.path.join(root, f))): continue
        text = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        if re.search(r"(?<![\w/])scratch/[\w.-]", text): F.append(f"{f}: cites a scratch/ path; scratch is exploration, never evidence (rule 23b-c)")
        if not any(ADDS.search(l) for l in added.get(f, [])): continue
        nid = os.path.basename(f)[:-3].replace("_OUTCOME", "").replace("_PREREG", "")
        for l in text.split("\n"):
            if not l.startswith("**Synthetic test:**"): continue
            for x in re.findall(r"[A-Za-z0-9_/.-]+\.(?:py|sh)", l):
                p = os.path.join(root, DOORS, "data", x)
                if not os.path.isfile(p): F.append(f"{f}: synthetic test {x} is not under doors/data/ (rule 23b-a)"); continue
                if not CHAIN_IMPORT.search(open(p, encoding="utf-8", errors="replace").read()):
                    F.append(f"{f}: synthetic test {x} does not run through the chain (no chain import; '# NO CHAIN' is no longer accepted for new tests) (rule 23b-a)")
        dd = os.path.join(DOORS, "data", nid)
        inputs = set()
        ip = os.path.join(root, dd, "INPUTS.txt")
        if os.path.exists(ip): inputs = {l.strip() for l in open(ip) if l.strip() and not l.startswith("#")}
        for g in changed:
            if g.startswith(dd + "/") and g.endswith((".json", ".csv")) and not g.endswith(".stamp.json") and os.path.basename(g) not in inputs and os.path.isfile(os.path.join(root, g)):
                m = verify_stamp(os.path.join(root, g), known)
                if m: F.append(m.replace(root + "/", ""))
    return F
def _git(*a): return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout
if __name__ == "__main__":
    if sys.argv[1] == "register":
        p = os.path.join(ROOT, REG); J = json.load(open(p)) if os.path.exists(p) else {"_rule": "chain versions that were committed; a chain stamp must name one (CANON/chain_gate.py)", "versions": []}
        s = chain_sha()
        if s not in {v["chain_sha"] for v in J["versions"]}:
            J["versions"].append({"chain_sha": s, "first_commit_after": _git("rev-parse", "--short=8", "HEAD").strip(), "date": _git("log", "-1", "--format=%cs").strip()})
            json.dump(J, open(p, "w"), indent=1); open(p, "a").write("\n"); print("chain_gate: registered", s[:12])
        else: print("chain_gate: chain version already registered", s[:12])
    else:
        base = sys.argv[2] if len(sys.argv) > 2 else "origin/main"
        changed = [x for x in _git("diff", "--name-only", base, "--").split("\n") if x]
        added = {f: [l[1:] for l in _git("diff", base, "--", f).split("\n") if l.startswith("+") and not l.startswith("+++")] for f in changed if f.endswith(".md")}
        reg_ok = chain_sha() in registry()
        F = check(changed, added) + ([] if reg_ok else [f"the working chain ({chain_sha()[:12]}) is not in {REG}: run python3 CANON/chain_gate.py register"])
        print("chain_gate: " + ("ok" if not F else "REFUSED")); [print("  " + x) for x in F]; sys.exit(1 if F else 0)
