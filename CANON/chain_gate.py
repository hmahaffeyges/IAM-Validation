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
 f. (2026-10-22, author: "the only way to do this right is to ALWAYS use the chain for everything so a researcher can replicate it using our
    own chain and we dont have to explain why it wasnt used or have to worry about missing steps.") Every script a push adds or changes under
    doors/data/ or website/art/ that uses chain code must run the one command, chain/MethylPhys_Interface/run_sample.py. Importing chain
    stages and calling them directly is refused: that is how Stage 0 and Stage T's self-tare were skipped (METAA-ATLAS-SEAL-01). Rule a now asks
    the same of every script on a synthetic-test line. A script may still import chain code (to plant a loss by the chain's own rule) if it
    also runs the one command, which makes every reading.
 g. (2026-10-22, lesson C34; author: "make sure this is added to the chain ... so it doesnt happen again") No changed script under doors/data/ or
    website/art/ downloads a sequencing file (.pat) itself: the chain fetches it (run_sample.py --geo-pat, or stage_q0_intake.geo_pat), always
    on the build it reads. A script that picked GEO's first file name took the hg38 file and counted it on hg19 positions.
Usage: chain_gate.py register (add the working chain's sha; checkpoint.sh runs it) | check [base] (default origin/main)"""
import os, sys, re, json, glob, hashlib, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CHAIN = "Biological_Physics/MethylPhys/chain"; DOORS = "Biological_Physics/MethylPhys/doors/"; REG = "CANON/chain_versions.json"
ADDS = re.compile(r"(?i)^\*\*Synthetic test:\*\*|^\*\*Bars?\b|^## Outcome|^\*\*Status: rule|^\*\*Sealed|^## Bars?\b")
LINE_NUMS = re.compile(r"(?<![\d.])(0\.0298|0\.0555|0\.74128\d*|0\.0402|0\.0347)(?!\d)")
CHAIN_IMPORT = re.compile(r"(?m)^\s*(from|import)\s+(conductor_v3|stage_\w+|dev_stages|cmb_tools|sky_statistics)\b")
PIECES = re.compile(r"(?m)^\s*(?:from|import)\s+(conductor_v3|stage_\w+|dev_stages|cmb_tools|sky_statistics|report_globe_v4|globe_viewer|report_v\w+|run_sample)\b")
ONE_COMMAND = re.compile(r"run_sample\.py")
SCOPE_F = ("Biological_Physics/MethylPhys/doors/data/", "website/art/")
OWN_PAT = re.compile(r"\\\.pat\\\.gz|\.pat\.gz")
OWN_FETCH = re.compile(r"urlopen|urlretrieve|requests\.get|curl |wget ")
def own_pat_refusal(f, text):
    """Rule 23b-g: a .pat file downloaded by the script instead of by the chain."""
    if OWN_PAT.search(text) and OWN_FETCH.search(text) and not re.search(r"geo_pat|--geo-pat|--prior-geo-pat", text):
        return f"{f}: downloads a .pat file itself; let the chain fetch it on the build it reads (run_sample.py --geo-pat GSM, or stage_q0_intake.geo_pat) (rule 23b-g)"
    return None

def one_command_refusal(f, text):
    """Rule 23b-f: chain code used without the one command."""
    p = sorted(set(PIECES.findall(text)))
    if p and not ONE_COMMAND.search(text):
        return f"{f}: calls chain pieces ({', '.join(p)}) instead of running the one command, chain/MethylPhys_Interface/run_sample.py; every reading must come from it (rule 23b-f)"
    return None
def chain_sha(root=ROOT):
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(root, CHAIN, "*.py")) + glob.glob(os.path.join(root, CHAIN, "MethylPhys_Interface", "*.py"))):   # 2026-10-22: the one command is chain
        h.update(os.path.relpath(f, os.path.join(root, CHAIN)).encode() if "MethylPhys_Interface" in f else os.path.basename(f).encode()); h.update(open(f, "rb").read())
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
        if f.startswith(SCOPE_F) and f.endswith((".py", ".sh")) and os.path.isfile(os.path.join(root, f)):
            _tx = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
            F += [m for m in (one_command_refusal(f, _tx), own_pat_refusal(f, _tx)) if m]
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
        for l in added.get(f, []):   # 2026-10-22: the lines this push adds; older lines are history
            if not l.startswith("**Synthetic test:**"): continue
            for x in re.findall(r"[A-Za-z0-9_/.-]+\.(?:py|sh)", l):
                if x.startswith("chain/") and os.path.isfile(os.path.join(root, CHAIN, x[len("chain/"):])): continue   # chain code is chain code (2026-10-16)
                p = os.path.join(root, DOORS, "data", x)
                if not os.path.isfile(p): F.append(f"{f}: synthetic test {x} is not under doors/data/ (rule 23b-a)"); continue
                if not ONE_COMMAND.search(open(p, encoding="utf-8", errors="replace").read()):
                    F.append(f"{f}: synthetic test {x} does not run the one command (chain/MethylPhys_Interface/run_sample.py; importing chain stages is not enough since 2026-10-22) (rule 23b-a)")
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
