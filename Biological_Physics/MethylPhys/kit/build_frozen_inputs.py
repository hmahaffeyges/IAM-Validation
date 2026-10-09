#!/usr/bin/env python3
"""Regenerate chain/FROZEN_INPUTS_v3.json (never edit it by hand; STATUS.md document rules).
Re-hashes every listed frozen input. --add PATH adds a new one; --supersede OLD=NEW moves OLD to 'superseded' (with its last hash) and adds NEW.
Paths are relative to chain/. Exits non-zero if a listed file is missing."""
import argparse, hashlib, json, os, sys, time
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chain"); P = os.path.join(HERE, "FROZEN_INPUTS_v3.json")
ap = argparse.ArgumentParser(); ap.add_argument("--add", action="append", default=[]); ap.add_argument("--supersede", action="append", default=[])
ap.add_argument("--why", default=""); a = ap.parse_args()
F = json.load(open(P)); sha = lambda f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
sup = F.setdefault("superseded", [])
for pair in a.supersede:
    old, new = pair.split("="); e = {"file": old, "sha256": F["files"].pop(old, None), "superseded_by": new, "date": time.strftime("%Y-%m-%d"), "why": a.why}
    sup.append(e) if isinstance(sup, list) else sup.__setitem__(old, e); a.add.append(new)
for f in a.add: F["files"][f] = None
missing = [f for f in F["files"] if not os.path.exists(os.path.join(HERE, f))]
if missing: sys.exit(f"missing frozen inputs: {missing}")
F["files"] = {f: sha(f) for f in F["files"]}; json.dump(F, open(P, "w"), indent=1); print(f"{len(F['files'])} frozen inputs hashed")
