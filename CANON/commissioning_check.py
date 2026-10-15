"""Commissioning gate (author 2026-10-15). Reads CANON/commissioning.json and CANON/results_register.json. Refuses when: a register entry of
kind 'commissioning' with status 'commissioned' has no registry entry; a registry entry lacks a required kind of place; a place's file is
missing or its pattern is not found; a forbidden stale statement is found. Usage: python3 CANON/commissioning_check.py"""
import os, re, sys, json
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C = json.load(open(os.path.join(ROOT, "CANON", "commissioning.json"), encoding="utf-8"))
REG = json.load(open(os.path.join(ROOT, "CANON", "results_register.json"), encoding="utf-8"))["entries"]
P = []; have = {i["register_id"]: i for i in C["instruments"]}
for e in REG:
    if e.get("kind") == "commissioning" and e.get("status") == "commissioned" and e["id"] not in have:
        P.append(f"{e['id']}: commissioned in the results register but not in CANON/commissioning.json (list every place it must appear)")
for i in C["instruments"]:
    kinds = {p["kind"] for p in i["places"]}
    for k in C["required_kinds"]:
        if k not in kinds: P.append(f"{i['register_id']}: no '{k}' place listed")
    for p in i["places"]:
        f = os.path.join(ROOT, p["file"])
        if not os.path.isfile(f): P.append(f"{i['register_id']} [{p['kind']}]: {p['file']} does not exist"); continue
        if not re.search(p["pattern"], open(f, encoding="utf-8", errors="replace").read()):
            P.append(f"{i['register_id']} [{p['kind']}]: {p['file']} does not state the commissioning (pattern {p['pattern'][:60]})")
    for x in i.get("forbid", []):
        f = os.path.join(ROOT, x["file"])
        if os.path.isfile(f):
            m = re.search(x["pattern"], open(f, encoding="utf-8", errors="replace").read())
            if m: P.append(f"{i['register_id']}: stale statement in {x['file']}: '{m.group(0)[:70]}'")
print("commissioning_check: " + ("ok (" + str(len(C["instruments"])) + " commissioned)" if not P else "REFUSED")); [print("  " + p) for p in P]
sys.exit(1 if P else 0)
