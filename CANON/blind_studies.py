"""Writes Biological_Physics/MethylPhys/BLIND_STUDIES.md from CANON/blind_studies.json (the one source). Usage: python3 CANON/blind_studies.py [--check]"""
import json, sys
J = json.load(open("CANON/blind_studies.json", encoding="utf-8")); O = "Biological_Physics/MethylPhys/BLIND_STUDIES.md"
L = ["# Blind studies: published methylation studies read by the chain before their findings are opened", "", "*Generated from CANON/blind_studies.json by CANON/blind_studies.py; do not edit by hand.*", "",
     J["_protocol"], "", "| Study | Platform | Arrays | Design (from the GEO record) | Specimen given to the chain | The question, written before reading | Status | Our reading | Published finding | Verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
for s in J["studies"]:
    L.append(f"| {s['gse']} | {s['platform']} | {s.get('arrays', s.get('n', ''))} | {s['design']} | {s['specimen']} | {s['question']} | {s['status']} | {s.get('ours') or ''} | {s.get('published') or ''} | {s.get('verdict') or ''} |")
L += ["", "**Reserved, not read** (rare designs kept for sealed tests): " + ", ".join(J["reserved"]) + "."]
t = "\n".join(L) + "\n"
if "--check" in sys.argv:
    ok = open(O, encoding="utf-8").read() == t; print("blind_studies:", "ok" if ok else "STALE"); sys.exit(0 if ok else 1)
open(O, "w", encoding="utf-8").write(t); print("blind_studies: wrote", O)
