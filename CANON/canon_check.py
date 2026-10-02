#!/usr/bin/env python3
"""canon_check.py - hold every LIVE file in the repo to CANON/iam_canon.json, the single source of truth for constants and names.

  python3 CANON/canon_check.py            scan the repo, write CANON/canon_report.json, exit 1 on any LIVE 'block' hit
  python3 CANON/canon_check.py --glossary regenerate CANON/GLOSSARY.md from the canon file (never edit GLOSSARY.md by hand)

Severity: block = push refused; warn = listed, push allowed; confirm = the author has not yet ruled on the name.
A hit is RECORD (allowed) when the file sits in a skip directory or a record marker appears within the 40 lines above it."""
import os, re, sys, json
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
C=json.load(open(os.path.join(HERE,"iam_canon.json"),encoding="utf-8"))
def glossary():
    L=["# IAM canon: constants and names", "", f"Generated from `iam_canon.json` v{C['version']} ({C['date']}). Do not edit by hand.", "", C["rule"], "",
       "## Constants", "", "| name | value | units | derivation / source |", "|---|---|---|---|"]
    for k,v in C["constants"].items():
        val=v["value"]; val=f"{val:.6g}" if isinstance(val,float) else str(val)
        L.append(f"| {k} | {val} | {v.get('units','')} | {v.get('derivation','')} {('— '+v['source']) if v.get('source') else ''} |")
    L+=["", "## Names in use", "", "| name | status | definition |", "|---|---|---|"]
    for k,v in C["names"].items(): L.append(f"| {k} | {v['status']} | {v['definition']} |")
    L+=["", "## Retired or pending names", "", "| pattern | use instead | severity |", "|---|---|---|"]
    for k,v in C["retired"].items(): L.append(f"| `{k}` | {v['replaced_by']} | {v['severity']} |")
    open(os.path.join(HERE,"GLOSSARY.md"),"w",encoding="utf-8").write("\n".join(L)+"\n"); print("GLOSSARY.md written")
    import importlib.util as _u; _s=_u.spec_from_file_location("c2t",os.path.join(HERE,"canon_to_tex.py")); _m=_u.module_from_spec(_s); _s.loader.exec_module(_m); _m.main()
def scan():
    S=C["scope"]; skip=[os.path.normpath(x) for x in S["skip_dirs"]]; mark=re.compile("|".join(re.escape(m) for m in S["record_markers"]))
    pats=[(re.compile(p, re.I if v["severity"]!="block" else 0),p,v) for p,v in C["retired"].items()]
    hits=[]
    for dp,dn,fn in os.walk(ROOT):
        rel=os.path.relpath(dp,ROOT)
        if any(rel==s or rel.startswith(s+os.sep) or (os.sep+s+os.sep) in (os.sep+rel+os.sep) for s in skip): dn[:]=[]; continue
        if rel.startswith("CANON"): continue
        for f in fn:
            if not any(f.endswith(e) for e in S["extensions"]): continue
            p=os.path.join(dp,f); rp=os.path.relpath(p,ROOT)
            if any(rp==r or (r.endswith("/") and rp.startswith(r)) for r in S.get("record_files",[])): continue
            try: lines=open(p,encoding="utf-8",errors="replace").read().split("\n")
            except Exception: continue
            for i,l in enumerate(lines):
                for rx,pat,v in pats:
                    for m in rx.finditer(l):
                        rec=bool(mark.search("\n".join(lines[max(0,i-40):i+1])))
                        hits.append(dict(file=os.path.relpath(p,ROOT),line=i+1,pattern=pat,use=v["replaced_by"],severity=v["severity"],status="RECORD" if rec else "LIVE",text=l.strip()[:160]))
    json.dump(hits,open(os.path.join(HERE,"canon_report.json"),"w"),indent=0)
    live=[h for h in hits if h["status"]=="LIVE"]
    from collections import Counter
    print("LIVE hits by pattern/severity:", dict(Counter((h["pattern"],h["severity"]) for h in live)))
    print("files with LIVE hits:", len({h['file'] for h in live}))
    blk=[h for h in live if h["severity"]=="block"]
    for h in blk[:15]: print("BLOCK", h["file"], h["line"], h["text"][:100])
    return 1 if blk else 0
if __name__=="__main__":
    if "--glossary" in sys.argv: glossary(); sys.exit(0)
    sys.exit(scan())
