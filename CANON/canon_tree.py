"""canon_tree.py (2026-10-20): canon_check.py over the WHOLE committed tree, read from git, so a partial (sparse) checkout cannot pass a
retired value that sits in a folder it lacks. Same rules as canon_check.py (skip dirs, record files, record markers, the CANON folder itself).
Usage: python3 CANON/canon_tree.py [REV]   (default HEAD); exit 1 on a LIVE block hit."""
import json, re, subprocess, os, sys, collections
REV=sys.argv[1] if len(sys.argv) > 1 else "HEAD"
C=json.loads(subprocess.run(["git","show",f"{REV}:CANON/iam_canon.json"],capture_output=True,text=True).stdout); S=C["scope"]
skip=[os.path.normpath(x) for x in S["skip_dirs"]]; mark=re.compile("|".join(re.escape(m) for m in S["record_markers"])); rf=S.get("record_files",[])
files=[f for f in subprocess.run(["git","ls-tree","-r","--name-only",REV],capture_output=True,text=True).stdout.split("\n") if f and any(f.endswith(e) for e in S["extensions"])]
def skipped(rel): return any(rel==s or rel.startswith(s+os.sep) or (os.sep+s+os.sep) in (os.sep+rel+os.sep) for s in skip) or any(rel==r or (r.endswith("/") and rel.startswith(r)) for r in rf)
files=[f for f in files if not skipped(f) and not f.startswith("CANON")]
pats=[(re.compile(k),v) for k,v in C["retired"].items() if v.get("severity")=="block"]
p=subprocess.Popen(["git","cat-file","--batch"],stdin=subprocess.PIPE,stdout=subprocess.PIPE); hits=[]
for f in files:
    p.stdin.write(f"{REV}:{f}\n".encode()); p.stdin.flush(); hdr=p.stdout.readline().split()
    if len(hdr)<3: continue
    data=p.stdout.read(int(hdr[2])); p.stdout.read(1)
    try: L=data.decode("utf-8").split("\n")
    except UnicodeDecodeError: continue
    for i,l in enumerate(L):
        for rx,v in pats:
            if rx.search(l) and not mark.search("\n".join(L[max(0,i-40):i+1])): hits.append((f,i+1,l.strip()[:120]))
print("canon_tree: " + ("ok (" + str(len(files)) + " files of the whole tree at " + REV + ")" if not hits else f"REFUSED: {len(hits)} LIVE block hit(s): " + "; ".join(f"{f}:{n}" for f,n,_ in hits[:8])))
sys.exit(1 if hits else 0)
