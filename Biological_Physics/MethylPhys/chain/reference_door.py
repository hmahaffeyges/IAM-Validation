"""The reference door (2026-11-07, lock 1 of the anti-drift locks; author: "healthy for comparisons MUST always come from the atlas's ... never tare
our instruments with anything other than what we can be certain is healthy"; "we need to make sure this doesnt happen again").

The chain's healthy references are the published atlases only: IAM-Atlas v2.1 (atlas/v2.1, its compact_manifest.json) and the IAM Sequencing
Atlas's files the chain runs on (chain/Runtime Matrices/Seq_Atlas, listed with their checksums in reference_manifest.json, which this file builds).
Every run of the one command checks those files against their checksums before any reading (check), and every bundle records which atlas it read
and every tare a reading applied (record). A file that does not match its checksum stops the run (REFERENCE_REFUSED): a reading against an
altered reference is not a reading against the atlas.

Usage: python3 reference_door.py build   (writes reference_manifest.json; re-run when a Seq_Atlas file is rebuilt)
       python3 reference_door.py check   (prints the check; exit 1 if any file differs)"""
import os, sys, json, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
V21 = os.path.join(HERE, "..", "atlas", "v2.1")
SEQ = os.path.join(HERE, "Runtime Matrices", "Seq_Atlas")
MANIFEST = os.path.join(HERE, "reference_manifest.json")
_done = {}

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def _seq_files():
    out = []
    for d, _, fs in os.walk(SEQ):
        for f in fs:
            if f.endswith((".parquet", ".json")) and not f.startswith("."): out.append(os.path.relpath(os.path.join(d, f), SEQ))
    return sorted(out)

def build():
    M = {"_about": "checksums of the IAM Sequencing Atlas files the chain reads as its healthy reference (chain/Runtime Matrices/Seq_Atlas); built by reference_door.py build",
         "files": {f: dict(bytes=os.path.getsize(os.path.join(SEQ, f)), sha256=_sha(os.path.join(SEQ, f))) for f in _seq_files()}}
    json.dump(M, open(MANIFEST, "w"), indent=1); return len(M["files"])

def check():
    """Each atlas file against its published checksum; cached for the process. Returns dict(ok, atlases, files, refused)."""
    if "r" in _done: return _done["r"]
    refused, n = [], 0
    V = json.load(open(os.path.join(V21, "compact_manifest.json")))
    for f, v in V.items():
        if not isinstance(v, dict) or "sha256" not in v: continue
        p = os.path.join(V21, f)
        if os.path.exists(p):
            n += 1
            if os.path.getsize(p) != v.get("bytes", os.path.getsize(p)) or _sha(p) != v["sha256"]: refused.append("IAM-Atlas v2.1/" + f)
    S = json.load(open(MANIFEST))["files"] if os.path.exists(MANIFEST) else {}
    if not S: refused.append("reference_manifest.json missing")
    for f, v in S.items():
        p = os.path.join(SEQ, f)
        if not os.path.exists(p): continue   # a file not shipped with this copy is not read
        n += 1
        if os.path.getsize(p) != v["bytes"] or _sha(p) != v["sha256"]: refused.append("IAM Sequencing Atlas/" + f)
    extra = [f for f in _seq_files() if f not in S]
    if extra: refused += ["not in the reference manifest: Seq_Atlas/" + f for f in extra]
    _done["r"] = dict(ok=not refused, atlases=["IAM-Atlas v2.1", "IAM Sequencing Atlas"], files_checked=n, refused=refused)
    return _done["r"]

def record(o):
    """The bundle's reference block: the atlases checked and every tare a reading in this bundle applied (a tare divides by other samples; the
    same-run healthy tare of the array instruments commissioned before 2026-11-07 is named as such, until their re-commissioning on the atlas)."""
    tares = []
    t = (o or {}).get("tare") or {}
    if t.get("A_rel") is not None: tares.append(dict(reading="Met-A (neutrophils)", tare=t.get("state", "tared"), on="same-run healthy arrays (commissioned before the atlas-only rule; being re-commissioned)"))
    for k in ("met_a_whole", "metaa_atlas"):
        x = (o or {}).get(k) or ((o or {}).get("development") or {}).get(k) or {}
        if isinstance(x, dict) and x.get("A_rel") is not None: tares.append(dict(reading=k, tare=x.get("state", "tared"), on="the healthy samples given with --refs"))
    r = check()
    return dict(healthy_reference=r["atlases"], checked=r["ok"], files_checked=r["files_checked"], refused=r["refused"], tares=tares,
                rule="the atlases are the only healthy reference (Decoder: 'The atlases are the only healthy reference, in every instrument')")

if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]: print("reference_door: manifest of", build(), "Seq_Atlas files"); sys.exit(0)
    r = check(); print("reference_door:", "ok" if r["ok"] else "REFUSED", r["files_checked"], "files", "; ".join(r["refused"][:5])); sys.exit(0 if r["ok"] else 1)
