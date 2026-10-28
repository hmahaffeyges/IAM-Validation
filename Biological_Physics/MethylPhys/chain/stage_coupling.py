"""stage_coupling.py - nearest-neighbor coupling on the A-score gauge (DEVELOPMENT, not commissioned; plan 49.5).

coupling (Stage Q2): adjacent unmethylated pairs on held molecules over the number independent placement gives each molecule from its own
length and error count; 1 when errors fall independently, above 1 when they cluster. coupling-A = the sample's coupling over the healthy
coupling of its own cell type (IAM-Atlas v2.1's healthy sequencing inputs, Runtime Matrices/Coupling/coupling_reference_v1.json), so a
healthy sample reads 1. The gauge's ends are counted, not fitted: 0 when no two errors are ever neighbors, and the ceiling, the sample's own
most-clustered arrangement (every molecule's errors in one run) over the same healthy coupling. Recorded on every sequencing count; read
and flagged in development until commissioned."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "Runtime Matrices", "Coupling", "coupling_reference_v1.json")
FLAG = "DEVELOPMENT - not commissioned (nearest-neighbor coupling, plan 49.5)"

def reference(path=REF):
    return json.load(open(path)) if os.path.exists(path) else None

def read(q, cell, ref=None):
    """q: the IAM-A record of Stage Q2 (its count carries coupling and coupling_max); cell: the sample's cell type."""
    c = (q or {}).get("count") or {}; R = ref if ref is not None else reference()
    out = dict(stage="Q2-coupling", reading="coupling-A", cell=cell, coupling=c.get("coupling"), flag=FLAG, A=None)
    if c.get("coupling") is None: out["reason"] = "no held molecule with two or more errors: the coupling is not counted"; return out
    if R is None: out["reason"] = "the healthy coupling reference is not built yet (DEV-ALL-INFO-01, coupling_02)"; return out
    h = (R.get("reference") or {}).get(cell)
    if h is None: out["reason"] = f"no healthy coupling for '{cell}' in the reference ({len(R.get('reference') or {})} cell types)"; return out
    b = R.get("band") or {}; A = c["coupling"] / h
    out.update(A=round(A, 4), healthy=round(h, 4), floor=0.0, ceiling=round(c["coupling_max"] / h, 4) if c.get("coupling_max") else None,
               band=dict(lo=b.get("lo"), hi=b.get("hi")), reference_n=b.get("n"),
               state=("Normal" if b.get("lo") is not None and b["lo"] <= A <= b["hi"] else ("above Normal" if b.get("hi") is not None and A > b["hi"] else "below Normal")))
    return out
