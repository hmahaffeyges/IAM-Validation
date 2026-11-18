#!/usr/bin/env python3
"""Reference coverage (author 2026-11-17: 'make sure nothing else we added after the fact is missing, like the megakaryocytes, so this doesnt happen
again'). Runs ON A GITHUB MACHINE. Lists every cell the Deconvolvers can name (sequencing and array, every reference version, including development
ones such as the megakaryocyte reference) and every per-cell table the chain's instruments read, then prints, for each nameable cell, which tables
lack it. Reads only column names and json keys; changes nothing."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"], check=True)
import os, re, json, glob, collections
import pyarrow.parquet as pq
RM = "Biological_Physics/MethylPhys/chain"
def norm(s): return re.sub(r"\s+", " ", re.sub(r"\|(mu|sd|se|n|mean|hi|lo)$", "", str(s).strip().lower()).replace("_", " "))
NONCELL = re.compile(r"^(chr|start|end|pos|site|cpg|probe|id|index|name|n|count|mean|sd|se|beta|reads?|depth|coverage|cluster|kind|version|unnamed.*|__.*)$")
def cells_of(path):
    """cell names a table carries: parquet column names, or json keys under 'cells' / 'types' / 'columns'"""
    try:
        if path.endswith(".parquet"):
            cols = pq.read_schema(path).names
            return {norm(c) for c in cols if not NONCELL.match(norm(c)) and not re.match(r"^\d", str(c))}
        if path.endswith(".json"):
            J = json.load(open(path))
            def walk(o, d=0):
                if d > 3: return set()
                if isinstance(o, dict):
                    for k in ("cells", "types", "cell_types", "columns", "reference_cells", "cell_list"):
                        if k in o:
                            v = o[k]
                            if isinstance(v, dict): return {norm(x) for x in v.keys()}
                            if isinstance(v, list): return {norm(x if not isinstance(x, dict) else x.get("cell") or x.get("name") or "") for x in v} - {""}
                    for v in o.values():
                        r = walk(v, d + 1)
                        if len(r) >= 5: return r
                return set()
            return walk(J)
    except Exception as e:
        print(f"::notice::unreadable {os.path.relpath(path, RM)}: {type(e).__name__}")
    return set()
files = sorted(f for f in glob.glob(f"{RM}/**/*.parquet", recursive=True) + glob.glob(f"{RM}/**/*.json", recursive=True) if "__pycache__" not in f and os.path.getsize(f) < 400_000_000)
T = {os.path.relpath(f, RM): cells_of(f) for f in files}
T = {k: v for k, v in T.items() if len(v) >= 5}
DEC = {k: v for k, v in T.items() if "deconv" in k.lower() or "atlas_reference" in k.lower() or "iam_atlas" in k.lower()}
print("::notice::Deconvolver references: " + json.dumps({k: len(v) for k, v in DEC.items()}))
names = set().union(*DEC.values()) if DEC else set()
print(f"::notice::cells the Deconvolvers can name: {len(names)}")
INSTR = collections.defaultdict(set)
for k, v in T.items():
    if k in DEC: continue
    INSTR[re.sub(r"_part\d+(\.parquet)$", r"_parts\1", k)] |= v   # tables split by cell column into parts are one table
INSTR = dict(INSTR)
groups = collections.defaultdict(dict)
for k, v in INSTR.items():
    g = k.split("/")[0]; groups[g][k] = v
print("::notice::Deconvolver tables: " + json.dumps({k: len(v) for k, v in DEC.items()})[:1500])
print("::notice::instrument tables with cell lists: " + json.dumps({k: len(v) for k, v in INSTR.items()})[:3500])
missing = {}
for c in sorted(names):
    lack = [k for k, v in INSTR.items() if c not in v and len(v & names) >= 0.5 * len(v)]
    if lack: missing[c] = lack
by_table = collections.defaultdict(list)
for c, lack in missing.items():
    for t in lack: by_table[t].append(c)
print(f"::notice::cells the Deconvolvers name that lack at least one table: {len(missing)} of {len(names)}")
for t, cs in sorted(by_table.items(), key=lambda kv: -len(kv[1])):
    print(f"::notice::TABLE {t} ({len(INSTR[t])} cells) lacks {len(cs)}: " + ", ".join(sorted(cs))[:1200])
in_dec = {c: [k for k, v in DEC.items() if c in v] for c in names}
odd = {c: v for c, v in in_dec.items() if len(v) < len(DEC)}
print("::notice::cells only in some Deconvolver versions: " + json.dumps(odd)[:1500])
os.makedirs("xout", exist_ok=True)
json.dump(dict(deconvolver={k: sorted(v) for k, v in DEC.items()}, tables={k: sorted(v) for k, v in INSTR.items()}, missing=missing), open("xout/ref_coverage.json", "w"), indent=1)
