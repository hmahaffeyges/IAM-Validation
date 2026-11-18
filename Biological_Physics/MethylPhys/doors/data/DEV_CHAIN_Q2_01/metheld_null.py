#!/usr/bin/env python3
"""Null test of Met-A sequencing's held-site score (A_derived_held, stage_metaa_atlas.derived_held), overhaul step 2 (exploration). Runs ON A GITHUB MACHINE.
The score compares a library's disorder at its cell's held identity sites (atlas level >= 0.80) with the disorder of ONE level, 1 - eps0 (0.955), at every
site. Question: on healthy atlas libraries, does it read about 1 (a reading of the sample) or well above 1 for every type (a reading of the 0.80 cutoff)?
Reads the joint run's kept bundles (artifacts of explore runs since 2026-11-17), the top cell of each library (share >= 0.5): A_derived_held, its ordinary
Met-A A, and, as the cutoff's own size, the atlas's mean disorder at those sites is not in the bundle, so the type medians are compared across types."""
import os, json, subprocess, statistics as st, collections, zipfile, io, glob, tempfile
R = os.environ.get("GITHUB_REPOSITORY", "hmahaffeyges/IAM-Validation")
def gh(*a): return subprocess.run(["gh", "api", *a], capture_output=True, text=True, check=True).stdout
runs = []
for pg in range(1, 8):
    J = json.loads(gh(f"/repos/{R}/actions/workflows/explore.yml/runs?per_page=100&page={pg}&created=%3E%3D2026-11-17T08:00:00Z"))["workflow_runs"]
    if not J: break
    runs += [w["id"] for w in J if w.get("conclusion") == "success"]
rows = []; seen = set()
for rid in runs:
    A = json.loads(gh(f"/repos/{R}/actions/runs/{rid}/artifacts"))["artifacts"]
    if not A or A[0]["expired"] or A[0]["size_in_bytes"] > 60_000_000: continue
    d = tempfile.mkdtemp(); subprocess.run(["gh", "run", "download", str(rid), "-R", R, "-D", d], capture_output=True)
    for f in glob.glob(f"{d}/**/joint_GSM*_bundle.json", recursive=True):
        g = os.path.basename(f).split("_")[1]
        if g in seen: continue
        try: B = json.load(open(f))
        except Exception: continue
        m = B.get("met_a_each_cell") or {}; cl = m.get("cells") if isinstance(m, dict) else None
        if not cl: continue
        top = max(cl, key=lambda r: r.get("share") or 0)
        if (top.get("share") or 0) < 0.5 or top.get("A_derived_held") is None: continue
        seen.add(g); rows.append(dict(gsm=g, cell=top["cell"], share=top["share"], A=top.get("A"), A_held=top["A_derived_held"], sites=top.get("derived_held_sites")))
    subprocess.run(["rm", "-rf", d])
by = collections.defaultdict(list)
for r in rows: by[r["cell"]].append(r)
med = sorted(((c, len(v), st.median(x["A_held"] for x in v), st.median(x["A"] for x in v if x["A"] is not None)) for c, v in by.items()), key=lambda t: t[2])
allh = [r["A_held"] for r in rows]
print(f"::notice::MET-A HELD NULL: {len(rows)} healthy libraries, {len(by)} types; A_held median {st.median(allh):.3f} (range {min(allh):.3f}-{max(allh):.3f}); "
      f"share of libraries above 1.2: {sum(v > 1.2 for v in allh) / len(allh):.2f}; ordinary Met-A A median {st.median(r['A'] for r in rows if r['A'] is not None):.3f}")
print("::notice::MET-A HELD by type (type, n, A_held median, A median): " + "; ".join(f"{c} {n} {h:.2f} {a:.2f}" for c, n, h, a in med)[:3800])
os.makedirs("xout", exist_ok=True); json.dump(rows, open("xout/metheld_null.json", "w"), indent=1)
