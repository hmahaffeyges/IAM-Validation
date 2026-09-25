#!/usr/bin/env python3
"""Is sub-floor material in healthy blood a stable small value or noise centred on zero?

Runs the commissioned chain (no report) on healthy Uppsala donors and records, per array, every class's
fraction from the class-level solve, the cell-level rollup, and the needlet solver - including everything
below the presence floor, which the report masks. A distribution measurement, not a test against a bar:
the answer decides whether a healthy band is possible for the small classes at all.
"""
import gzip, json, os, re, glob, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "stage1"); sys.path.insert(0, "Biological_Physics/MethylPhys/chain")
from stage_1_idat_calibration import calibrate_idat_to_beta
import cpg_conductor as C

ATLAS = os.path.abspath("Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD.csv")
CMAP = json.load(open("Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD_celltype_to_class.json"))

ages = {}
with gzip.open("geo/GSE87571_series_matrix.txt.gz", "rt", errors="replace") as f:
    acc, chars = None, []
    for line in f:
        if line.startswith("!Sample_geo_accession"): acc = re.findall(r'"([^"]*)"', line)
        elif line.startswith("!Sample_characteristics_ch1"): chars.append(re.findall(r'"([^"]*)"', line))
        elif line.startswith("!series_matrix_table_begin"): break
for i, g in enumerate(acc or []):
    for cl in chars:
        if i < len(cl) and cl[i].lower().startswith("age"):
            try: ages[g] = float(cl[i].split(":")[1])
            except (IndexError, ValueError): pass

pat = re.compile(r"(GSM\d+)_(\d{9,12})_(R0\dC0\d)_(Grn|Red)\.idat\.gz$")
pairs = {}
for p in glob.glob("idats_gse87571/*.idat.gz"):
    m = pat.search(os.path.basename(p))
    if m: pairs.setdefault(m.group(1), {})[m.group(4)] = p
usable = sorted(g for g, v in pairs.items() if len(v) == 2 and g in ages)
sel = usable[:40]
print(f"{len(usable)} Uppsala arrays have both IDATs and a published age; scanning {len(sel)}", flush=True)

out = {}
if os.path.exists("handoff/floor_scan.json"):
    out = json.load(open("handoff/floor_scan.json"))
for n, g in enumerate(sel, 1):
    if g in out: continue
    t0 = time.time()
    try:
        beta, _ = calibrate_idat_to_beta(pairs[g]["Grn"], pairs[g]["Red"])
        s = beta.iloc[:, 0] if hasattr(beta, "columns") else beta
        o = C.run_full(s.dropna().to_dict(), ATLAS,
                       cfg={"age": ages[g], "pipeline": "stage1_noob_450K", "lab_zero": -0.0117,
                            "lab": "GSE87571", "substrate": "whole_blood"})
    except Exception as e:
        out[g] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  {g}: {out[g]['error'][:90]}", flush=True); continue
    cls = (o.get("composition") or {}).get("class") or {}
    cells = (o.get("composition") or {}).get("celltype") or []
    rollup = {}
    for x in cells:
        k = CMAP.get(x.get("cell"))
        if k and x.get("pct"): rollup[k] = round(rollup.get(k, 0.0) + float(x["pct"]), 4)
    so = (o.get("second_opinion") or {}).get("by_class") or {}
    sky = ((o.get("patient_sky") or {}).get("classes") or {})
    out[g] = {"age": ages[g], "secs": round(time.time() - t0, 1),
              "class_level_pct": {k: float(v) for k, v in cls.items()},
              "cell_rollup_pct": rollup,
              "needlet_frac": {k: v.get("nilc") for k, v in so.items()},
              "primary_frac_in_by_class": {k: v.get("walther") for k, v in so.items()},
              "sky": {k: {"f": v.get("fraction"), "status": v.get("status")} for k, v in sky.items()},
              "agreement": (o.get("second_opinion") or {}).get("agreement"),
              "L1_class": (o.get("second_opinion") or {}).get("L1_class")}
    json.dump(out, open("handoff/floor_scan.json", "w"), indent=1)
    print(f"  [{n}/{len(sel)}] {g} age {ages[g]:.0f} in {out[g]['secs']}s | class-level {cls} | "
          f"agreement {out[g]['agreement']}", flush=True)
print("DONE", len([k for k in out if "error" not in out[k]]), "arrays scanned", flush=True)
