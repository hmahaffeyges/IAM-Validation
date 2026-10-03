#!/usr/bin/env python3
"""Builds cell_descriptions_v1.json (chain runtime) and manual/cells_data.py (OM) from the author's webpage drafts
(Webpage Drafts.zip, 19 immune + 8 progenitor cell pages) and the atlas itself.

From each draft ONLY the biology is taken - 'What this cell does', 'Where these cells live', other names, lifespan, what
moves the cell's abundance - and every sentence that is product, population or comparison language is dropped by rule:
engine name, customer, subscriber, research/cohort observations, reference ranges, trajectory advice, recommended actions,
cellular age, NLR. What remains is what the cell IS. The physics fields come from the atlas and the chain's runtime files:
class, H_min, identity loci, coverage family, markers. A cell with no draft gets the atlas facts and 'biology page: not yet
written'. Draft cells with no atlas entry are listed as candidates for atlas v2.

The render-time vocabulary guards (FORBIDDEN and COHORT in build_methylphys.py) are run over every description here, so
nothing that would fail the report can enter the file. 2026-09-27, PLAN item 13."""
import re, json, glob, os, sys, collections, html

K = os.path.dirname(os.path.abspath(__file__)); MP = os.path.dirname(K); CH = os.path.join(MP, "chain")
DRAFTS = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.environ.get("IAM_WEBDRAFTS", "")
assert DRAFTS and os.path.isdir(DRAFTS), "usage: build_cell_descriptions.py <folder holding the unzipped Webpage Drafts>"
W = None
sys.path.insert(0, CH); import cpg_conductor as C

# ---- the guards, lifted from the interface so the two cannot drift
src = open(os.path.join(CH, "MethylPhys_Interface/build_methylphys.py"), encoding="utf-8").read()
def _rx(name):
    m = re.search(name + r'\s*=\s*re\.compile\((r?"(?:[^"\\]|\\.)*"(?:\s*r?"(?:[^"\\]|\\.)*")*)\s*,\s*re\.I\)', src, flags=re.S)
    assert m, name
    return re.compile(eval("(" + m.group(1) + ")"), re.I)
FORBIDDEN, COHORT = _rx("FORBIDDEN"), _rx("COHORT")

# ---- draft -> atlas cell name(s). One draft may describe several atlas entries (twins from different sources).
MAP = {
 "neutrophils": ["Neutrophils_reinius", "Neutrophils_EPIC", "Neu", "Neutro", "neutrophil"],
 "eosinophils": ["Eosinophils_reinius", "Eos", "Eosino", "eosinophil"],
 "basophils": ["Baso"],
 "monocytes": ["CD14_monocytes", "Monocytes_EPIC", "Mono", "monocyte"],
 "macrophages": ["Macro", "macrophage", "MP"],
 "kupffer_cells": ["Kup"],
 "microglia": ["Microglia"],
 "dendritic_cells": ["dendritic"],
 "NK_cells": ["CD56_NK-cells", "NK-cells_EPIC", "NK"],
 "CD4_T_cells": ["CD4_T-cells", "CD4T-cells_EPIC", "CD4T"],
 "naive_CD4_T_cells": ["CD4Tnv"],
 "CD8_T_cells": ["CD8_T-cells", "CD8T-cells_EPIC", "CD8T"],
 "naive_CD8_T_cells": ["CD8Tnv"],
 "memory_T_cells": ["CD4Tmem", "CD8Tmem"],
 "regulatory_T_cells": ["Treg"],
 "B_cells": ["CD19_B-cells", "B-cells_EPIC", "B", "Bcell"],
 "naive_B_cells": ["Bnv"],
 "memory_B_cells": ["Bmem"],
 "plasma_cells": ["Plasma"],
 "MPP": ["MPP"], "L-MPP": ["L-MPP"], "MEP": ["MEP"], "NeuIm": ["NeuIm"], "OPC": ["OPC"],
 "megakaryocyte": ["megakaryocyte"],
 "erythroid_downstream_progenitors": ["Erythrocyte_progenitors", "erythroblast", "nRBC"],
 "myeloid_commitment_progenitors": ["CMP", "GMP"],
}
DROP = re.compile(r"ED"r"EAR|customer|subscri|clinician|your report|your reading|research (has|on|observ|literature|cohort)|in research|cohort|reference range|trajectory|recommended|retest|NLR|neutrophil-to-lymphocyte|"
                  r"cellular age|Stage 3|IDOL|Salas|TIM atlas|19-cell|deconvolution|what ED"r"EAR|we read|framework|A-score|flag|"
                  r"healthy (baseline|range|reference)|calibrated for your|wellness|Astro-Genetics|placeholder|\[.*?\]", re.I)

def sentences(txt):
    txt = re.sub(r"\*\*|`", "", txt); txt = re.sub(r"\s+", " ", txt).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", txt) if s.strip()]

def section(md, title):
    m = re.search(r"^## " + re.escape(title) + r"[^\n]*\n(.*?)(?=^## |\Z)", md, flags=re.M | re.S)
    return m.group(1) if m else ""

def clean(block, max_sent=6):
    out = []
    for s in sentences(block):
        if DROP.search(s) or FORBIDDEN.search(s) or COHORT.search(s): continue
        if len(s) < 25: continue
        out.append(s)
    return " ".join(out[:max_sent])

def other_names(md):
    m = re.search(r"Other names[^:]*:\s*([^\n]+)", md)
    if not m: return ""
    names = re.sub(r"\.$", "", re.sub(r"\s+", " ", m.group(1))).strip()
    return ", ".join(x.strip() for x in names.split(",") if x.strip() and not FORBIDDEN.search(x) and not COHORT.search(x))

# ---- atlas facts
c2c = json.load(open(C._find("IAMAtlasREBUILD_celltype_to_class.json"))); c2c = c2c.get("celltype_to_class", c2c)
ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
hmin = {k: v["H_min"] for k, v in ident.items() if isinstance(v, dict) and "H_min" in v}
n_ident = {k: len(v["loci"]) for k, v in ident.items() if isinstance(v, dict) and "loci" in v}
pci_path = C._find("iamatlas_percell_identity_loci_v1_1.json", required=False)
pci = json.load(open(pci_path)) if pci_path else {}
pci_n = {k: (v.get("n_loci") or len(v.get("loci", []))) for k, v in (pci.get("cells") or pci).items() if isinstance(v, dict)}
cov = {}
if os.path.exists(os.path.join(K, "results", "atlas_cell_coverage.csv")):
    import csv
    for r in csv.DictReader(open(os.path.join(K, "results", "atlas_cell_coverage.csv"))):
        cov[r["cell"]] = (int(float(r["n_defined"])), float(r["coverage"]))
mk_path = C._find("iamatlas_celltype_markers_v0_2.json", required=False)
mk = json.load(open(mk_path)) if mk_path else {}
mk_n = {}
for k, v in (mk.get("markers_by_celltype") or mk.get("celltypes") or mk).items():
    if isinstance(v, dict) and isinstance(v.get("markers") or v.get("loci"), list): mk_n[k] = len(v.get("markers") or v.get("loci"))
    elif isinstance(v, list): mk_n[k] = len(v)

# ---- build
entries = {}; candidates = []; used_drafts = set()
for p in sorted(glob.glob(os.path.join(DRAFTS, "**", "Cell Pages *", "*.md"), recursive=True)):
    if "__MACOSX" in p: continue
    stem = os.path.basename(p)[:-3]; md = open(p, encoding="utf-8").read()
    title = (re.search(r"^# (.+)$", md, flags=re.M) or [None, stem])[1].strip()
    what = clean(section(md, "What this cell does"), 6); where = clean(section(md, "Where these cells live"), 3)
    names = other_names(md)
    moves = clean(section(md, "Healthy reference range and what affects it"), 5)   # only the physiology survives the filter
    bio = {"title": title, "what": what, "where": where, "other_names": names, "what_moves_abundance": moves, "source": f"author's draft {stem}.md, 2026-09, biology only"}
    for cell in MAP.get(stem, []):
        if cell in c2c: entries[cell] = dict(bio); used_drafts.add(stem)
        else: candidates.append((stem, cell))
    if stem not in MAP: candidates.append((stem, "?"))
for cell, cl in c2c.items():
    e = entries.setdefault(cell, {"title": cell.replace("_", " "), "what": "", "where": "", "other_names": "", "what_moves_abundance": "", "source": "biology page: not yet written"})
    e.update({"cell": cell, "class": cl, "H_min": hmin.get(cl), "class_identity_loci": n_ident.get(cl), "cell_identity_loci": pci_n.get(cell),
              "markers": mk_n.get(cell), "atlas_loci_defined": cov.get(cell, (None, None))[0], "atlas_coverage": cov.get(cell, (None, None))[1]})
# guard every text field
bad = []
for cell, e in entries.items():
    for k in ("what", "where", "other_names", "what_moves_abundance", "title"):
        t = e.get(k) or ""
        for rx, nm in ((FORBIDDEN, "FORBIDDEN"), (COHORT, "COHORT")):
            hit = rx.search(t)
            if hit: bad.append((cell, k, nm, hit.group(0)))
assert not bad, bad[:8]
out = {"_meta": {"built": "2026-09-27", "from": "Webpage Drafts.zip (author) - biology sections only; atlas facts from IAMAtlasREBUILD_celltype_to_class.json, iamatlas_gauge_identity_loci_v1_0.json, iamatlas_percell_identity_loci_v1_1.json, iamatlas_celltype_markers_v0_2.json",
                  "rule": "what a cell IS, never what a reading means for anyone; the report's vocabulary guards were run over every field",
                  "drafts_used": sorted(used_drafts), "draft_cells_not_in_atlas": [f"{s} -> {c}" for s, c in candidates]},
       "cells": entries}
dst = os.path.join(CH, "Runtime Matrices", "Cell_Descriptions"); os.makedirs(dst, exist_ok=True)
json.dump(out, open(os.path.join(dst, "cell_descriptions_v1.json"), "w"), indent=1)
with_bio = sum(1 for e in entries.values() if e["what"])
print(f"cells: {len(entries)} | with biology from a draft: {with_bio} | drafts used: {len(used_drafts)} of 27 | draft cells not in atlas: {len(candidates)}")
for s, c in candidates: print("   candidate for atlas v2:", s, "->", c)
ex = entries["Neutrophils_reinius"]; print("\nexample:", ex["title"], "|", ex["what"][:300])
