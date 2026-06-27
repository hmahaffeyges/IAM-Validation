import json, csv, math, html
ROOT="/home/claude/CPG_CMB_v4"
MATRIX=f"{ROOT}/Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_8.csv"
C2C=json.load(open(f"{ROOT}/IAM_Atlas/IAMAtlasREBUILD_celltype_to_class.json"))
MAP=json.load(open(f"{ROOT}/Disease Matrix/DISEASE_MATRIX/iamatlas_115_to_matrix_v0_2_mapping.json"))["mapping"]

# matrix_column -> architecture class (compose atlas_cell->col and atlas_cell->class)
col2class={}
for atlas_cell, col in MAP.items():
    if col and atlas_cell in C2C:
        col2class.setdefault(col, C2C[atlas_cell])

# myeloid subset of immune (for the lymphoid/myeloid display split)
MYELOID={"neutrophils","eosinophils","basophils","granulocytes_pooled","monocytes",
 "macrophages_peripheral","microglia","kupffer_cells","pulmonary_macrophages",
 "oral_macrophages","dendritic_cells","IC_immune_pooled"}
BBB={"cortical_neurons","neurons_pooled","glia","astrocytes","brain_pooled","brain_astrocytes",
 "oligodendrocytes","OPC","microglia","NeuMa","NeuIm"}

def section_of(col):
    cls=col2class.get(col)
    if cls=="immune": return "myeloid" if col in MYELOID else "lymphoid"
    if cls in ("stem_adult","stem_pluri"): return "stem"
    if cls: return cls
    # name-based fallback for unmapped cols
    n=col.lower()
    if any(t in n for t in ("neuron","glia","oligo","astro","cardio","atrium","muscle_skeletal","kera","epi_upper","brain")): return "terminal"
    if any(t in n for t in ("fibro","endothel","adipo","smooth_muscle","stellate","stromal","BMSC","osteo","placenta","mammary_stromal")): return "stromal"
    if any(t in n for t in ("cycling","undiff","basal_cycling","keratinocytes","ductal_cycling")): return "cycling"
    if any(t in n for t in ("MPP","CMP","GMP","MEP","erythro","nRBC","megakaryo","HSPC","HSC","L_MPP")): return "progenitor"
    if any(t in n for t in ("epithelial","secretory","hepato","duct","acinar","beta_cell","LE","BE","gastric","thyroid","salivary","colon","intestine","rectal","lung","bladder","cervix","prostate","skin","eso","pancreatic")): return "secretory"
    return "secretory"

SECTIONS=["lymphoid","myeloid","progenitor","cycling","secretory","terminal","stromal","stem"]
SECLABEL={"lymphoid":"Lymphoid","myeloid":"Myeloid","progenitor":"Progenitor","cycling":"Cycling",
 "secretory":"Secretory","terminal":"Terminal","stromal":"Stromal","stem":"Stem"}

rows=list(csv.DictReader(open(MATRIX)))
META={"disease_id","phase","time_range","substrate","disease_severity_class","mechanism","organ_pages_to_link","evidence_anchors"}
cell_cols=[c for c in rows[0].keys() if c not in META]

def parse_val(v):
    """return (signed d float or None, qualitative arrow or None)"""
    if v is None: return None,None
    v=str(v).strip()
    if not v: return None,None
    if v.startswith("\u2191"): return None,"up"
    if v.startswith("\u2193"): return None,"down"
    try:
        if "/" in v:
            a,b=v.split("/"); return (float(a)+float(b))/2.0,None
        return float(v),None
    except ValueError:
        return None,None

# which cells are populated (>=1 disease has a numeric or arrow value)
populated=set()
disease_rows=[]
for r in rows:
    cells={}
    for c in cell_cols:
        d,arr=parse_val(r.get(c,""))
        if d is not None or arr is not None:
            cells[c]={"d":d,"arr":arr}; populated.add(c)
    disease_rows.append({"disease":r["disease_id"],"phase":r.get("phase",""),
        "time_range":r.get("time_range",""),"substrate":r.get("substrate",""),
        "severity":r.get("disease_severity_class",""),"cells":cells})

# columns per section (only populated), keep matrix order
sec_cols={s:[] for s in SECTIONS}
for c in cell_cols:
    if c in populated:
        sec_cols[section_of(c)].append(c)
# drop empty sections (e.g. stem if no disease uses it) but keep cycling always
sections_used=[s for s in SECTIONS if sec_cols[s] or s=="cycling"]

print("=== STRAWMAN DATA SUMMARY ===")
print("disease rows (disease x phase):", len(disease_rows))
print("distinct diseases:", len(set(r["disease"] for r in disease_rows)))
print("populated cells:", len(populated))
for s in SECTIONS:
    print(f"  {s}: {len(sec_cols[s])} cells {'(KEPT)' if (sec_cols[s] or s=='cycling') else '(EMPTY-dropped)'}")
# stash for the renderer
json.dump({"disease_rows":disease_rows,"sec_cols":sec_cols,"sections_used":sections_used,
           "col2class":col2class},open("/home/claude/strawman_data.json","w"))
print("\nsaved /home/claude/strawman_data.json")
