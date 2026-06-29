import json, csv, re
ROOT="/home/claude/CPG_CMB_v4"
D=json.load(open("/home/claude/strawman_data.json"))
C2C=json.load(open(f"{ROOT}/IAM_Atlas/IAMAtlasREBUILD_celltype_to_class.json"))
MAP=json.load(open(f"{ROOT}/Disease Matrix/DISEASE_MATRIX/iamatlas_115_to_matrix_v0_2_mapping.json"))
mapping=MAP["mapping"]
lens=json.load(open(f"{ROOT}/Disease Cards : Residual Maps/Immune_Atlas/immune-atlas_card_v2_0.json"))["disease_immune_lens"]["entries"]
mrows=list(csv.DictReader(open(f"{ROOT}/Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_13.csv")))

# index lens + matrix meta by matrix_row_id (1-based row order == disease_rows order)
lens_by_idx={int(e["meta_matrix_row_id"]) if "meta_matrix_row_id" in e else int(e.get("matrix_row_id", e.get("matrix_row_id",0))):e for e in lens}
# the entries carry matrix_row_id inside; fall back to order
for i,e in enumerate(lens):
    e["_idx"]=i
lens_by_order={i:e for i,e in enumerate(lens)}
meta_by_order={i:r for i,r in enumerate(mrows)}

# parse organ_pages_to_link -> origin cell matrix columns
def origin_cells(organ_str):
    cells=set()
    for tok in (organ_str or "").split(","):
        tok=tok.strip()
        if ":" in tok:
            cls,cell=tok.split(":",1)
            cells.add(cell.strip())
    return cells

# attach enrichment to each disease row (same order as matrix rows / lens)
for i,row in enumerate(D["disease_rows"]):
    e=lens_by_order.get(i,{}); m=meta_by_order.get(i,{})
    row["perspective"]=(e.get("immune_perspective","") or "")
    row["mechanism"]=(m.get("mechanism","") or e.get("mechanism_code","") or "")
    row["anchors"]=(m.get("evidence_anchors","") or "")
    row["organ"]=(m.get("organ_pages_to_link","") or "")
    row["origin_cells"]=sorted(origin_cells(row["organ"]))
    row["severity"]=row.get("severity") or m.get("disease_severity_class","")

# completeness: defined-but-unpopulated, split redundant-aggregate vs genuine cell-of-origin
populated=set(c for s in D["sec_cols"].values() for c in s)
AGG_REDUNDANT={"leukocyte_pooled","PBMC","Mye","Tcell","tcell","whole_blood","Leu","Lym",
 "HSPC_pooled","stromal_pooled","brain_pooled"}
unfilled={}  # class -> list of (matrix_col)
seen=set()
for cell,cls in C2C.items():
    col=mapping.get(cell)
    if col and col not in populated and col not in seen:
        # is it a redundant aggregate or a genuine origin cell?
        kind="aggregate" if (col in AGG_REDUNDANT or cell in AGG_REDUNDANT) else "origin_unfilled"
        unfilled.setdefault(cls,[]).append((col,kind)); seen.add(col)
D["completeness"]={cls:sorted(set(v)) for cls,v in unfilled.items()}

json.dump(D,open("/home/claude/strawman_data_v2.json","w"))
# quick verify
ne=sum(1 for r in D["disease_rows"] if r["perspective"])
na=sum(1 for r in D["disease_rows"] if re.search(r"VAL",r["anchors"]))
noc=sum(1 for r in D["disease_rows"] if r["origin_cells"])
print(f"enriched {len(D['disease_rows'])} rows | perspective:{ne} | VAL-anchored:{na} | origin-cells:{noc}")
print("origin cell example (breast row 0):", D["disease_rows"][0]["origin_cells"], "| anchors:", D["disease_rows"][0]["anchors"][:80])
print("completeness classes:", {k:len(v) for k,v in D["completeness"].items()})
gen=[(c) for cls,v in D["completeness"].items() for c,k in v if k=="origin_unfilled"]
print("genuine unfilled cell-of-origin cells:", sorted(gen))
