set -e; cd iamrepo/Biological_Physics; D=Testing_and_Code/PROC_data/PROC-MATCH-01; mkdir -p "$D"
cat > "$D/PREREG.md" <<'MD'
# PREREG — PROC-MATCH-01: Stage 8 disease matching (CHAIN_COMMISSIONING row 8)

**Sealed 2026-09-21 before any run.** Read before sealing, disclosed: `walther_clinical.stage_8_dual_matching` loads `disease_origin_cells.json` inside `try/except Exception: origin_map = {}` — a missing or corrupt origin file silently degrades the specificity rule (fails OPEN). Its patient departure profile is (A_cell − 1.0) over PRESENT cells on the per-cell separation surface, where PROC-ANCHOR-01 measured healthy per-cell A median ≈ 0.52 with the class H_min floor ≈ 0.84.

**Bars.**
- **M1 origin gate fail-CLOSED.** With `disease_origin_cells.json` absent or unparsable, Stage 8 returns `available=False` with a reason and scores NO candidates; with the file present it runs. Kit test.
- **M2 surface = seal.** The per-cell A entering Stage 8 (conductor Stage A `cells[*].A`) equals mean_i H(β_i)/H_min over the v0_2 markers (the formula that reproduced the sealed anchors at r = 1.00000) on GSM2333901 to 1e-9 for every cell with markers present.
- **M3 matrix integrity.** `disease_cell_signature_matrix_v1_13.csv` parses (81 rows, 53 diseases); every non-metadata column is a key of `iamatlas_115_to_matrix_v0_2_mapping.json` or a known matrix column; SHA-256 recorded.
- **M4 substrate firewall.** A whole-blood patient is scored against zero `plasma_cfDNA` / tissue signatures; a `plasma_cfDNA` patient against zero whole-blood signatures.
- **M5 REPORT (detection rule — no bar).** The 11 cached arrays through Stage 8: number of PRESENT cells entering the departure profile, top route-B cosine and disease, whether any concern fires. **Analyst's sealed prediction:** on healthy whole blood the departure profile admits **< 10 cells** because (A − 1.0) with a below-H_min gate excludes nearly every cell whose healthy A ≈ 0.52 — i.e. Stage 8's departure is referenced to a level (per-cell A ≈ 1 healthy) that the commissioned separation surface does not sit at. If so, the row is NOT commissionable on M1–M4 alone: the departure reference must be re-derived on the separation surface (healthy per-cell level from the four-lab panels) before any disease matching is declared, and that is recorded as the gate.
- Row 8 commissioned only if M1–M4 pass AND M5 shows the departure profile is populated on healthy arrays (≥ 30 present cells) — otherwise row 8 stays OPEN with the re-referencing as its gate.
MD
python3 -c "import hashlib;p='$D/PREREG.md';s=open(p).read();open(p,'a').write('\n---\n**SEALED** sha256 \`'+hashlib.sha256(s.encode()).hexdigest()+'\` · 2026-09-21\n')"; echo sealed
cd CPG_Engine && python3 - <<'PY' 2>&1 | grep -v "Deprecat\|^INFO\|Warning\|Scanning\|class columns\|Selecting\|markers:"
import sys, os, json, pickle, hashlib, csv, math, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"."); import cpg_conductor as C, walther_clinical as W
ATLAS=os.path.abspath("../IAM_Atlas/IAMAtlasREBUILD.csv"); MD="Disease Matrix/DISEASE_MATRIX"
cfg={"disease_matrix_csv":f"{MD}/disease_cell_signature_matrix_v1_13.csv","matrix_mapping_json":f"{MD}/iamatlas_115_to_matrix_v0_2_mapping.json"}
# M3
rows=list(csv.DictReader(open(cfg["disease_matrix_csv"]))); meta={"disease_id","phase","time_range","substrate","disease_severity_class","mechanism","organ_pages_to_link","evidence_anchors"}
cols=[c for c in rows[0] if c not in meta]; mp=json.load(open(cfg["matrix_mapping_json"])); mcols=set(mp.values()) if all(isinstance(v,str) for v in mp.values()) else set(x for v in mp.values() for x in (v if isinstance(v,list) else [v]))
unk=[c for c in cols if c not in mcols]; print(f"M3 matrix: {len(rows)} rows, {len(set(r['disease_id'] for r in rows))} diseases, {len(cols)} cell cols, not in mapping: {len(unk)} {unk[:6]} | sha {hashlib.sha256(open(cfg['disease_matrix_csv'],'rb').read()).hexdigest()[:12]} | substrates {sorted(set(r['substrate'] for r in rows))}")
# M2 + M5
c=pickle.load(open("../../../testdata/10_TEST_DATA/betas_cache.pkl","rb")); asc=C._load_module("iamatlas_a_scoring",str(C._find("iamatlas_a_scoring.py")))
mk=json.load(open(C._find("iamatlas_celltype_markers_v0_2.json"))); HM=mk["H_min_by_class"] if "H_min_by_class" in mk else None; c2c=json.load(open(C._find("IAMAtlasREBUILD_celltype_to_class.json")))
def Hb(b): b=min(max(b,1e-12),1-1e-12); return -b*math.log2(b)-(1-b)*math.log2(1-b)
g="GSM2333901"; b=c[g].dropna(); a=C.stage_a_cells(b.to_dict(),ATLAS)
md=0;n=0
for cell,rec in a["cells"].items():
    loci=[l for l in mk["markers_by_celltype"].get(cell,[]) if l in b.index]
    if not loci or rec.get("A") is None: continue
    cls=c2c.get(cell); hm=(HM or {}).get(cls) or rec.get("H_min")
    if hm is None: continue
    A=sum(Hb(b[l]) for l in loci)/len(loci)/hm; md=max(md,abs(A-rec["A"])); n+=1
print(f"M2 surface: {n} cells compared, max|A_conductor - mean-of-H/H_min| = {md:.2e} -> {'PASS' if n>50 and md<1e-9 else 'FAIL'}")
# M5
s4={"celltype_ascores":{cell:{"A":r.get("A"),"below_floor":r.get("below_floor", (r.get("A") is not None and r.get("class") and r["A"]<(HM or {}).get(r["class"],0))),"celltype_fraction":r.get("fraction")} for cell,r in a["cells"].items()}}
dep=W._build_patient_departure_profile(s4,cfg["matrix_mapping_json"]); print(f"M5 {g}: cells with A {sum(1 for r in s4['celltype_ascores'].values() if r['A'] is not None)}, below_floor {sum(1 for r in s4['celltype_ascores'].values() if r['below_floor'])}, PRESENT entering departure profile: {len(dep) if hasattr(dep,'__len__') else dep}")
As=sorted(r["A"] for r in s4["celltype_ascores"].values() if r["A"] is not None); print(f"   per-cell A on this surface: median {As[len(As)//2]:.3f}, max {As[-1]:.3f}; class H_min range {min(HM.values()):.3f}-{max(HM.values()):.3f}" if HM else "   H_min_by_class absent in markers file")
print("prediction '<10 present cells':", "CORRECT" if (len(dep) if hasattr(dep,'__len__') else 0)<10 else "WRONG")
PY