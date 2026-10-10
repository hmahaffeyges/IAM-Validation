set -e; cd iamrepo && git pull -q origin main 2>/dev/null || true; cd Biological_Physics
D=Testing_and_Code/PROC_data/PROC-SWITCH-02; cp ../../results/switch02_results.json "$D/"
cat > "$D/OUTCOME.md" <<'MD'
# OUTCOME — PROC-SWITCH-02: the gauge switch commissioned (CHAIN_COMMISSIONING row B)

**Run 2026-09-21.** 80 synthetic healthy whole-blood patients (WHOLE_BLOOD_ALPHA, no disease, seed 2027, ages 20–80); patients 0–39 the PANEL, 40–79 the TEST, disjoint.

| test | bar | result | verdict |
|---|---|---|---|
| S4a the atlas's own zero | report (expected ≈ −0.015) | **z_atlas = −0.0146** from `lab_zero.compute_lab_zero` on the 40-panel | measured |
| S4b the rest reads 1.00 | median within ±0.010, ≥ 80 % in band | **median A″ 1.0009** [0.985–1.019], **100 % in identity_band_v3** | **PASS** |
| S4c the defect is gone | switched in-band − marker-union in-band ≥ 0.5 | marker-union median **1.125, 0 % in band**; switched 100 % → **+1.00** | **PASS** |
| S4d UNSET refuses | 80/80 | 80/80 reportable = False without a zero | **PASS** |

With PROC-SWITCH-01's S1, S2, S3, S5: **row B COMMISSIONED.** The reported class A is the identity-loci gauge, on mapped β, age-referenced, lab-zeroed, placed in identity_band_v3; the marker-union statistic is `diagnostic_marker_union` and is never the reported A; Stages 5 and 6 carry `pending_recalibration=True` until their rows are run.

**Finding recorded (RECON B5): the atlas is a fifth laboratory.** The G-002 floor (A = 1 at β = 0.7318, 37 reference cells) and the IAMAtlas REBUILD posterior (immune identity-loci mean 0.7373) are different reference sets. A synthetic patient built from atlas means reads 0.985 with zero 0 and 1.001 with the atlas's own panel zero (−0.0146). Real Uppsala healthy blood reads 0.988 mapped. Consistent. Consequence: any synthetic-patient test must zero the synthetic cohort like a laboratory; SWITCH-01 S4 did not and failed as sealed.

**Lesson (RUNBOOK):** every β source is a laboratory — including the atlas, including a simulator. Before reading one absolutely, ask what its zero is.
MD
python3 -c "import hashlib;p='$D/OUTCOME.md';s=open(p).read();open(p,'a').write('\n---\n**SEALED** sha256 \`'+hashlib.sha256(s.encode()).hexdigest()+'\` · 2026-09-21\n')"
# ---- generator docstring: the atlas zero
python3 - <<'PY'
p="CPG_Engine/Synthetic_Patient_Generator/synthetic_patient_generator.py"; s=open(p,encoding="utf-8").read()
tag="ATLAS ZERO (PROC-SWITCH-02, 2026-09-21)"
if tag not in s:
    i=s.find('"""'); j=s.find('"""',i+3)
    s=s[:j]+f"\n\n{tag}: patients built from IAMAtlasREBUILD class means read A = 0.985 on the immune identity loci with lab zero 0,\nbecause the atlas posterior (immune identity-loci mean beta 0.7373) and the G-002 floor (A = 1 at beta 0.7318) are different reference\nsets. The atlas is a fifth laboratory: zero any synthetic cohort from its own 40-patient panel (lab_zero.compute_lab_zero) before\nreading it absolutely; measured z_atlas = -0.0146. Any test that assumes the synthetic zero is 0 will read 0.015 low.\n"+s[j:]
    open(p,"w",encoding="utf-8").write(s); print("generator docstring taught")
PY
# ---- registers
python3 - <<'PY'
import ast,re
p="Physics_of_Methylation/Issue003/data003.py"; s=open(p,encoding="utf-8").read()
s+='''
RECON += [("B5 (PROC-SWITCH-02)", "the atlas posterior's own zero", "assumed 0 (synthetic patients read as if on the floor's scale)", "G-002 floor A=1 at beta 0.7318 (37 reference cells); atlas immune identity-loci mean 0.7373; synthetic whole blood reads 0.985 with zero 0, 1.001 with its own 40-panel zero (-0.0146)", "the atlas is a fifth laboratory; every beta source is zeroed before it is read absolutely", "PROC-SWITCH-01 OUTCOME (S4 FAIL as sealed), PROC-SWITCH-02 OUTCOME"),]
FALSIFICATION += [("PROC-SWITCH-01 S4: 'synthetic healthy patients with lab zero 0 read 1.000 +/- 0.010'", "median 0.985 (97.5% in band); the atlas carries its own constant -0.0146", "FAIL as sealed; re-sealed PROC-SWITCH-02 with the synthetic cohort zeroed from a disjoint 40-panel: median 1.0009, 100% in band; marker-union on the same patients 1.125, 0% in band"),]
SWITCH_PROC = [("question","Does the conductor now REPORT the identity-loci gauge with the three-layer reference, and does the N7 defect disappear?"),
 ("finding","S1 7/7 identity_loci/MAPPED/reportable; S2 7/7 UNSET refuses; S3 4/5 healthy in band; S5 anchors r = 1.00000 (both), canonical A-score test PASS. S4 as sealed FAILED: 40 synthetic healthy read median 0.985 with zero 0 - not the generator's 2,745-locus subset (pure atlas immune reads 0.990 on both), not noise/age/batch (all off: 0.985), but the atlas posterior itself, which sits at beta 0.7373 where the G-002 floor puts A=1 at 0.7318. The analyst's first mixture arithmetic (weights summing to 0.989) was wrong and is recorded."),
 ("correction","PROC-SWITCH-02: the synthetic cohort read as a fifth laboratory. z_atlas = -0.0146 from a 40-panel; 40 disjoint patients read median 1.0009, 100% in identity_band_v3; the marker-union statistic on the same patients 1.125, 0% in band; UNSET refuses 80/80. ROW B COMMISSIONED. In code: cpg_conductor (classes = identity gauge; diagnostic_marker_union; pending_recalibration for Stages 5/6), identity_band_v3.json, Reproduction_Kit/test_gauge_switch.py."),]
'''
ast.parse(s); open(p,"w",encoding="utf-8").write(s)
b=open("Physics_of_Methylation/Issue003/build_gape_issue003.py",encoding="utf-8").read()
line=[l for l in b.split("\n") if l.startswith("    proc(story, 'PROC-HISTORY-01")][0]
if "PROC-SWITCH-02" not in b:
    b=b.replace(line, line+"\n    proc(story, 'PROC-SWITCH-01 -> PROC-SWITCH-02 - the gauge switch, commissioned', 'The reported A is now the identity-loci gauge with the three-layer reference; the atlas turned out to be a fifth laboratory', [(k, dict(D.SWITCH_PROC)[k]) for k in ('question','finding','correction')])",1)
# What's new bullet: row B status
b=b.replace('("Healthy is a band, not a line - and the reference has three layers (s3.5)."','("The gauge switch is done (PROC-SWITCH-02): the reported A is the identity-loci gauge with the three-layer reference; the marker-union statistic is diagnostic only.", "Row B commissioned 2026-09-21. 40 held-out synthetic healthy patients read 1.001 (100% in band) where the old statistic read 1.125 (0%). Finding on the way: the atlas posterior is itself a fifth laboratory with zero -0.0146."),\n     ("Healthy is a band, not a line - and the reference has three layers (s3.5)."',1)
ast.parse(b); open("Physics_of_Methylation/Issue003/build_gape_issue003.py","w",encoding="utf-8").write(b)
so=open("Physics_of_Methylation/Issue003/switching_order.py",encoding="utf-8").read()
so=so.replace('status="KNOWN-DEFECT as wired; correct statistic has a PROVISIONAL band"','status="COMMISSIONED 2026-09-21 (PROC-SWITCH-02): identity-loci gauge + three-layer reference is the reported A; identity_band_v3; marker union diagnostic only"',1)
so=so.replace('("PROC-HISTORY-01",','("PROC-SWITCH-02","row B commissioned: the reported A = H(beta_mean)/H_min on identity loci, mapped, minus c(decade), minus lab zero, in identity_band_v3 (four zeroed labs, n=1,379; pooled p10-p90 0.9724-1.0248). Marker-union statistic retained as diagnostic_marker_union only. Stages 5/6 pending_recalibration. The atlas is a fifth laboratory (z=-0.0146); SWITCH-01 S4 failed as sealed for assuming otherwise."),\n           ("PROC-HISTORY-01",',1)
so=so.replace("conductor still wired to the marker union (labelled gauge_surface='marker_union'); switch only when identity_band_v1 is confirmed on an independent cohort (Phase 1c)","RESOLVED 2026-09-21 (PROC-SWITCH-02): conductor reports the identity-loci gauge; marker union is diagnostic_marker_union",1)
ast.parse(so); open("Physics_of_Methylation/Issue003/switching_order.py","w",encoding="utf-8").write(so); print("registers")
PY
# commissioning table row B
python3 - <<'PY'
p="Physics_of_Methylation/CHAIN_COMMISSIONING.md"; s=open(p,encoding="utf-8").read()
import re
rows=[l for l in s.split("\n") if l.startswith("| B ")]; print("row B before:",rows[0][:200] if rows else "NOT FOUND")
if rows:
    cells=rows[0].split("|"); 
    # find status-like cell (contains KNOWN-DEFECT / NOT WIRED / PROVISIONAL / OPEN)
    for k,c in enumerate(cells):
        if re.search(r"KNOWN-DEFECT|NOT WIRED|PROVISIONAL|OPEN|pending|switch",c,re.I): cells[k]=" **COMMISSIONED** 2026-09-21 (PROC-SWITCH-01 → PROC-SWITCH-02): identity-loci gauge + three-layer reference is the reported A; identity_band_v3; marker union diagnostic only; atlas zero −0.0146 recorded "; break
    s=s.replace(rows[0],"|".join(cells)); open(p,"w",encoding="utf-8").write(s); print("row B after:", "|".join(cells)[:220])
PY
grep -n "^| 5 \|^| 6 " Physics_of_Methylation/CHAIN_COMMISSIONING.md | cut -c1-160
# ---- doors
PARA="
**THE GAUGE SWITCH (PROC-SWITCH-01 → PROC-SWITCH-02, 2026-09-21; row B COMMISSIONED).** \`cpg_conductor.run_full\` now REPORTS the identity-loci gauge: A = H(β̄)/H_min on \`iamatlas_gauge_identity_loci_v1_0.json\`, on mapped β, minus c(decade) (\`reference_age_curve_v1.json\`), minus the laboratory zero (\`lab_zero.py\`), placed in \`identity_band_v3.json\` (four zeroed labs, n = 1,379, pooled p10–p90 0.9724–1.0248). The marker-union statistic is \`diagnostic_marker_union\` — never the reported A. Stages 5 and 6 carry \`pending_recalibration=True\`. Test: \`Reproduction_Kit/test_gauge_switch.py\`. **Finding:** the atlas posterior is a fifth laboratory (z = −0.0146) — SWITCH-01's S4 assumed zero and failed as sealed; every β source, including a simulator, is zeroed before it is read absolutely.
"
for d in HANDOFF.md README.md CPG_Engine/README.md Testing_and_Code/README.md IAM_Atlas/README.md Physics_of_Methylation/Reproduction_Kit/RUNBOOK.md CPG_Engine/CPG_Lessons_Learned_2026-06-29.md "CPG_Engine/README's/README_FOR_FUTURE_AI.md" Physics_of_Methylation/SOP/CPG_Chain_of_Custody_SOP_v2*.md; do grep -q "PROC-SWITCH-02" "$d" || printf "%s" "$PARA" >> "$d"; done; echo "doors taught"
# ---- kit test into kit README and the kill list
python3 - <<'PY'
import re
p="Physics_of_Methylation/Reproduction_Kit/README_FIRST.md"; s=open(p,encoding="utf-8").read()
if "test_gauge_switch.py" not in s: s+="\n| `test_gauge_switch.py` | generated 2026-09-21 | PROC-SWITCH-01 S1–S3 conformance on the kit's cached whole-blood betas: the reported A is the identity-loci gauge; UNSET refuses — run |\n"; open(p,"w",encoding="utf-8").write(s)
PY
cd Physics_of_Methylation/Issue003 && CPG_TRIAL=../../../../trial/CPG_TRIAL_CODE python3 build_gape_issue003.py IAMPerformance_GAPEIssue003_RC1.pdf 2>&1 | grep -v "^INFO" | tail -2
python3 -c "
import pypdfium2 as p; d=p.PdfDocument('IAMPerformance_GAPEIssue003_RC1.pdf'); n=len(d); full=''.join(d[i].get_textpage().get_text_range() for i in range(n))
assert 'PROC-SWITCH-02' in full and 'fifth laboratory' in full; print('pages',n,'; switch proc renders')
pg=[i+1 for i in range(n) if 'the gauge switch, commissioned' in d[i].get_textpage().get_text_range()]; print('proc page',pg); d[pg[0]-1].render(scale=1.3).to_pil().save('../../../../chk_switch.png')"
cd ../Reproduction_Kit && python3 finding_check.py "PROC-SWITCH-02" --registers RECON FALSIFICATION COMMISSIONING SWITCHING --doors --retires "conductor still wired to the marker union (labelled gauge_surface='marker_union'); switch only when" "KNOWN-DEFECT as wired; correct statistic has a PROVISIONAL band"
cp ../Issue003/IAMPerformance_GAPEIssue003_RC1.pdf ../../../../
cd ../../.. && git add -A && git -c user.name="IAMPerformance" -c user.email="iamperformance@users.noreply.github.com" commit -q -m "PROC-SWITCH-02: THE GAUGE SWITCH COMMISSIONED (row B) - cpg_conductor reports the identity-loci gauge with the three-layer reference (mapped, age-referenced, lab-zeroed) in identity_band_v3 (four zeroed labs); marker-union statistic is diagnostic_marker_union only; Stages 5/6 flagged pending_recalibration; kit test test_gauge_switch.py. Held-out synthetic healthy read 1.001 (100% in band) vs old statistic 1.125 (0%). FINDING: the atlas posterior is a fifth laboratory (z=-0.0146) - SWITCH-01 S4 failed as sealed for assuming zero; RECON B5, FALSIFICATION, switching order, commissioning row B, doors, generator docstring" && git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/hmahaffeyges/IAM-Validation.git"; git push -q origin main; git remote set-url origin https://github.com/hmahaffeyges/IAM-Validation.git; echo "pushed $(git rev-parse --short origin/main)"
rm -rf ../push_copies && mkdir -p ../push_copies && git diff --name-only --diff-filter=AM 9f4570c HEAD | grep -v '\.pdf$' | while read f; do mkdir -p "../push_copies/$(dirname "$f")"; cp "$f" "../push_copies/$f"; done; git diff --name-only 9f4570c HEAD > ../push_copies/CHANGED_FILES.txt
cd .. && rm -f push_copies_2026-09-21_9f4570c_to_HEAD.zip && zip -qr push_copies_2026-09-21_9f4570c_to_HEAD.zip push_copies && cp iamrepo/Biological_Physics/Testing_and_Code/PROC_data/PROC-SWITCH-02/OUTCOME.md PROC_SWITCH_02_OUTCOME.md && cp iamrepo/Biological_Physics/Testing_and_Code/PROC_data/PROC-SWITCH-01/OUTCOME.md PROC_SWITCH_01_OUTCOME.md && echo "bundle files: $(wc -l < push_copies/CHANGED_FILES.txt)"