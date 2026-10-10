set -e; cd iamrepo/Biological_Physics
python3 - <<'PY'
import csv,json,re,glob
rows=json.load(open("Testing_and_Code/VAL_INDEX.json"))
# fill CPG 001-014 verdicts from OUTCOME files
for r in rows:
    if r["series"]=="CPG-VAL" and r["status"] in ("see v10 report",""):
        f=glob.glob(r["path"].replace("Biological_Physics/","")+"/*OUTCOME*.md")
        if f:
            t=open(f[0],encoding="utf-8",errors="replace").read()
            m=re.search(r"\*\*Outcome code:\*\*\s*([A-Z0-9_ +]+)",t); s=re.search(r"\*\*Status:\*\*\s*([^\n]{0,60})",t)
            r["status"]=((m.group(1).strip()+"; ") if m else "")+("RESTATED; " if "RESTATED" in t[:200] else "")+(s.group(1).strip() if s else "")
cols=["series","id","title","date","era","cohorts","result","status","sealed","note","family","path"]
with open("Testing_and_Code/VAL_INDEX.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); [w.writerow({c:r.get(c,"") for c in cols}) for r in rows]
json.dump(rows,open("Testing_and_Code/VAL_INDEX.json","w"),indent=1)
import shutil; shutil.copy("Testing_and_Code/VAL_INDEX.csv","Physics_of_Methylation/Reproduction_Kit/results/VAL_INDEX.csv"); shutil.copy("Testing_and_Code/VAL_INDEX.json","Physics_of_Methylation/Reproduction_Kit/results/VAL_INDEX.json")
print("CPG statuses:",[(r["id"][-3:],r["status"][:22]) for r in rows if r["series"]=="CPG-VAL"][:8])
PY
D=Testing_and_Code/PROC_data/PROC-HISTORY-01; mkdir -p "$D"
cat > "$D/OUTCOME.md" <<'MD'
# PROC-HISTORY-01 — the complete validation history, counted from the record (2026-09-21)

**Why.** The author read the "103 indexed pre-registered validations (81 pre-Atlas, 15 post-Atlas, 7 retired)" sentence in Paper 1 and said: "No, we have 128 VAL's plus VAL-049 includes 15 of them in our T Series (T1 → T15). Not to mention the full G series before VAL001 ever started. THEN post IAM-Atlas build we actually tested cpg_val_001-022, not 15." He was right on every count.

**What was wrong, and why.** `VAL_INDEX.csv` (built 2026-09-19 by walking repository paths) had two defects. (1) It keyed on the bare number, so pre-Atlas **VAL-001** and post-Atlas **CPG-VAL-001** collapsed into one row — 22 collisions. (2) It counted only identifiers with a folder at HEAD; 51 pre-Atlas VALs whose records live in the RETIRED evidence report and the author's Zenodo deposit (CC-BY-4.0, 10.5281/zenodo.19633499) have no folder and were not counted. Seven post-Atlas AD folders (CPG-VAL-008..014) had also been mis-filed under `VAL_PreAtlas/` during the 2026-09-19 reorganisation; moved to `VAL_PostAtlas/`.

**The record, counted from `RETIRED_VAL_inventory_report.md` (compiled line-by-line from the April evidence report and README, each outcome cross-checked against `validation_runs/`) and the post-Atlas v10 evidence report, plus repository folders:**

| series | when | count | executed with a recorded outcome | notes |
|---|---|---|---|---|
| **G-series** (H_min calibration, before VAL-001) | Apr 2026 | G-002 (17 chains, 8 methylation floors), G-003b (32 floors), bootstrap of the 32; also G-008, E_A,bio, n_bio ordering | 3 sealed | code restored to `Hmin_Calibration/`; methylation bootstrap run 2026-09-20 (PROC-HMIN-BOOT-01) |
| **VAL-001 → VAL-128** (pre-Atlas) | Apr–May 2026 | **119 distinct identifiers** (numbering gaps 34–36, 78–80, 103–105) in six families: methylation 001–013; five-substrate 014–033; drift cascade 037–046 (35/39 predictions); EDEAR disease cards 047–128 across 12 cards | **107**; 12 not run / queued / dbGaP-gated / excluded at runtime / voided (VAL-102, 4 min) | VAL-050 onward individually pre-registered and SHA-256 sealed before β access |
| **T1 → T15** (VAL-049 cross-population) | Apr 2026 | 15 | 12 (T4, T6, T7 dbGaP-gated) | US / AU / UY / UK / PL / CN-SG populations, frozen panel + frozen H_min |
| **CPG-VAL-001 → CPG-VAL-022** (post-Atlas, IAMAtlas REBUILD) | 29 May – 7 Jun 2026 | **22 slots** | **21** (021 deferred, cohort acquisition) | breast 001–007 (2 RESTATED), AD 008–014 (three cohorts, AD/FTD/PSP-CBD direction discrimination), immune/aging 015–020 & 022 (Hannum full chain 020); L9 null suite N1–N8 per VAL; PREREGs marked RETROSPECTIVE by the author |
| **Mahalanobis HC hull** v0_1 → v0_5 | 6 Jun 2026 | 5 versions | n_HC 601 → 2,523; 8 cohorts; 4 populations incl. **Han Chinese GSE141682 n=42 (first Asian)** | fixed d ≥ 2.0 found invalid in 112-D and replaced by percentile-of-HC; anchor d fell honestly as the hull broadened |
| **L9 N7** chain integrity | 5 Jun 2026 | 1 | synthetic truth through Walther → A-scoring → Mahalanobis | R1 MAE < 1 %; the September PROC-N7-01 rerun is what found the gauge reading the wrong loci |
| **September 2026 procedures** (rebuild from scratch) | 18–21 Sep | PROC-CAL/DECON/ANCHOR/FORMULA/N7/NILC/SEP/CHAIN/STAGE0/WB-IMMUNE/HMIN-BOOT/PANEL-01..03, PHASE 1/1c, band_v2, LAB-ZERO-01/02, CPG-NEW-001 | all sealed before data | this document's own record |

**Corrected sentence for Paper 1 and Issue 003:** *Between April and June 2026 the author ran a G-series calibration, 119 pre-Atlas validations (107 executed; 12 not run for stated reasons) including a 15-cohort cross-population series, and 22 post-Atlas validations (21 executed) on the rebuilt atlas, followed by a five-version healthy-hull expansion to 2,523 controls across four populations and a chain-integrity test with synthetic truth. Records: RETIRED evidence report and inventory; v10 evidence report; Zenodo 10.5281/zenodo.19633499; `Testing_and_Code/VAL_INDEX.csv` (175 rows, unique keys by series).*

**What none of them could see (unchanged):** every one was a within-pipeline comparison or a reading against a control centroid; the pipeline-scale offset and the age curve were found in September by absolute reading. The April caveats tab had predicted the scale offset.

**Lesson (RUNBOOK):** an index built by walking the tree counts folders, not the record. When the record and the tree disagree, the record wins and the index is rebuilt from it with unique keys per series.
MD
python3 -c "import hashlib;p='$D/OUTCOME.md';s=open(p).read();open(p,'a').write('\n---\n**SEALED** sha256 \`'+hashlib.sha256(s.encode()).hexdigest()+'\` · 2026-09-21\n')"
python3 - <<'PY'
import glob,re,ast
# ---- KILL the undercount everywhere
K={"all 103 VAL identifiers":"all 175 validation records (G, VAL-001..128, T1..T15, CPG-VAL-001..022, hull, N7, September PROCs; unique keys by series)",
   "mechanical index of all 103 VAL identifiers in the repo":"index of all 175 validation records, rebuilt from the record 2026-09-21 (PROC-HISTORY-01)",
   "Validation index: all 103 VAL identifiers in the repository with title, date, cohort, stated decision, reco":"Validation index: all 175 validation records - G-series, VAL-001..128, T1..T15, CPG-VAL-001..022, the hull lineage, N7 and the September PROCs - with title, date, cohort, stated decision, reco"}
for p in glob.glob("**/*.md",recursive=True)+glob.glob("Physics_of_Methylation/Issue003/*.py")+glob.glob("Physics_of_Methylation/Reproduction_Kit/*.md"):
    if "RETIRED" in p or "PROC-HISTORY-01" in p: continue
    s=open(p,encoding="utf-8").read(); n=s
    for a,b in K.items(): n=n.replace(a,b)
    if n!=s: open(p,"w",encoding="utf-8").write(n); print("killed in",p)
p="Physics_of_Methylation/Issue003/data003.py"; s=open(p,encoding="utf-8").read()
old=s[s.find("VAL_INDEX_NOTE = ("):]; old=old[:old.find("\n")]
new='VAL_INDEX_NOTE = ("Rebuilt 2026-09-21 from the record, not the tree (PROC-HISTORY-01): 175 rows with unique keys by series - 3 G-series calibrations, 119 pre-Atlas VAL identifiers (107 executed), the 15-cohort T-series of VAL-049, 22 post-Atlas CPG-VALs (21 executed), the five-version Mahalanobis hull lineage, L9 N7, and the September procedures. The 2026-09-19 index had counted 103: it keyed on the bare number (VAL-001 and CPG-VAL-001 collided) and counted only identifiers with a folder at HEAD. Verdicts are recorded from each OUTCOME, not re-verified.")'
s=s.replace(old,new)
s+='''
HISTORY = [  # THE COMPLETE VALIDATION HISTORY (PROC-HISTORY-01, 2026-09-21) - series, when, count, executed, what it was
 ("G-series", "April 2026, before VAL-001", "G-002 (8 methylation floors, 17 chains), G-003b (32 substrate floors), bootstrap of the 32; G-008, E_A,bio, n_bio", "3 sealed", "the H_min calibration; code restored to Hmin_Calibration/; the methylation bootstrap was first run 2026-09-20 (8/8 in CI)"),
 ("VAL-001 -> VAL-128 (pre-Atlas)", "April-May 2026", "119 identifiers in six families", "107 executed; 12 not run / gated / excluded / voided", "methylation 001-013; five substrates 014-033; drift cascade 037-046 (35/39 predictions); EDEAR disease cards 047-128 across 12 cards, SHA-sealed from VAL-050"),
 ("T1 -> T15 (VAL-049)", "April 2026", "15 cohorts, 6 populations", "12 executed; T4/T6/T7 dbGaP-gated", "frozen panel + frozen H_min transferred across US/AU/UY/UK/PL/CN-SG - the first cross-population test"),
 ("CPG-VAL-001 -> 022 (post-Atlas)", "29 May - 7 June 2026", "22 slots", "21 executed; 021 deferred", "breast 001-007 (Mahalanobis d +1.88/+2.10; 2 RESTATED), AD 008-014 (AD up / PSP-CBD down / FTD between), immune-aging 015-020 & 022; L9 nulls N1-N8 per VAL; PREREGs retrospective, so marked"),
 ("Mahalanobis HC hull v0_1 -> v0_5", "6 June 2026", "5 versions, 8 cohorts", "n_HC 601 -> 2,523", "four populations incl. Han Chinese (GSE141682, n=42, first Asian); fixed d>=2.0 shown invalid in 112-D, replaced by percentile-of-HC; anchor d fell honestly as the hull broadened"),
 ("L9 N7 chain integrity", "5 June 2026", "3 synthetic cohorts x 250", "R1 MAE 0.0076-0.0093", "synthetic truth through Walther -> A-scoring -> Mahalanobis; its September rerun (PROC-N7-01) found the gauge on the wrong loci"),
 ("September 2026 procedures", "18-21 September", "PROC-CAL/DECON/ANCHOR/FORMULA/N7/NILC/SEP/CHAIN/STAGE0/WB-IMMUNE/HMIN-BOOT/PANEL-01..03; PHASE 1/1c; band_v2; LAB-ZERO-01/02; CPG-NEW-001", "all sealed before data", "the rebuild from scratch: raw IDATs, absolute reading, the scale map, the age curve, the lab zero - this document"),
]
HISTORY_NOTE = ("What none of the April-June validations could see, and why this document exists: every one was a within-pipeline comparison or a reading against a control centroid. The pipeline-scale offset (predicted in the April caveats tab, measured in September) and the healthy age curve cancel in any difference. They were found by reading single arrays absolutely against the floor. The history is evidence of six months of consistent, sealed testing; it is not a set of results this document re-asserts.")
FALSIFICATION += [("'103 indexed pre-registered validations (81 pre-Atlas, 15 post-Atlas, 7 retired)' (Paper 1 draft and VAL_INDEX, 2026-09-19/20)", "the index keyed on bare numbers (VAL-001 collided with CPG-VAL-001) and counted folders, not the record: 119 pre-Atlas + 15 T-series + 22 post-Atlas + G-series", "CORRECTED 2026-09-21 by the author's reading; PROC-HISTORY-01; index rebuilt, 175 rows"),]
RECON += [("H1 (PROC-HISTORY-01)", "the validation count", "VAL_INDEX 2026-09-19: 103", "RETIRED inventory + v10 report + repo: 3 G + 119 VAL (107 run) + 15 T + 22 CPG-VAL (21 run) + 5 hull + N7 + Sept PROCs", "175-row index, unique keys by series; AD folders 008-014 moved to VAL_PostAtlas", "PROC-HISTORY-01 OUTCOME"),]
'''
open(p,"w",encoding="utf-8").write(s); ast.parse(s)
b=open("Physics_of_Methylation/Issue003/build_gape_issue003.py",encoding="utf-8").read()
# render HISTORY as a table right before the validation index appendix
anchor="    story.append(Paragraph(D.VAL_INDEX_NOTE, sBodySm)); story.append(SP(0.08))"
assert b.count(anchor)==1
new=("    story.append(Paragraph('THE COMPLETE VALIDATION HISTORY, APRIL-SEPTEMBER 2026 (PROC-HISTORY-01)', sLabel))\n"
     "    story.append(tbl([[Pb('series'),Pb('when'),Pb('count'),Pb('executed'),Pb('what it was')]]+[[Pb(a),P(bb),P(c),P(d),P(e)] for a,bb,c,d,e in D.HISTORY],[0.17,0.11,0.20,0.14,0.38], fs=6.2))\n"
     "    story.append(SP(0.04)); story.append(Paragraph(D.HISTORY_NOTE, sBodySm)); story.append(SP(0.10))\n"+anchor)
b=b.replace(anchor,new)
line=[l for l in b.split("\n") if l.startswith("    proc(story, 'PROC-PANEL-01")][0]
b=b.replace(line, line+"\n    proc(story, 'PROC-HISTORY-01 - the validation count, corrected from the record', 'The 2026-09-19 index said 103; the record says 3 G + 119 VAL + 15 T + 22 CPG-VAL + hull + N7', [(k, dict(D.HISTORY_PROC)[k]) for k in ('question','finding','correction')])",1)
b=b.replace("import data003 as D","import data003 as D",1)
open("Physics_of_Methylation/Issue003/build_gape_issue003.py","w",encoding="utf-8").write(b); ast.parse(b)
s=open(p,encoding="utf-8").read()+'''
HISTORY_PROC = [("question","How many validations were run before this document, and how were they counted?"),
 ("finding","The author caught the undercount: 'we have 128 VALs plus VAL-049 includes 15 in our T series, the full G series before VAL-001, and post-Atlas we tested cpg_val_001-022, not 15.' The 2026-09-19 index keyed on the bare number (VAL-001 and CPG-VAL-001 collided, 22 times) and counted only identifiers with a folder at HEAD (51 pre-Atlas records live in the RETIRED report and Zenodo, not in folders). Seven AD folders were also mis-filed under VAL_PreAtlas."),
 ("correction","Index rebuilt from the record with unique keys by series: 175 rows. 3 G; 119 VAL-001..128 (107 executed, 12 not run for stated reasons); T1-T15 (12 executed); 22 CPG-VAL (21 executed); Mahalanobis hull v0_1-v0_5 (n=2,523, four populations incl. Han Chinese n=42); L9 N7; September PROCs. Paper 1 and this document corrected at source. Lesson: an index built by walking the tree counts folders, not the record.")]
'''
open(p,"w",encoding="utf-8").write(s); ast.parse(s)
# switching order + commissioning
so=open("Physics_of_Methylation/Issue003/switching_order.py",encoding="utf-8").read()
so=so.replace('("PROC-PANEL-01 → PROC-PANEL-03"','("PROC-HISTORY-01","the validation count corrected from the record: 3 G-series + 119 pre-Atlas VALs (107 run) + T1-T15 + 22 post-Atlas CPG-VALs (21 run) + hull v0_1-v0_5 + N7; the 2026-09-19 index had said 103 (bare-number key collisions, folders not record). AD folders 008-014 moved to VAL_PostAtlas."),\n           ("PROC-PANEL-01 → PROC-PANEL-03"',1)
ast.parse(so); open("Physics_of_Methylation/Issue003/switching_order.py","w",encoding="utf-8").write(so)
# ---- Paper 1 provenance paragraph
T="Physics_of_Methylation/Papers/Landauer_Metrology_of_the_Methylome.tex"; t=open(T).read()
i=t.find("The instrument tested here was not built for this test."); j=t.find("\n",t.find("Because they were within-pipeline comparisons",i)); j=t.find("\n\n",i) if j<0 else t.find("\n\n",i)
para=t[i:j]
new=("The instrument tested here was not built for this test. Its floors were calibrated in April 2026 by the G-series MCMC described above. Between April and June 2026 the author then ran 119 pre-registered pre-Atlas validations (VAL-001 to VAL-128; 107 executed, 12 not run for stated reasons such as controlled-access data, with every non-execution recorded) in six families --- methylation, five-substrate, a multi-class drift cascade, and per-disease cards --- including a fifteen-cohort cross-population series (VAL-049, T1--T15; twelve executed across US, Australian, Uruguayan, UK, Polish and Singaporean cohorts) in which a frozen panel and frozen floors were transferred without refitting. After the atlas was rebuilt, 22 post-Atlas validations (CPG-VAL-001 to 022; 21 executed) were run on the new chain --- breast pre-diagnostic, Alzheimer's with a three-way direction discrimination against FTD and PSP/CBD, and immune ageing --- each with a per-VAL null suite, followed by a five-version expansion of the healthy Mahalanobis reference from 601 to 2,523 controls across four populations including a first Han Chinese cohort ($n = 42$), and a chain-integrity test in which synthetic patients of declared composition were passed through the production chain and recovered to within 1\\%. Pre-registrations from VAL-050 onward were hash-sealed before data access; the post-Atlas pre-registrations were written retrospectively and are marked as such in the record. We cite this history as evidence of sustained, sealed testing, not as results this paper asserts: every one of those studies was a within-pipeline comparison or a reading against a control centroid, and none could see the scale offset of Section~\\ref{sec:results} or the age curve of Section~5.4, both of which cancel in any difference. The author's April 2026 evidence report \\citep{mahaffey2026zenodo} stated the expectation explicitly --- that the calibration pipeline and other pipelines would differ by a systematic entropy offset and that absolute thresholds would require cross-pipeline validation --- but the offset was not measured until the present work. The complete record (RETIRED evidence report and inventory, post-Atlas v10 evidence report, Zenodo deposit, and a 175-row index with unique keys by series) is in the repository.")
assert i>0 and "103" in para
t=t[:i]+new+t[j:]; assert "103 indexed" not in t; open(T,"w").write(t); print("paper provenance rewritten; old para len",len(para))
# ---- doors
PARA="\n**THE VALIDATION COUNT, CORRECTED (PROC-HISTORY-01, 2026-09-21).** The record, not the tree: 3 G-series calibrations; **119 pre-Atlas VALs (VAL-001..128; 107 executed)** incl. the **T1–T15** cross-population series of VAL-049 (12 executed, 6 populations); **22 post-Atlas CPG-VALs (001..022; 21 executed)**; the Mahalanobis hull v0_1→v0_5 (n=2,523, four populations incl. Han Chinese n=42); L9 N7; the September PROCs. The 2026-09-19 index said 103 — it keyed on bare numbers (VAL-001 collided with CPG-VAL-001) and counted folders. `Testing_and_Code/VAL_INDEX.csv` is rebuilt (175 rows, unique keys by series); AD folders CPG-VAL-008..014 moved to `VAL_PostAtlas/`. Record: `Testing_and_Code/PROC_data/PROC-HISTORY-01/`.\n"
for d in ["HANDOFF.md","README.md","CPG_Engine/README.md","Testing_and_Code/README.md","IAM_Atlas/README.md","Physics_of_Methylation/Reproduction_Kit/RUNBOOK.md","CPG_Engine/CPG_Lessons_Learned_2026-06-29.md","CPG_Engine/README's/README_FOR_FUTURE_AI.md"]+glob.glob("Physics_of_Methylation/SOP/CPG_Chain_of_Custody_SOP_v2*.md"):
    x=open(d,encoding="utf-8").read()
    if "PROC-HISTORY-01" not in x: open(d,"a",encoding="utf-8").write(PARA)
r=open("Testing_and_Code/README.md",encoding="utf-8").read()
r=r.replace("| [`VAL_INDEX.csv`](VAL_INDEX.csv) | **the map** — all 175 validation records (G, VAL-001..128, T1..T15, CPG-VAL-001..022, hull, N7, September PROCs; unique keys by series): title, date, era, cohorts, stated decision, record completeness, path |","| [`VAL_INDEX.csv`](VAL_INDEX.csv) | **the map** — all 175 validation records (G, VAL-001..128, T1..T15, CPG-VAL-001..022, hull, N7, September PROCs; unique keys by series): title, date, era, cohorts, result, status, path |")
open("Testing_and_Code/README.md","w",encoding="utf-8").write(r)
c=open("Physics_of_Methylation/CHAIN_COMMISSIONING.md",encoding="utf-8").read()
if "PROC-HISTORY-01" not in c: open("Physics_of_Methylation/CHAIN_COMMISSIONING.md","a",encoding="utf-8").write("\n**Record note (PROC-HISTORY-01, 2026-09-21):** the validation count behind this table is 3 G + 119 VAL + 15 T + 22 CPG-VAL + hull + N7 + the September PROCs (175 index rows), not 103.\n")
print("doors taught")
PY
cd Physics_of_Methylation/Issue003 && CPG_TRIAL=../../../../trial/CPG_TRIAL_CODE python3 build_gape_issue003.py IAMPerformance_GAPEIssue003_RC1.pdf 2>&1 | grep -v "^INFO" | tail -1
python3 -c "
import pypdfium2 as p,re; d=p.PdfDocument('IAMPerformance_GAPEIssue003_RC1.pdf'); n=len(d); full=''.join(d[i].get_textpage().get_text_range() for i in range(n))
assert 'all 103 VAL' not in full and '103 identifiers' not in full, 'stale 103 still renders'
pg=[i+1 for i in range(n) if 'COMPLETE VALIDATION HISTORY' in d[i].get_textpage().get_text_range()]; assert pg; print('pages',n,'history table on',pg)"
cd ../Reproduction_Kit && python3 finding_check.py "PROC-HISTORY-01" --registers RECON FALSIFICATION COMMISSIONING SWITCHING --doors --retires "all 103 VAL identifiers" "103 identifiers, 1,175 files" "81 pre-Atlas, 15 post-Atlas, 7 retired"
cp ../Issue003/IAMPerformance_GAPEIssue003_RC1.pdf ../Papers/Landauer_Metrology_of_the_Methylome.tex ../../../../
cd ../../.. && git add -A && git -c user.name="IAMPerformance" -c user.email="iamperformance@users.noreply.github.com" commit -q -m "PROC-HISTORY-01: the validation count corrected from the record (author's catch) - 3 G + 119 pre-Atlas VAL (107 run) + T1-T15 + 22 post-Atlas CPG-VAL (21 run) + Mahalanobis hull v0_1-v0_5 (n=2,523, 4 populations incl. Han Chinese) + N7 + Sept PROCs; VAL_INDEX rebuilt with unique keys by series (175 rows; the old index collided VAL-001 with CPG-VAL-001 and counted folders); CPG-VAL-008..014 moved to VAL_PostAtlas; Issue 003 history table + App. V note; Paper 1 provenance paragraph rewritten; RECON H1, FALSIFICATION, switching order, doors" && git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/hmahaffeyges/IAM-Validation.git"; git push -q origin main; git remote set-url origin https://github.com/hmahaffeyges/IAM-Validation.git; echo "pushed $(git rev-parse --short origin/main)"
rm -rf ../push_copies && mkdir -p ../push_copies && git diff --name-only 417d013 HEAD --diff-filter=AM | grep -v '\.pdf$\|per_sample' | while read f; do mkdir -p "../push_copies/$(dirname "$f")"; cp "$f" "../push_copies/$f"; done; git diff --name-only 417d013 HEAD > ../push_copies/CHANGED_FILES.txt
cd .. && rm -f push_copies_2026-09-21_417d013_to_HEAD.zip && zip -qr push_copies_2026-09-21_417d013_to_HEAD.zip push_copies && cp iamrepo/Biological_Physics/Testing_and_Code/PROC_data/PROC-HISTORY-01/OUTCOME.md PROC_HISTORY_01_OUTCOME.md && cp iamrepo/Biological_Physics/Testing_and_Code/VAL_INDEX.csv VAL_INDEX.csv && echo "bundle files: $(wc -l < push_copies/CHANGED_FILES.txt)"