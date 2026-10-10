set -e; W="$(pwd)"; MP="$W/iamrepo/Biological_Physics/MethylPhys"
cd "$MP/kit" && for t in test_percell_physics.py test_tiers.py test_gauge_switch.py test_patient_sky.py; do CPG_KIT_DATA="$W/testdata/10_TEST_DATA" HOME="$W/stage1/mp_home" python3 "$t" > /tmp/$t.log 2>&1 && echo "  $t PASS" || { echo "  $t FAIL:"; grep -vE "Deprecat|^INFO|Scanning|Selecting|class markers|class columns|coverage:|cell-type markers|exclusive markers|twins:" /tmp/$t.log | tail -3 | cut -c1-200; }; done
CPG_KIT_DATA="$W/testdata/10_TEST_DATA" HOME="$W/stage1/mp_home" python3 release_check.py 2>&1 | grep -E "pass,|FAIL " | tail -3
# PLAN: items done tonight
cd "$MP" && python3 - <<'PY'
p="doors/PLAN.md"; s=open(p,encoding="utf-8").read()
note="""- (2026-09-27, done: **build_all.py** - one command regenerates every document that reports the chain (SOP mirror + repoint, OM PDF, cell descriptions from the author's drafts, reviewer manifest, fresh reference report + tab reference, measured REPO_INVENTORY, RUNBOOK/README generated blocks, folder READMEs, GENERATED_MANIFEST.json) and gates on vocab_scan (report guards over SOP, OM, every report tab, the paper), SOP-mirror reconciliation and link_check; guarded_push runs it before propagate. **Item 13 done**: cell_descriptions_v1.json (52 of 115 cells with biology from 27 drafts; drafts stripped to biology, guarded). **SOP checked step by step** against chain_sequence.json: 131 sections bannered LIVE/RECORD/NOT IN CHAIN/NOT BUILT/KIT. **Paper** (Landauer_Metrology_of_the_Methylome.tex) revised to three laboratories + Munich as low-signal input, population layers moved to a record subsection, under the same guard; the report links the PDF once the author commits the Overleaf build.)
"""
if "build_all.py** - one command" not in s:
    i=s.find("\n## Standing (not tasks)"); s=s[:i]+"\n"+note+s[i:]
    s=s.replace("13. **Cells section from the webpage drafts**","13. ~~Cells section from the webpage drafts~~ DONE 2026-09-27 — **Cells section from the webpage drafts**",1)
    open(p,"w",encoding="utf-8").write(s); print("PLAN updated")
PY
cd "$W/iamrepo" && git add -A Biological_Physics && PYTHON="$(which python3)" IAM_WEBDRAFTS="$W/webdrafts" sh Biological_Physics/MethylPhys/chain/guarded_push.sh "One command regenerates every document that reports the chain (author, 2026-09-27: 'the canonicals are only updated via the repo files').

chain/build_all.py: chain sequence -> inventory -> cell descriptions (from the author's webpage drafts, biology only, guarded) -> run index -> SOP mirror (STATUS banner on all 131 sections from a code-derived status table; Part II-A generated from chain_sequence.json) + repoint -> OM two-pass PDF under the chain environment's interpreter -> reviewer manifest -> a FRESH reference report rendered from one repo-shipped GSE87571 array -> report tab reference from THAT render -> measured REPO_INVENTORY.md -> generated blocks in RUNBOOK.md and README.md -> folder READMEs -> GENERATED_MANIFEST.json (sha256 of 178 chain inputs and every output). Gates: kit/vocab_scan.py runs the report's own FORBIDDEN and COHORT guards over the SOP (banner-aware), the OM at source (builder functions + data constants, record sets declared), every tab of the fresh report, and the paper; SOP-mirror reconciliation (every live step named by a LIVE section); link_check. guarded_push.sh runs build_all before propagate: a hand edit to a generated document is overwritten before it can be committed.

Vocabulary: 0 live hits across SOP, OM, report and paper. OM: GAUGE_TIERS read through cpg_tiers.scheme(), never typed; ceiling text removed; reference-layers passage restated as floor + one measured scale with the removed layers as record; sec3/RULES/CHAIN_LINKS/CLINICIAN rewritten. Interface: Chain-tab intro (stages 5/6/8 not in chain), Integrity constants table (no sky scale row), sky captions on the no-panel sigma, donor -> array, verdict -> conclusion. Dead conductor functions stage_5_hull_marker_union and stage_8_matching removed.

Paper: Landauer_Metrology_of_the_Methylome.tex revised to the chain as it runs - three laboratories in the title and methods; GSE125105 reported as low-signal input with the control-probe table and the intake gate that never fired; results table rebuilt from PROC_TARE_01_per_array.parquet (0.992 / 1.016 / 0.961; 94.6% inside NORMAL); the three-layer reference, laboratory zero, age curve and band moved to a 'Record' subsection stating the author's ruling; control-probe model reframed as the recorded on-array route. The report's paper link resolves to the PDF once one is committed beside the source.

PLAN item 13 done; SOP step-by-step check done; build_all recorded." 2>&1 | tail -3; echo "pushed $(git rev-parse --short origin/main)"
rm -rf "$W/push_copies"; mkdir -p "$W/push_copies"; git diff --name-only --diff-filter=AMR HEAD~1 HEAD | grep -vE '\.png$|\.xz$|\.parquet$|\.idat' | while read f; do mkdir -p "$W/push_copies/$(dirname "$f")"; cp "$f" "$W/push_copies/$f" 2>/dev/null || true; done; git diff --name-only HEAD~1 HEAD > "$W/push_copies/CHANGED_FILES.txt"
cd "$W" && rm -f push_copies_2026-09-27d.zip && zip -qr push_copies_2026-09-27d.zip push_copies && echo "copies: $(wc -l < push_copies/CHANGED_FILES.txt) files, $(du -h push_copies_2026-09-27d.zip | cut -f1)"
cp "$MP/papers/Landauer_Metrology_of_the_Methylome.tex" "$MP/doors/PLAN.md" "$MP/manual/MethylPhys_CPG_Operations_Manual.pdf" "$MP/sop/MethylPhys_CPG_SOP.md" . ; cp "$MP/kit/results/reference_report/reference.html" MethylPhys_REFERENCE_report.html
# evidence bundle: the night's downloads and derived results
rm -rf evidence && mkdir -p evidence/idats_panel evidence/results evidence/findings
cp tare01/idats/* evidence/idats_panel/ 2>/dev/null || true
cp results/tare01/PROC_TARE_01_per_array.parquet handoff/tare01_results.json handoff/tare01_diag.json handoff/unmix01_results.json handoff/sky01_results.json handoff/sky01_diag.json handoff/intake01_48.csv handoff/munich_controls.csv evidence/results/ 2>/dev/null || true
cp -r results/sky01/shards evidence/results/sky01_shards 2>/dev/null || true; cp results/sky01/sky_*_atlas_sigma.png evidence/results/ 2>/dev/null || true; cp -r results/intake01 evidence/results/intake01_shards 2>/dev/null || true
cp "$MP/doors/FINDING_GSE125105_LOW_SIGNAL.md" "$MP/doors/PROC_TARE_01_OUTCOME.md" "$MP/doors/PROC_UNMIX_01_OUTCOME.md" "$MP/doors/PROC_SKY_01_OUTCOME.md" "$MP/doors/PROC_INTAKE_01_PREREG.md" evidence/findings/ 2>/dev/null || true
cat > evidence/README.md <<EOF
# Evidence bundle, 2026-09-26/27 night - MethylPhys chain, repo commit $(git -C iamrepo rev-parse --short HEAD)

idats_panel/   raw IDAT pairs fetched from GEO for the 36 detection-panel arrays (12 each: GSE42861 Karolinska, GSE111629 UCLA, GSE125105 Munich). GSE87571 IDATs (732 pairs, ~20 GB) are not in this zip; fetch script: kit/PROC_TARE_01.py.
results/       PROC-TARE-01 per-array record (768 arrays) + bars + diagnostics; PROC-UNMIX-01 bars; PROC-SKY-01 bars, per-array shards (48) and plate; PROC-INTAKE-01 48-array intake run (detection, call rate, signal/background); the Munich control-probe table (3 arrays per laboratory).
findings/      the outcome and finding documents these results back.
Every file is also in the repository at the commit above (kit/results/, doors/). SHA-256 of every file: SHA256SUMS.txt.
EOF
(cd evidence && find . -type f ! -name SHA256SUMS.txt -exec shasum -a 256 {} \; | sort -k2 > SHA256SUMS.txt)
rm -f evidence_2026-09-27_night.zip && zip -qr evidence_2026-09-27_night.zip evidence && echo "evidence: $(find evidence -type f | wc -l) files, $(du -h evidence_2026-09-27_night.zip | cut -f1)"