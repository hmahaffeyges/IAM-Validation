set -e; W="$(pwd)"; MP="$W/iamrepo/Biological_Physics/MethylPhys"; cd "$MP"
python3 - <<'PYEOF'
import re, ast, glob
edits=0
def sub_file(p, pairs, must=False):
    global edits
    s=open(p,encoding="utf-8").read(); o=s
    for a,b in pairs:
        s=s.replace(a,b)
    if s!=o:
        if p.endswith(".py"): ast.parse(s)
        open(p,"w",encoding="utf-8").write(s); edits+=1; print("  edited", p.split("MethylPhys/")[-1])
    elif must: raise AssertionError(p)
COMMON=[("0.95-1.04","0.95-1.05"),("0.95 - 1.04","0.95 - 1.05"),("1.04-1.07","1.05-1.07"),("1.04 to 1.07","1.05 to 1.07"),("[0.95, 1.04)","[0.95, 1.05)"),("A above 1.04","A above 1.05"),("above 1.04","above 1.05"),("1.04 and 1.07","1.05 and 1.07"),("ELEVATED - 1.04","ELEVATED - 1.05")]
for p in ["chain/MethylPhys_Interface/build_methylphys.py","kit/build_percell_reference_identity.py","kit/test_percell_physics.py","doors/PROC_UNMIX_01_PREREG.md","doors/PLAN.md","doors/ENHANCEMENTS.md","sop/MethylPhys_CPG_SOP.md","manual/om_data.py","manual/build_operations_manual.py","kit/test_tiers.py","chain/cpg_tiers.py","doors/CHAIN_COMMISSIONING.md","chain/Runtime Matrices/A_Scoring_Module/test_a_score_canonical.py","kit/PROC_TARE_01.py"]:
    import os
    if os.path.exists(p): sub_file(p, COMMON)
# the tolerance sentence: the 1,379 donors' 0.954-1.041 is an observation about people, on the Instrument tab only
s=open("chain/MethylPhys_Interface/build_methylphys.py",encoding="utf-8").read()
s=s.replace("<td>Tier scale</td><td>NORMAL [0.95, 1.05); ELEVATED to 1.07 (Warburg line); BREACH at 1.10 - the tolerance about A = 1.00, one file, one function</td><td>-</td>",
            "<td>Tier scale</td><td>NORMAL [0.95, 1.05) - symmetric about the fixed point A = 1.00 (v1.5, 2026-09-27; v1.4's 1.04 edge was a cohort's central-95 % bound and is retired); ELEVATED to 1.07 (Warburg line); BREACH at 1.10. One file, one function.</td><td>the 1,379 healthy donors read 0.954-1.041 on the pooled gauge - an observation about people, not the definition</td>",1)
s=s.replace("NORMAL = healthy central 95 % = [0.95, 1.05); 1.07 and 1.10 are the physics lines","NORMAL = [0.95, 1.05) about A = 1.00; 1.07 and 1.10 are the physics lines")
ast.parse(s); open("chain/MethylPhys_Interface/build_methylphys.py","w",encoding="utf-8").write(s)
# OM data: the row that says the tier edge is the healthy central 95 %
t=open("manual/om_data.py",encoding="utf-8").read()
t=t.replace('("tier NORMAL [0.95, 1.05)", "MEASURED convention", "healthy central 95% (PROC-TIER-02)")','("tier NORMAL [0.95, 1.05)", "PHYSICS tolerance", "symmetric about A = 1.00 (tier_breakpoints v1.5, 2026-09-27); v1.4 had set the upper edge from a cohort\'s central 95 % and is retired")')
t=re.sub(r"PROC-TIER-02 then set NORMAL to the healthy central 95% - tier_breakpoints v1\.\d[^\"]*?\.", "PROC-TIER-02 set NORMAL's upper edge from the healthy central 95 % (1.04); v1.5 (2026-09-27) retired that edge as cohort methodology and made NORMAL symmetric, [0.95, 1.05).", t)
ast.parse(t); open("manual/om_data.py","w",encoding="utf-8").write(t)
print(f"{edits} files edited")
PYEOF
echo "--- remaining 1.04 edges in live text ---"; grep -rn "0\.95-1\.04\|1\.04-1\.07\|\[0\.95, 1\.04)\|above 1\.04\|1\.04 to 1\.07\|central 95" --include=*.py --include=*.md --include=*.json --include=*.tex . | grep -v "RETIRED\|CHANGELOG\|_OUTCOME.md\|_PREREG.md\|supersedes_v1_4\|v1.4 had\|v1.4's\|retired that edge\|is retired" | cut -c1-160 | head
echo "--- kit tests ---"; cd kit && for t in test_tiers.py test_percell_physics.py test_gauge_switch.py; do CPG_KIT_DATA="$W/testdata/10_TEST_DATA" HOME="$W/stage1/mp_home" python3 "$t" > /tmp/$t.log 2>&1 && echo "  $t PASS" || { echo "  $t FAIL:"; grep -vE "Deprecat|^INFO|Scanning|Selecting|class markers|class columns|coverage:|cell-type markers|exclusive markers|twins:" /tmp/$t.log | tail -4 | cut -c1-200; }; done