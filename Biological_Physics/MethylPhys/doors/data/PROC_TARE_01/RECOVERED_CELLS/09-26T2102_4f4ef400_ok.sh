set -e; W="$(pwd)"; MP="$W/iamrepo/Biological_Physics/MethylPhys"
python3 - <<'PY'
p="iamrepo/Biological_Physics/MethylPhys/doors/PROC_TARE_01_PREREG.md"; s=open(p,encoding="utf-8").read()
old=s[s.index("## Evidence files (named before they exist)"):]
new=("## Evidence files (named before they exist)\n\nIn the kit folder: **PROC_TARE_01.py**. In the kit results folder: **PROC_TARE_01.json**, **PROC_TARE_01_per_array.parquet**. "
     "In the plates folder: **PROC_TARE_01.png**. Linked from the outcome once they exist; not linked here because a link to a file that does not yet exist is a broken link.\n")
s=s.replace(old,new); s=s.replace("(`lab_zero.compute_lab_zero`)","(lab_zero.compute_lab_zero)").replace("`kit/PROC_TARE_01.py`","PROC_TARE_01.py")
open(p,"w",encoding="utf-8").write(s); print("plain names")
PY
cd "$MP/chain" && python3 propagate.py > /tmp/prop.txt 2>&1 && echo "propagate: PASS" || { echo "propagate FAILED:"; grep -E "FAIL" /tmp/prop.txt | cut -c1-300; exit 1; }
cd "$W/iamrepo" && git add -A Biological_Physics && sh Biological_Physics/MethylPhys/chain/guarded_push.sh "PROC-TARE-01 pre-registered before any SNP probe was read: can the array's own known-value probes tare the instrument?

The author's framing: 'a scale tare, before we weigh an object, rather than a calibration by comparison.' The laboratory zero as built - median(A - c(age)) - 1.0 over a healthy panel - assumes the panel's median person reads exactly 1.0 and shifts the laboratory until they do: a population defining zero, the cohort logic this instrument replaces. The array carries a tare of its own: 65 SNP probes (450K) whose beta is 0, 0.5 or 1 by genotype, on every chip. Seven bars fixed in advance, including that the tare must be independent of age and sex (or it is not a tare) and must see the Sentrix-chip term. Decision rule: B1-B4 met -> lab_zero.py retired, the panel of 40 needed only for the Stage 2d noise floor, a laboratory outside NORMAL after taring is FLAGGED and never re-centred.

Only GSE87571 had raw IDATs on disk; the 12-array panels of the other three laboratories were fetched from GEO (36 pairs) so all four can be tared. The first array read before the restart showed T_scale 0.93 and a linear tare moving A away from 1.0 - the form is fixed by the pre-registration and will not be adjusted after seeing that.

ENHANCEMENTS standing to-do: item 0 (tare, physics front page, serial mode) placed ahead of the SOP and OM so the canonicals are written once." 2>&1 | tail -1; echo "pushed $(git rev-parse --short origin/main)"
cp "$MP/doors/PROC_TARE_01_PREREG.md" "$MP/doors/ENHANCEMENTS.md" "$W/"
grep -viE "Deprecat|^INFO|Reading IDATs|it/s|Processing|methylprep" "$W/tare01.log" | tail -3