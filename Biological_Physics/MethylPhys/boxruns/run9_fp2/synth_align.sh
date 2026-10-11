#!/bin/bash
# Runs synth_align.py on the box (needs the Box Run 9 environments and the hg19 bwa-meth index). Writes synth_align_result.csv and the chosen T.
set -uo pipefail
S=/mnt/scratch/run2; W=$S/synth9; mkdir -p $W; cd $W; PYB=/home/ubuntu/env/bin/python; HERE=~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run9_fp2
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/wgbs_tools:$PATH; REF=$S/ref/hg19_lambda_puc19.fa
$PYB $HERE/synth_align.py gen $REF reads.fq truth.json
echo "T,unique_share,misplaced,eps_T,eps_ref,ratio" > synth_align_result.csv
for T in 16 18 20 22 25 28 30 40; do
  bwameth.py --threads 60 --reference $REF -T$T reads.fq 2> align_T$T.log | samtools view -b -o u.bam -
  grep -q "running: bwa mem .* -T$T " align_T$T.log && ! grep -q "c2t [0-9]" align_T$T.log || { echo "T$T: bwa-meth did not pass -T$T through"; exit 3; }   # LESSONS B1b
  samtools sort -@ 16 -o T$T.bam u.bam && samtools index T$T.bam && rm -f u.bam
  $PYB $HERE/synth_align.py place T$T.bam place_T$T.json > /dev/null
  rm -f T$T.pat.gz*; wgbstools bam2pat T$T.bam -o . --genome hg19 -@ 16 > bam2pat_T$T.log 2>&1
  if [ -s T$T.pat.gz ]; then $PYB $HERE/../run2/eps_of.py ~/IAM-Validation/Biological_Physics/MethylPhys/chain eps_T$T.json T$T.pat.gz > /dev/null; else echo '{}' > eps_T$T.json; fi
  $PYB - $T <<'PY' >> synth_align_result.csv
import json, sys
T = sys.argv[1]; p = json.load(open(f"place_T{T}.json")); e = json.load(open(f"eps_T{T}.json")); r = json.load(open("truth.json"))["eps_ref_stageQ_rule"]
eT = list(e.values())[0]["eps"] if e else float("nan")
print(f"{T},{p['unique_share']:.4f},{p['misplaced']:.4f},{eT:.5f},{r:.5f},{eT / r:.4f}")
PY
done
$PYB - <<'PY'
import pandas as pd
d = pd.read_csv("synth_align_result.csv"); ok = d[(d.unique_share >= 0.80) & (d.misplaced <= 0.01) & ((d.ratio - 1).abs() <= 0.05)]
print(d.to_string(index=False)); print("CHOSEN_T", int(ok["T"].max()) if len(ok) else "NONE")
PY
