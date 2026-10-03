set -eo pipefail
PY=~/env/bin/python
OUT=/home/ubuntu/data/dev_selftare_01
tar xzf chain_head.tgz
$PY selftare_intensity.py --chain ./chain --jobs $OUT/jobs.csv --out $OUT --sites subset_sites.csv --workers 48 > intensity.log 2>&1 || true
grep -v "it/s\|s/it" intensity.log | tail -5; grep -c " ok\| cached" intensity.log || true; grep " error" intensity.log | head -3 || true
cp intensity_subset.parquet $OUT/ || true
for f in intensity_subset.parquet; do
  $PY - "$f" <<'PYEOF' || true
import json,sys,subprocess
p=json.load(open('post.json')); f=sys.argv[1]; args=['curl','-sS','-o','/dev/null','-w','%{http_code}']
for k,v in p['fields'].items(): args += ['-F', f'{k}={v.replace("${filename}", f)}' if k=='key' else f'{k}={v}']
args += ['-F', f'file=@{f}', p['url']]; print(f, subprocess.run(args, capture_output=True, text=True).stdout)
PYEOF
done
ls -lh intensity_subset.parquet || true
