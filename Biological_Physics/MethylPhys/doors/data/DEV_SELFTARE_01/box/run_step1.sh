set -eo pipefail
PY=~/env/bin/python
D=/home/ubuntu/data/G_chain_tests
OUT=/home/ubuntu/data/dev_selftare_01
mkdir -p $OUT
tar xzf chain_head.tgz
# second-lab isolated neutrophils: from S3 (presigned GET), once
for G in GSE247193 GSE247195; do
  mkdir -p $D/$G/idat
  if [ ! -f $D/$G/RAW_extracted.ok ]; then
    $PY - "$G" <<'EOF'
import json,sys,subprocess
G=sys.argv[1]; u=json.load(open('urls.json'))
for f in (f'{G}_RAW.tar', f'{G}_series_matrix.txt.gz'):
    subprocess.check_call(['curl','-sS','-f','-o',f'/home/ubuntu/data/G_chain_tests/{G}/{f}',u[f]])
EOF
    ls -l $D/$G/
    tar xf $D/$G/${G}_RAW.tar -C $D/$G/idat && touch $D/$G/RAW_extracted.ok
  fi
  ls $D/$G/idat | wc -l
done
$PY - <<'EOF'
import glob, os, pandas as pd
D='/home/ubuntu/data/G_chain_tests'; rows=[]
def pairs(pattern):
    for g in sorted(glob.glob(pattern)):
        r = g.replace('_Grn.idat', '_Red.idat')
        if os.path.exists(r): yield os.path.basename(g).split('_')[0], g, r
for gsm, g, r in pairs(f'{D}/GSE250556/idat/GSM*_Grn.idat.gz'):
    if gsm != 'GSM7981500': rows.append(dict(gsm=gsm, series='GSE250556', grn=g, red=r, specimen='whole blood', group='replicate'))
for G in ('GSE247193', 'GSE247195'):
    for gsm, g, r in pairs(f'{D}/{G}/idat/GSM*_Grn.idat.gz'):
        rows.append(dict(gsm=gsm, series=G, grn=g, red=r, specimen='isolated neutrophils', group='second_lab_neutrophils'))
ct = pd.read_csv('gse110554_celltypes.csv').set_index('gsm')['cell_type'].to_dict()
for gsm, g, r in pairs('/home/ubuntu/data/atlas_sources/blood/GSE110554/idats/GSM*_Grn.idat.gz'):
    c = ct.get(gsm)
    if c in (None, 'MIX'): continue
    rows.append(dict(gsm=gsm, series='GSE110554', grn=g, red=r, specimen=('isolated neutrophils' if c == 'Neu' else 'none'), group='purified_' + c))
df = pd.DataFrame(rows); df.to_csv('jobs.csv', index=False); df.to_csv('/home/ubuntu/data/dev_selftare_01/jobs.csv', index=False); print(df.groupby(['series', 'group']).size())
EOF
$PY selftare_calibrate.py --chain ./chain --jobs jobs.csv --out $OUT --workers 48 > calibrate.log 2>&1 || true
tail -5 calibrate.log; grep -c " ok\| cached" calibrate.log || true; ls $OUT/rec/*.err 2>/dev/null | head || true
$PY selftare_subset.py $OUT "chain/Runtime Matrices/Met_A_Floors"
cp beta_subset.parquet untared_records.csv probe_design.csv $OUT/
# copy off the box
for f in beta_subset.parquet untared_records.csv probe_design.csv jobs.csv calibrate.log; do
  $PY - "$f" <<'EOF' || true
import json,sys,subprocess
p=json.load(open('post.json')); f=sys.argv[1]; args=['curl','-sS','-o','/dev/null','-w','%{http_code}']
for k,v in p['fields'].items(): args += ['-F', f'{k}={v.replace("${filename}", f)}' if k=='key' else f'{k}={v}']
args += ['-F', f'file=@{f}', p['url']]; print(f, subprocess.run(args, capture_output=True, text=True).stdout)
EOF
done
ls -lh beta_subset.parquet untared_records.csv probe_design.csv || true
