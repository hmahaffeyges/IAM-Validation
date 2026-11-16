#!/usr/bin/env python3
"""A8H step 1 (register A8, candidate iii: the meter's 5hmC blind spot; 2026-11-16). Reads ONE GSE70519 TAB-seq allc file (Hohos et al. 2016,
sorted blood type; TAB-seq: a methylated call is 5hmC) ON A GITHUB MACHINE by streaming; nothing is kept but a few numbers. Usage:
python3 a8h_tabseq_read.py GSM1808328. Reports, on CG context: pooled 5hmC share (mc/cov); among sites covered >= 5x: share with 5hmC rate
>= 0.1 and >= 0.25; coverage-weighted mean rate at sites with rate > 0 (where 5hmC sits). A published input (not a chain reading, not an IAM-A
reading of a sample), recorded for the cell simulator's 5hmC lever. Expected (Hohos Table 2): pooled share CD4 T 3.67 %, CD8 T 1.91 %, etc."""
import sys, os, gzip, json, urllib.request, time
FILES = {"GSM1808328": "CD4_T_cell", "GSM1808329": "CD8_T_cell", "GSM1808330": "CD14_monocyte", "GSM1808331": "CD16_neutrophil",
         "GSM1808332": "CD19_Bcell", "GSM1808333": "CD56_NKcell", "GSM1808334": "Sgilec8_eosinophil"}
g = sys.argv[1]; name = FILES[g]
url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1808nnn/{g}/suppl/{g}_Schmitz_allc_{name}.tsv.gz"
for attempt in range(4):
    try:
        resp = urllib.request.urlopen(url, timeout=600); break
    except Exception as e:
        print(f"::warning::attempt {attempt + 1}: {e}"); time.sleep(60 * (attempt + 1))
else:
    raise SystemExit("download refused")
mc = cov = n5 = n5_10 = n5_25 = pos_mc = pos_cov = 0; first = None
with gzip.GzipFile(fileobj=resp) as fh:
    for raw in fh:
        p = raw.decode("ascii", "replace").rstrip("\n").split("\t")
        if len(p) < 6: continue
        if first is None: first = p[:7]
        if not p[3].startswith("CG"): continue
        try: m, c = int(p[4]), int(p[5])
        except ValueError: continue
        mc += m; cov += c
        if c >= 5:
            n5 += 1; r = m / c; n5_10 += r >= 0.10; n5_25 += r >= 0.25
        if m > 0: pos_mc += m; pos_cov += c
out = dict(gsm=g, cell=name, first_row=first, cg_calls=cov, hmC_share_pooled=mc / max(cov, 1), sites_cov5=n5,
           share_sites_rate_ge_0p10=n5_10 / max(n5, 1), share_sites_rate_ge_0p25=n5_25 / max(n5, 1), mean_rate_where_present=pos_mc / max(pos_cov, 1))
os.makedirs("xout", exist_ok=True); json.dump(out, open(f"xout/a8h_{g}.json", "w"), indent=1)
print("::notice::" + json.dumps(out))
