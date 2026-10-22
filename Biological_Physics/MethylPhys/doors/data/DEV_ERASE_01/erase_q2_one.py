"""DEV-ERASE-01 recount (to-do step 14, 2026-10-22): the commissioned counter, chain Stage Q2's count_pat, unchanged, on one sample's .pat.
The held error (isolated unmethylated interior calls over interior calls, on molecules of at least 6 calls and at least 80 % methylated) is
what the ratio uses; the loose share needs hg19 tiles and is empty for a mouse sample. Usage: erase_q2_one.py PAT GSM OUTDIR GENOME   (hg19 tiles for an hg19 sample; none for mm10, so its loose share is empty)"""
import sys, os, gzip, json
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain"))
import stage_q2_selfref as Q2
pat, gsm, out, gen = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with gzip.open(pat, "rt") as f: r = Q2.count_pat(f, Q2.hg19_tiles() if gen == "hg19" else {})
keep = {k: v for k, v in r.items() if isinstance(v, (int, float, str, type(None)))}
json.dump(dict(gsm=gsm, genome=gen, counter="chain Stage Q2 count_pat", **keep), open(os.path.join(out, f"{gsm}_q2.json"), "w"), indent=1); print(json.dumps(keep)[:400])
