"""Box Run 2 session 1: does our 1M-read PAT file have the format and CpG indexing of a Loyfer 2023 PAT file?
Usage: python format_check.py OURS.pat.gz LOYFER_HEAD.pat CpG.bed.gz OUT.json"""
import gzip, json, re, sys, collections
ours, loy, cpg, out = sys.argv[1:5]
op = lambda p: gzip.open(p, "rt") if p.endswith(".gz") else open(p)
# CpG-index range of every chromosome, from the wgbstools hg19 dictionary (chr, pos, index)
rng = {}
for l in op(cpg):
    c, _, i = l.rstrip("\n").split("\t")[:3]; i = int(i)
    lo, hi = rng.get(c, (i, i)); rng[c] = (min(lo, i), max(hi, i))
pat = re.compile(r"^[CTN.]+$")
def scan(path, nmax=200000):
    s = collections.Counter(); bad = []; lens = []
    for k, l in enumerate(op(path)):
        if k >= nmax: break
        f = l.rstrip("\n").split("\t"); s["lines"] += 1
        if len(f) != 4: bad.append(("columns", l[:80])); continue
        c, i, p, n = f
        ok = c in rng and i.isdigit() and n.isdigit() and bool(pat.match(p))
        if ok:
            i = int(i); lo, hi = rng[c]; ok = lo <= i and i + len(p) - 1 <= hi
        if not ok: bad.append(("value", l[:80])); continue
        s["ok"] += 1; s["chr:" + c] += 1; lens.append(len(p))
    return {"lines": s["lines"], "ok": s["ok"], "bad": len(bad), "bad_examples": bad[:5],
            "chroms": sorted(k[4:] for k in s if k.startswith("chr:")),
            "mean_cpgs_per_read": round(sum(lens) / max(1, len(lens)), 3)}
res = {"ours": scan(ours), "loyfer": scan(loy), "cpg_total": sum(1 for _ in op(cpg)) if False else None,
       "n_chromosomes_in_dict": len(rng)}
res["pass"] = (res["ours"]["lines"] > 0 and res["ours"]["bad"] == 0 and res["loyfer"]["lines"] > 0 and res["loyfer"]["bad"] == 0)
json.dump(res, open(out, "w"), indent=1); print(json.dumps({k: res[k] for k in ("pass", "n_chromosomes_in_dict")}), res["ours"]["mean_cpgs_per_read"], res["loyfer"]["mean_cpgs_per_read"])
