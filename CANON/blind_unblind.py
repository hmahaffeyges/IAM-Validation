"""Blind studies, steps 2 and 3 (written 2026-10-23, before any study's readings land; plan 43.3). The readings are fixed here in code before
any label is seen, so the comparison cannot be chosen after the fact.

  readings GSE BUNDLE_DIR   the chain's pass-1 bundles (run_sample.py --bundle, through series.yml) -> doors/data/BLIND_01/GSE_readings.csv,
                            one row per array, no labels; its SHA-256 goes into CANON/blind_studies.json and the status becomes 'read'.
                            Commit this before step 3.
  unblind GSE               refuses unless GSE_readings.csv is committed (git) and unchanged. Fetches the GEO series matrix, takes each
                            array's title, source and characteristics, sorts arrays into the study's two groups by fixed words (GROUPS
                            below, chosen by the design recorded before reading), and writes GSE_labels.csv and GSE_unblinded.json: for each
                            reading, the median in each group and the AUC (the chance that an array of the first group reads higher than
                            one of the second; 0.5 = no separation). The status becomes 'unblinded'. The publication is opened after this,
                            and its finding and the verdict are entered by hand in the ledger.

The readings (each a number per array, from the bundle the one command wrote):
  top_cell, top_share     the cell the IAM-Deconvolver found most of, and its share
  metaa_cells             Met-A for each cell found, share-weighted mean (met_a_each_cell)
  metaa_allsites          the every-site reading, share-weighted mean (met_a_each_cell_allsites)
  cscore_genome           the whole-genome C-Score (cscore_each_cell.whole_genome.C)
  sphere_distance         the information sphere's Fisher-Rao distance from the sample's own expected healthy self (info_sphere)
  sphere_lit              territories changed on the information sphere
Usage: python3 CANON/blind_unblind.py readings GSE DIR | unblind GSE"""
import csv, glob, gzip, hashlib, io, json, os, re, subprocess, sys, urllib.request
L = "CANON/blind_studies.json"; D = "Biological_Physics/MethylPhys/doors/data/BLIND_01"
READ = ["metaa_cells", "metaa_allsites", "cscore_genome", "sphere_distance", "sphere_lit", "top_share"]
GROUPS = {   # first group, second group: words matched (lower case) in each array's title, source and characteristics; fixed before reading
    "tumor": (("tumor", "tumour", "cancer", "carcinoma", "adenocarcinoma", "malignant", "mesothelioma", "neoplasm"), ("normal", "adjacent", "healthy", "control", "non-tumor", "nontumor", "benign")),
    "gradient": (("cancer", "carcinoma", "tumor", "tumour", "precursor", "hyperplasia", "leukoplakia", "dysplasia", "atypical"), ("normal", "healthy", "control")),
    "blood": (("cancer", "carcinoma", "tumor", "patient", "case", "hcc"), ("control", "healthy", "normal")),
    "smoke": (("current smoker", "smoker", "smoking: yes", "smoking status: 1"), ("never", "non-smoker", "nonsmoker", "smoking: no", "smoking status: 0")),
}
def ledger(): return json.load(open(L, encoding="utf-8"))
def save(J): json.dump(J, open(L, "w", encoding="utf-8"), indent=1, ensure_ascii=False); open(L, "a").write("\n")
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def wmean(rows, key="A"):
    v = [(r.get("share") or 0, r.get(key)) for r in rows if r.get(key) is not None and r.get("share")]
    w = sum(s for s, _ in v); return round(sum(s * a for s, a in v) / w, 6) if w else None
def row_of(p):
    b = json.load(open(p)); g = os.path.basename(p)[:-5]
    cs = (b.get("cell_shares") or {}).get("cells") or []; top = max(cs, key=lambda r: r.get("share") or 0) if cs else {}
    S = b.get("info_sphere") or {}
    return dict(gsm=g, intake=(b.get("intake") or {}).get("stage0_verdict"), refusal=(str(b.get("refusal"))[:120] if b.get("refusal") else ""),
                top_cell=top.get("cell"), top_share=top.get("share"), metaa_cells=wmean((b.get("met_a_each_cell") or {}).get("cells") or []),
                metaa_allsites=wmean((b.get("met_a_each_cell_allsites") or {}).get("cells") or []),
                cscore_genome=((b.get("cscore_each_cell") or {}).get("whole_genome") or {}).get("C"),
                sphere_distance=S.get("fisher_rao_rms_rad"), sphere_lit=S.get("territories_changed"))
def study(J, gse): return next(s for s in J["studies"] if s["gse"] == gse)
def auc(x, y):
    x = [a for a in x if a is not None]; y = [a for a in y if a is not None]
    if not x or not y: return None
    return round(sum((a > b) + 0.5 * (a == b) for a in x for b in y) / (len(x) * len(y)), 3)
def median(v):
    v = sorted(a for a in v if a is not None); n = len(v)
    return None if not n else (v[n // 2] if n % 2 else round((v[n // 2 - 1] + v[n // 2]) / 2, 6))

if sys.argv[1] == "readings":
    gse, d = sys.argv[2], sys.argv[3]; J = ledger(); s = study(J, gse)
    want = {r["gsm"] for r in csv.DictReader(open(s["samples"]))}
    rows = [row_of(p) for p in sorted(glob.glob(os.path.join(d, "**", "GSM*.json"), recursive=True)) if not p.endswith(("_planted.json", "_values.json"))]
    rows = [r for r in rows if r["gsm"] in want]; got = {r["gsm"] for r in rows}
    rows += [dict(gsm=g, intake="", refusal="no bundle (not read)") for g in sorted(want - got)]
    p = os.path.join(D, gse + "_readings.csv"); k = ["gsm", "intake", "refusal", "top_cell", "top_share"] + READ[:-1]
    w = csv.DictWriter(open(p, "w", newline=""), fieldnames=k, extrasaction="ignore"); w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["gsm"]))
    s.update(status="read", readings=p, readings_sha256=sha(p), arrays_read=len(got)); save(J); print(gse, "read", len(got), "of", len(want), "->", p)

elif sys.argv[1] == "unblind":
    gse = sys.argv[2]; J = ledger(); s = study(J, gse); p = s.get("readings")
    assert p and s["status"] == "read", "step 2 first"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", p]).returncode == 0 and subprocess.run(["git", "ls-files", "--error-unmatch", p], capture_output=True).returncode == 0, \
        "the readings must be committed, unchanged, before unblinding"
    assert sha(p) == s["readings_sha256"], "readings changed since step 2"
    kind = s["kind"]   # recorded in the ledger before reading
    u = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/matrix/"
    names = sorted(set(re.findall(rf"{gse}[^\"<>]*_series_matrix\.txt\.gz", urllib.request.urlopen(u, timeout=120).read().decode())))
    lab = {}
    for n in names:
        T = gzip.decompress(urllib.request.urlopen(u + n, timeout=300).read()).decode("utf-8", "replace").split("\n")
        F = {}
        for l in T:
            if l.startswith("!Sample_"):
                k, *v = l.rstrip("\n").split("\t"); F.setdefault(k, []).append([x.strip('"') for x in v])
        acc = F.get("!Sample_geo_accession", [[]])[0]
        for i, g in enumerate(acc):
            lab[g] = " | ".join(x[i] for k in ("!Sample_title", "!Sample_source_name_ch1", "!Sample_characteristics_ch1") for x in F.get(k, []) if i < len(x))
    first, second = GROUPS[kind]
    def grp(t):
        t = t.lower(); a = any(w in t for w in first); b = any(w in t for w in second)
        return "first" if a and not b else ("second" if b and not a else ("both words" if a and b else "neither"))
    R = list(csv.DictReader(open(p)))
    out = [dict(gsm=r["gsm"], group=grp(lab.get(r["gsm"], "")), label=lab.get(r["gsm"], "")[:300]) for r in R]
    w = csv.DictWriter(open(os.path.join(D, gse + "_labels.csv"), "w", newline=""), fieldnames=["gsm", "group", "label"]); w.writeheader(); w.writerows(out)
    G = {o["gsm"]: o["group"] for o in out}; num = lambda v: float(v) if v not in ("", None) else None
    res = dict(gse=gse, kind=kind, first_words=first, second_words=second, counts={k: sum(1 for v in G.values() if v == k) for k in ("first", "second", "both words", "neither")}, readings={})
    for k in READ:
        x = [num(r.get(k)) for r in R if G[r["gsm"]] == "first"]; y = [num(r.get(k)) for r in R if G[r["gsm"]] == "second"]
        res["readings"][k] = dict(median_first=median(x), median_second=median(y), auc_first_higher=auc(x, y))
    tc = {}
    for r in R: tc.setdefault(G[r["gsm"]], {}).setdefault(r.get("top_cell") or "none", 0); tc[G[r["gsm"]]][r.get("top_cell") or "none"] += 1
    res["top_cell_by_group"] = tc
    json.dump(res, open(os.path.join(D, gse + "_unblinded.json"), "w"), indent=1)
    s.update(status="unblinded", unblinded=os.path.join(D, gse + "_unblinded.json")); save(J); print(json.dumps(res, indent=1)[:3000])
