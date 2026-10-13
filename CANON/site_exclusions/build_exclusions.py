"""Site exclusion register (2026-10-13): every array probe a reading may NOT use, and why. A list that only removes sites.

Layer 1, not_in_atlas: probes on each chip's manifest with no atlas v2 site (the atlas keeps array CpGs that Loyfer's sequencing read at
>= 10 reads in >= 90 % of samples), plus every non-CpG (ch) and SNP (rs) probe. Computed from the Illumina manifests the chain's
Stage 1 uses (methylprep copies) and the atlas site list (any atlas v2 posterior file: the 'cpg' column).
Layer 2, nuisance masks (planned; manifest test): sites that mark a confounding cell (stroma, tissue macrophages, minor blood cells),
taken from the source atlases' published marker tables, each mask declared by a test before its data is read.

Writes not_in_atlas_<chip>.txt.gz (one probe id per line, sorted) and register.json (counts, input checksums). --check rebuilds from the
same inputs and refuses if anything differs.
Usage: python3 build_exclusions.py MANIFEST_DIR ATLAS_POSTERIOR.parquet [--check]
"""
import os, sys, gzip, json, hashlib, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CHIPS = {"450K": "HumanMethylation450k_15017482_v3.csv.gz", "EPIC_v1": "MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def probes(path):
    hdr = next(i for i, l in enumerate(gzip.open(path, "rt")) if l.startswith("IlmnID"))
    M = pd.read_csv(path, skiprows=hdr, usecols=["IlmnID"], low_memory=False, dtype=str).IlmnID
    return set(M[M.str.match(r"^(cg|ch|rs)", na=False)])
def build(mdir, post):
    A = set(pd.read_parquet(post, columns=["cpg"]).cpg.astype(str))
    reg = {"_rule": __doc__.split("\n\n")[0], "atlas_sites": len(A), "inputs": {"atlas_posterior": {"file": os.path.basename(post), "sha256": sha(post)}}, "chips": {}}
    out = {}
    for chip, f in CHIPS.items():
        p = os.path.join(mdir, f); S = probes(p)
        cg = {s for s in S if s.startswith("cg")}; off = sorted((cg - A) | {s for s in S if not s.startswith("cg")})
        reg["inputs"][chip] = {"file": f, "sha256": sha(p)}
        reg["chips"][chip] = {"probes": len(S), "cpg_probes": len(cg), "cpg_in_atlas": len(cg & A), "cpg_not_in_atlas": len(cg - A),
                              "non_cpg_ch": sum(s.startswith("ch") for s in S), "snp_rs": sum(s.startswith("rs") for s in S), "excluded_total": len(off)}
        out[chip] = "\n".join(off) + "\n"
    reg["chips_not_on_hand"] = {"EPIC_v2": "the EPIC v2 manifest (about 937,000 probes) is not on hand; the file named HumanMethylationEPIC_manifest_v2 reads identical to EPIC v1"}
    return reg, out
if __name__ == "__main__":
    reg, out = build(sys.argv[1], sys.argv[2])
    if "--check" in sys.argv:
        P = [c for c, t in out.items() if gzip.open(os.path.join(HERE, f"not_in_atlas_{c}.txt.gz"), "rt").read() != t]
        old = json.load(open(os.path.join(HERE, "register.json")))
        if old != reg: P.append("register.json")
        print("site_exclusions: " + ("ok" if not P else "REFUSED, differs from a rebuild: " + ", ".join(P))); sys.exit(1 if P else 0)
    for c, t in out.items(): open(os.path.join(HERE, f"not_in_atlas_{c}.txt.gz"), "wb").write(gzip.compress(t.encode(), mtime=0))
    json.dump(reg, open(os.path.join(HERE, "register.json"), "w"), indent=1); print(json.dumps(reg["chips"], indent=1))
