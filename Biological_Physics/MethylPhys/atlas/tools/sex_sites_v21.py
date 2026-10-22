"""The atlas sites on the X and Y chromosomes, frozen for the chain (2026-10-21; author: "we MUST show we did account for it").
Reads the chromosome of every IAM-Atlas v2.1 site from the two Illumina manifests Stage 1 is pinned to (their SHA-256 must match
chain/stage_1_idat_calibration.MANIFEST_SHA256, or this refuses), and writes chain/Runtime Matrices/Atlas_V21/sex_chromosome_sites_v21.json:
the ids on the X, the ids on the Y, the manifests' checksums and the count of atlas sites. The globe uses it to count every change with and
without the X and Y; the Decoder's sex section reads its counts. Usage: python3 sex_sites_v21.py MANIFEST_DIR"""
import os, sys, io, gzip, json, hashlib, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..")); CH = os.path.join(MP, "chain"); sys.path.insert(0, CH)
from stage_1_idat_calibration import MANIFEST_SHA256
NAMES = ("HumanMethylation450k_15017482_v3.csv.gz", "MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz")
def chrom(p):
    t = gzip.open(p, "rt").read().split("\n"); i = next(i for i, l in enumerate(t) if l.startswith("IlmnID"))
    j = next((k for k in range(i, len(t)) if t[k].startswith("[Controls]")), len(t))
    return pd.read_csv(io.StringIO("\n".join(t[i:j])), usecols=["IlmnID", "CHR"], low_memory=False).set_index("IlmnID").CHR.astype(str).str.replace("chr", "")
def main(d):
    sha = {}
    for n in NAMES:
        h = hashlib.sha256(open(os.path.join(d, n), "rb").read()).hexdigest()
        assert h == MANIFEST_SHA256[n], f"{n}: SHA-256 {h[:16]}... is not the manifest Stage 1 is pinned to"; sha[n] = h
    C = pd.concat([chrom(os.path.join(d, n)) for n in reversed(NAMES)]); C = C[~C.index.duplicated()]
    ids = pq.read_table(os.path.join(MP, "atlas", "v2.1", "sites.parquet")).column("cpg").to_pylist(); c = C.reindex(ids)
    X = sorted(c.index[c == "X"]); Y = sorted(c.index[c == "Y"])
    # 2026-10-21: every probe of the two pinned array manifests too, so instrument site lists that are not atlas sites (447 of them) are placed
    XA = sorted(C.index[C == "X"]); YA = sorted(C.index[C == "Y"])
    out = dict(about=__doc__.split("\n")[0], atlas_sites=len(ids), x_sites=len(X), y_sites=len(Y), sites_without_chromosome=int(c.isna().sum()),
               manifests_sha256=sha, array_probes=int(len(C)), X_array=XA, Y_array=YA, source="atlas/tools/sex_sites_v21.py", X=X, Y=Y)
    p = os.path.join(CH, "Runtime Matrices", "Atlas_V21", "sex_chromosome_sites_v21.json"); json.dump(out, open(p, "w"), indent=0); open(p, "a").write("\n")
    print({k: v for k, v in out.items() if k not in ("X", "Y", "manifests_sha256")})
if __name__ == "__main__": main(sys.argv[1])
