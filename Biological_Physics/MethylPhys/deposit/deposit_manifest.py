"""The deposit manifest: every file a test recalls, so the whole record can be re-run from a Zenodo deposit without any other source.
One row per file: kind (chain, atlas, input, test file), the tests that recall it, where it is now (repository path, the Google bucket key,
or only a public archive's address), size and sha256 where the file is in hand. Inputs that live only in a public archive are listed so
they can be staged in the bucket before the deposit. Nothing is uploaded by this script. Usage: python3 deposit_manifest.py OUT.csv OUT.txt"""
import os, sys, glob, hashlib, pandas as pd
MP = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RP = os.path.abspath(os.path.join(MP, "..", ".."))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
rows = []
def repo(p, kind, tests):
    q = os.path.relpath(p, RP); rows.append(dict(file=q, kind=kind, tests=tests, where="repository", location=q, size=os.path.getsize(p), sha256=sha(p)))
# the chain and both rulers, as committed
for p in sorted(glob.glob(os.path.join(MP, "chain", "**", "*.py"), recursive=True)) + [os.path.join(MP, "chain", "requirements.txt")]:
    if "__pycache__" not in p: repo(p, "chain", "every test")
for p in sorted(glob.glob(os.path.join(MP, "atlas", "v2.1", "**", "*"), recursive=True)):
    if os.path.isfile(p): repo(p, "atlas: IAM-Atlas v2.1", "every array test; the bridge")
for p in sorted(glob.glob(os.path.join(MP, "chain", "Runtime Matrices", "**", "*"), recursive=True)):
    if os.path.isfile(p): repo(p, "atlas: runtime tables" + (" (IAM Sequencing Atlas)" if "Seq_Atlas" in p else ""), "every test reading against them")
# inputs read by the sequencing tests, from the committed sample lists
SEQ = os.path.join(MP, "doors", "data")
LISTS = [("DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv", "the IAM Sequencing Atlas build", "gsm", "file"),
         ("DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/healthy_fullchain5_samples.csv", "healthy pass 5; Group-A; transfer function; globe", "gsm", None),
         ("DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/selfree_samples.csv", "selection-free healthy test; globe outside", "gsm", "url"),
         ("DEV_ATLAS_V21_01/seq_atlas/outside_data/blueprint_healthy_test_samples.csv", "healthy samples from another laboratory", "gsm", "call_bw"),
         ("DEV_ALL_INFO_01/full_depth/hopkins_fullchain_samples.csv", "the Johns Hopkins study", "gsm", None)]
INP = pd.read_csv(os.path.join(SEQ, "DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv")); INP["gsm"] = INP.gsm.str.strip(); URL = INP.set_index("gsm").file.to_dict()
SHA = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs_sha256_*.csv"))]) if glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs_sha256_*.csv")) else pd.DataFrame(columns=["gsm", "sha256"])
CACHE = {f"seq_atlas/inputs/{g}.beta": h for g, h in zip(SHA.gsm, SHA.sha256)}   # inputs staged in the bucket, with the checksum taken as they were staged
for rel, tests, idc, urlc in LISTS:
    S = pd.read_csv(os.path.join(SEQ, rel))
    for r in S.itertuples():
        g = str(getattr(r, idc)).strip(); urls = [getattr(r, urlc)] if urlc else [URL.get(g)]
        if urlc == "call_bw": urls = [r.call_bw, r.cov_bw]
        for u in urls:
            key = f"seq_atlas/inputs/{g}.beta" if (u and str(u).endswith(".beta")) else None
            if rel.startswith("DEV_ALL_INFO_01"): where, loc = "public archive (reads; aligned pieces are GitHub artifacts until they expire)", "SRA, by the sample list"
            elif key and key in CACHE: where, loc = "Google bucket", key; rows.append(dict(file=f"{g} {os.path.basename(str(u))}", kind="input", tests=tests, where=where, location=loc, size=56434896, sha256=CACHE[key])); continue
            else: where, loc = "public archive only (to stage)", str(u)
            rows.append(dict(file=f"{g} {os.path.basename(str(u)) if u else ''}".strip(), kind="input", tests=tests, where=where, location=loc, size=None, sha256=None))
# every test's own committed files (sample lists, scripts, outputs) under the door data folders the sequencing work uses
for d in ("DEV_METAA_ATLAS_CURVE_01", "DEV_ATLAS_V21_01", "DEV_TERRITORIES_01", "DEV_ALL_INFO_01"):
    for p in sorted(glob.glob(os.path.join(SEQ, d, "**", "*"), recursive=True)):
        if os.path.isfile(p) and "__pycache__" not in p: repo(p, "test file", d)
D = pd.DataFrame(rows).drop_duplicates(["file", "where", "location"]); D.to_csv(sys.argv[1], index=False)
L = [f"deposit manifest: {len(D)} files"] + [f"  {k}: {len(g)} files, {g['size'].sum() / 1e6:.1f} MB in hand" for k, g in D.groupby("kind")] + \
    [f"  where: " + "; ".join(f"{k} {v}" for k, v in D["where"].value_counts().items())]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
