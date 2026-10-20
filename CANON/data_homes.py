#!/usr/bin/env python3
"""data_homes.py (2026-10-20, to-do step 19; author after the AWS suspension: "pull what we need to replicate or test our physics from Zenodo
and not have to rely on AWS"). Writes Biological_Physics/MethylPhys/DATA_HOMES.md: where every input lives permanently.
  A. what the chain reads: its frozen inputs (chain/FROZEN_INPUTS_v4.json), each checked against its committed checksum;
  B. what each reproduction test reads (reproduce/tests.json and the scripts its steps run), each source classed by its archive;
  C. what we hold in our S3 bucket: dataset downloads (doors/DATA_REGISTER.csv) and job outputs (CANON/box_jobs.json), with any other home.
--check refuses: the document stale; a frozen input whose checksum differs; any code the chain or a reproduction test runs that reads our
bucket (the AWS run wrapper, reproduce/box/, only uploads and is exempt); a test input with no archive outside our bucket."""
import hashlib, json, os, re, sys, collections, csv
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MP = "Biological_Physics/MethylPhys/"
OUT = os.path.join(ROOT, MP, "DATA_HOMES.md"); BUCKET = "methylphys-data-945451304272-us-west-2-an"
J = lambda p: json.load(open(os.path.join(ROOT, p), encoding="utf-8"))
ARCH = [(r"ncbi\.nlm\.nih\.gov/geo|\bGSE\d+|\bGSM\d+", "GEO (NCBI)"), (r"ebi\.ac\.uk/ena|\bSRR\d+|\bERR\d+|\bPRJ", "ENA / SRA"),
        (r"ebi\.ac\.uk/biostudies|E-MTAB-\d+|ArrayExpress", "ArrayExpress (EBI)"), (r"encodeproject\.org|encode-public|\bENCFF", "ENCODE"),
        (r"gdc\.cancer\.gov|TCGA-", "NCI GDC (TCGA, open access)"), (r"hgdownload\.soe\.ucsc\.edu|UCSC mm10|\bmm10\b", "UCSC genome browser"),
        (r"zenodo\.\d+|zenodo\.org", "Zenodo"), (r"GRCh38 analysis set", "ENCODE")]
TOOLS = re.compile(r"micro\.mamba\.pm|astral\.sh|github\.com/nloyfer|conda|pypi")
OURS = re.compile(re.escape(BUCKET) + r"|boto3|(?<![\w-])s3://(?!encode-public)")
problems = []

def sha(p): return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()

def chain_inputs():
    F = J(MP + "chain/FROZEN_INPUTS_v4.json"); rows = []
    for rel, h in F["files"].items():
        p = MP + "chain/" + rel; ok = os.path.exists(os.path.join(ROOT, p)) and sha(p) == h
        if not ok: problems.append(f"frozen input {rel}: checksum differs or file missing")
        rows.append((rel, h[:12], "repository" + ("" if ok else " (CHECKSUM DIFFERS)")))
    return rows

def scripts_of(t):
    """the committed files a test's steps run, and the files those shell scripts run in turn"""
    seen, todo = [], []
    for s in t.get("steps", []):
        todo += re.findall(r"(?:\$MP/|Biological_Physics/MethylPhys/)([\w./-]+\.(?:py|sh))", s["run"])
    while todo:
        f = MP + todo.pop(0)
        if f in seen or not os.path.exists(os.path.join(ROOT, f)): continue
        seen.append(f)
        if f.endswith(".sh"): todo += re.findall(r"(?:\$MP/|Biological_Physics/MethylPhys/)([\w./-]+\.(?:py|sh))", open(os.path.join(ROOT, f), encoding="utf-8").read())
    return seen

def test_inputs():
    T = J(MP + "reproduce/tests.json")["tests"]; rows = []
    for t in T:
        files = scripts_of(t); text = " ".join(t["public_inputs"]) + " " + " ".join(open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read() for f in files)
        for f in files:
            if OURS.search(open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()): problems.append(f"test {t['id']}: {f} reads our S3 bucket")
        for inp in t["public_inputs"]:
            homes = sorted({n for rx, n in ARCH if re.search(rx, inp)})
            if not homes: homes = ["repository"] if "listed in" in inp else []
            if not homes: problems.append(f"test {t['id']}: input '{inp}' has no archive outside our bucket")
            rows.append((t["id"], inp, ", ".join(homes) or "NONE"))
        tools = sorted(set(m.group(0) for m in TOOLS.finditer(text)))
        if tools: rows.append((t["id"], "tools installed by the test", ", ".join(tools)))
    return rows

def code_reads_bucket():
    hits = []
    for base in (MP + "chain", MP + "reproduce"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, base)):
            rel = os.path.relpath(dp, ROOT)
            if "/box" in rel.replace(os.sep, "/") or "RETIRED" in rel or "__pycache__" in rel: continue
            for f in fn:
                if f.endswith((".py", ".sh")):
                    p = os.path.join(dp, f)
                    if OURS.search(open(p, encoding="utf-8", errors="replace").read()): hits.append(os.path.relpath(p, ROOT))
    for h in hits: problems.append(f"{h} reads our S3 bucket")
    return hits

def held():
    R = list(csv.DictReader(open(os.path.join(ROOT, MP + "doors/DATA_REGISTER.csv"), encoding="utf-8")))
    dl = collections.OrderedDict()
    for r in R:
        work = r["track"].endswith("_work") or r["dataset"] in ("out", "_test", "dlqueue", "geoq", "geoq2", "l2rsd", "mu0wide")
        acc = bool(re.match(r"(GSE|E-MTAB|SRP|PRJ|TCGA|ENC|phs)", r["dataset"]))
        kind = "our work files" if work else ("public download, by accession" if acc else "public downloads, grouped by topic")
        other = "none: deposit (step 19)" if work else ("its public archive" if acc else "public archives; accession list recorded at the bucket inventory (step 19)")
        dl.setdefault((kind, other), [0, 0.0]); dl[(kind, other)][0] += 1; dl[(kind, other)][1] += float(r["GB"] or 0)
    jobs = collections.defaultdict(int)
    for j in J("CANON/box_jobs.json")["jobs"]:
        for o in j.get("outputs") or []:
            if o.startswith("results/"): jobs["/".join(o.split("/")[:2]) + "/"] += 1
    ZEN = {"results/ATLAS_V21/": "Zenodo 10.5281/zenodo.23441999 and 10.5281/zenodo.23422688 (the atlas and its fit)"}
    REPO = {"results/REPRODUCE_V1/": "verdict summaries in reproduce/verified.json", "results/CHAIN_V4_CLEANCLONE/": "release-check records in doors/data/CHAIN_V4_CLEANCLONE",
            "results/REPORT_EXAMPLES_V4/": "final reports and bundles in doors/data/REPORT_EXAMPLES_V4", "results/SHOWCASE_V4/": "readings in doors/data/SHOWCASE_V4",
            "results/BOOK_CHECK/": "book-check records in docs/book/verification"}
    EXPL = {"results/PRED_TEMPERATURE_01/", "results/DEV_ARCH_01/"}   # exploratory runs, kept off public surfaces by name (author 2026-10-18)
    ne = sum(n for p, n in jobs.items() if p in EXPL)
    out = [(f"`{p}`", n, ZEN.get(p) or REPO.get(p) or "the committed record of each job (doors/)", "full outputs and logs: deposit (step 19)") for p, n in sorted(jobs.items(), key=lambda x: -x[1]) if p not in EXPL]
    if ne: out.append(("exploratory runs", ne, "the development log", "full outputs and logs: deposit (step 19)"))
    return dl, out

def text():
    A = chain_inputs(); B = test_inputs(); C = code_reads_bucket(); dl, out = held()
    L = ["# Where every input lives", "", "_Generated by CANON/data_homes.py; do not edit by hand. Checked on every push._", "",
         "Everything the chain and the reproduction tests read has a permanent home outside our own cloud account: the repository, a public",
         "archive, or a Zenodo record. Our AWS account only rents computers and keeps working copies. This file shows it, and the push is",
         "refused if any of it stops being true.", "",
         "## A. What the chain reads", "", "Every reference file the chain loads, with the checksum it is frozen at (chain/FROZEN_INPUTS_v4.json).", "",
         "| file | sha256 (first 12) | permanent home |", "|---|---|---|"] + [f"| `{a}` | `{b}` | {c} |" for a, b, c in A] + [
         "", "## B. What each reproduction test reads", "", "The inputs of each test in reproduce/tests.json, and the archive each comes from (read from the test's own scripts).", "",
         "| test | input | archive |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in B] + [
         "", f"Code that reads our S3 bucket, in the chain or any reproduction test: **{'none' if not C else ', '.join(C)}**. (The AWS run wrapper, reproduce/box/, only uploads a run's status and results.)",
         "", "## C. What we hold in our S3 bucket", "", "Working copies and our own outputs. Nothing a researcher needs is only here; what is only here is our history, and step 19 deposits it on Zenodo.", "",
         "| downloads (doors/DATA_REGISTER.csv) | folders | GB | other home |", "|---|---|---|---|"] + [f"| {k} | {v[0]} | {v[1]:.1f} | {o} |" for (k, o), v in dl.items()] + [
         "", "| job outputs (CANON/box_jobs.json) | jobs | already outside S3 | still only in S3 |", "|---|---|---|---|"] + [f"| {p} | {n} | {z} | {s} |" for p, n, z, s in out] + [""]
    return "\n".join(L)

if __name__ == "__main__":
    t = text()
    if "--check" in sys.argv:
        stale = not os.path.exists(OUT) or open(OUT, encoding="utf-8").read() != t
        if stale: problems.append("DATA_HOMES.md differs (run python3 CANON/data_homes.py)")
        print("data_homes: " + ("ok (every chain and test input has a home outside our bucket)" if not problems else f"REFUSED: {len(problems)}: " + "; ".join(problems[:6])))
        sys.exit(1 if problems else 0)
    open(OUT, "w", encoding="utf-8").write(t); print("data_homes: written" + (f" ({len(problems)} problem(s): " + "; ".join(problems[:6]) + ")" if problems else ""))
