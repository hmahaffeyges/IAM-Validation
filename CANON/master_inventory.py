"""Master Inventory of every data set (author 2026-10-12): one generated table, never edited by hand.
Sources, all committed: doors/DATA_REGISTER.csv (what is in S3: prefix, GB, files, planned tests), doors/data/READY_TO_TEST/s3_format_check.csv
(format: chain-readable or not), CANON/data_spent.csv (read before, and by which method), every doors/data/*/candidates_search_*.json and
doors/data/DEV_ATLAS_COMMISSION_01/gold_mine_map.csv (public sets not downloaded, with their use), every doors/*.md note (test status from each test note's
"**Test data:**" line and its title: its own _OUTCOME file = scored, else SEALED, NOT SUITABLE, design), and the Master Testing Manifest (ranked tests; every accession in it must be
in the inventory). Writes Biological_Physics/MethylPhys/MASTER_INVENTORY.csv and MASTER_INVENTORY.md.
Usage: python3 CANON/master_inventory.py [--check]   (--check: exit 1 if the committed files differ from what this would write)"""
import os, re, sys, json, glob, io
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); MP = os.path.join(ROOT, "Biological_Physics", "MethylPhys"); DOORS = os.path.join(MP, "doors")
OUTC, OUTM = os.path.join(MP, "MASTER_INVENTORY.csv"), os.path.join(MP, "MASTER_INVENTORY.md")
ACC = re.compile(r"\b(GSE\d{4,6}|E-MTAB-\d{3,5}|PRJ[A-Z]{2}\d{3,8}|TCGA-[A-Z]{2,4})\b")
ORDER = {"scored": 6, "sealed": 5, "not suitable": 4, "designed": 3, "candidate": 2, "named in a note": 1, "": 0}
def note_status(path, text):
    t = text.split("\n", 1)[0]
    if "NOT SUITABLE" in t: return "not suitable"
    if "SEALED" in t: return "sealed"
    if "candidate" in t.lower(): return "candidate"
    if "design" in t.lower(): return "designed"
    return "named in a note"
def rows_from(obj, out):
    if isinstance(obj, dict):
        if isinstance(obj.get("accession"), str): out.append(obj)
        for v in obj.values(): rows_from(v, out)
    elif isinstance(obj, list):
        for v in obj: rows_from(v, out)
def build():
    R = {}
    def row(a):
        return R.setdefault(a, dict(accession=a, in_s3="", s3_gb="", format="", chain_readable="", spent="", test_status="", tested_by="", test_records="",
                                     use="", cell_types="", independent="", raw_public="", what="", source=""))
    def add(a, k, v):
        r = row(a); v = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
        if v and v not in r[k].split("; "): r[k] = (r[k] + "; " + v) if r[k] else v
    reg = pd.read_csv(os.path.join(DOORS, "DATA_REGISTER.csv"))
    for x in reg.itertuples():
        for a in ACC.findall(str(x.dataset)) or [str(x.dataset)]:
            add(a, "in_s3", x.prefix); add(a, "s3_gb", x.GB); add(a, "what", x.what); add(a, "source", "S3 register")
    fc = pd.read_csv(os.path.join(DOORS, "data", "READY_TO_TEST", "s3_format_check.csv"))
    for x in fc.itertuples():
        add(x.series, "format", x.format); add(x.series, "chain_readable", "yes" if str(x.format).upper() == "IDAT" else "no"); add(x.series, "in_s3", x.prefix)
    sp = pd.read_csv(os.path.join(ROOT, "CANON", "data_spent.csv"))
    for x in sp.itertuples(): add(x.dataset, "spent", x.spent_status)
    gm = pd.read_csv(os.path.join(DOORS, "data", "DEV_ATLAS_COMMISSION_01", "gold_mine_map.csv"))
    U = {"A": "atlas build", "B": "held-out test", "C": "truth set", "D": "research", "E": "not usable"}
    for x in gm.itertuples():
        add(x.accession, "use", U.get(str(x.use_primary)[:1], x.use_primary) + (f" ({x.not_usable_reason})" if str(x.use_primary).startswith("E") and isinstance(x.not_usable_reason, str) else ""))
        add(x.accession, "cell_types", x.atlas_types if isinstance(x.atlas_types, str) else x.new_types_norm); add(x.accession, "independent", x.independent_of_atlas)
        add(x.accession, "raw_public", x.raw_data_public); add(x.accession, "what", x.title); add(x.accession, "source", "gold-mine map")
    for f in sorted(glob.glob(os.path.join(DOORS, "data", "*", "candidates_search_*.json"))):
        L = []; rows_from(json.load(open(f, encoding="utf-8")), L); tag = os.path.basename(os.path.dirname(f))
        for d in L:
            a = d["accession"].strip()
            if ACC.fullmatch(a): add(a, "source", f"candidates {tag}"); add(a, "what", d.get("title"))
    # Test status comes ONLY from a design note's "**Test data:**" line (the sets the test reads); its own _OUTCOME file means scored.
    # Every other mention is listed under test_records without a status.
    for f in sorted(glob.glob(os.path.join(DOORS, "*.md"))):
        nm = os.path.basename(f)[:-3]
        if not nm.startswith(("DEV_", "PROC_", "COMMISSIONING")): continue
        t = open(f, encoding="utf-8").read()
        for a in sorted(set(ACC.findall(t))): add(a, "test_records", nm)
        if nm.endswith("_OUTCOME"): continue
        for line in re.findall(r"^\*\*Test data:\*\*(.*)$", t, re.M):
            st = "scored" if os.path.exists(os.path.join(DOORS, nm + "_OUTCOME.md")) else note_status(f, t)
            for a in sorted(set(ACC.findall(line))):
                r = row(a); add(a, "tested_by", nm)
                if ORDER[st] > ORDER[r["test_status"]]: r["test_status"] = st
    T = pd.DataFrame(R.values()).sort_values("accession").reset_index(drop=True)
    for c in ("what", "test_records", "raw_public", "cell_types"): T[c] = T[c].str.slice(0, 160)
    return T
def render(T):
    # the ranked job list is the Master Testing Manifest's open tests (CANON/manifest.json, 2026-10-13); next_jobs.json is retired
    IMP = {"replication": 0, "commissioning": 0, "atlas": 1, "public result": 2, "physics": 3, "research": 4}; COST = {"laptop": 0, "box": 1}
    n = lambda m: int(m.sum())
    L = ["# Master Inventory: every data set, where it is, what it has been used for, and what it is for", "",
         "Generated by `CANON/master_inventory.py` from the committed records (S3 register, format check, spent-data list, candidate searches,",
         "gold-mine map, test notes); never edited by hand. The full table: [`MASTER_INVENTORY.csv`](MASTER_INVENTORY.csv).",
         "Candidates found by search are metadata only and their counts are provisional until a design re-queries them; see each search's limits.", "",
         "## Totals", "",
         f"- data sets: {len(T)}; in S3: {n(T.in_s3 != '')}; chain-readable in S3 (IDAT): {n(T.chain_readable == 'yes')}",
         f"- public, not downloaded: {n((T.in_s3 == '') & (T.source != ''))}",
         f"- read before (spent or partly spent): {n(T.spent != '')}",
         "- by test status: " + ", ".join(f"{k} {n(T.test_status == k)}" for k in ("scored", "sealed", "not suitable", "designed", "candidate")),
         "- by use (gold-mine map): " + ", ".join(f"{k} {n(T.use.str.startswith(k))}" for k in ("atlas build", "held-out test", "truth set", "research", "not usable")), "",
         "## What is next", "", "Every test, open and done, with the data each one needs: the Master Testing Manifest, [MANIFEST.md](../../MANIFEST.md).", ""]
    L += ["", "## Sets tested (from each test note's Test data line)", "", "| set | status | test | in S3 | use |", "|---|---|---|---|---|"]
    S = T[T.tested_by != ""].sort_values(["test_status", "accession"])
    L += [f"| {r.accession} | {r.test_status} | {r.tested_by} | {'yes' if r.in_s3 else 'no'} | {r.use} |" for r in S.itertuples()]
    return "\n".join(L) + "\n"
if __name__ == "__main__":
    T = build(); md = render(T); buf = io.StringIO(); T.to_csv(buf, index=False); csv = buf.getvalue()
    if "--check" in sys.argv:
        old = [open(p, encoding="utf-8").read() if os.path.exists(p) else "" for p in (OUTC, OUTM)]
        if old != [csv, md]: print("master_inventory: MASTER_INVENTORY differs from the records (run python3 CANON/master_inventory.py)"); sys.exit(1)
        print("master_inventory: ok"); sys.exit(0)
    open(OUTC, "w", encoding="utf-8").write(csv); open(OUTM, "w", encoding="utf-8").write(md); print(f"master_inventory: {len(T)} sets written")
