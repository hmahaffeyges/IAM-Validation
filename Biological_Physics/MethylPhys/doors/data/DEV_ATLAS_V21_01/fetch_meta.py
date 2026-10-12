"""Atlas v2.1 prep: fetch METADATA ONLY for candidate series (GEO SOFT text, GEO suppl filelist.txt, ENA read_run file report).
Never downloads an IDAT, beta table, read, BAM or .pat file. Writes one JSON per series into meta_cache/ and resumes."""
import json, os, re, sys, time, urllib.request, urllib.parse

CACHE = "meta_cache"; os.makedirs(CACHE, exist_ok=True)

def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if "404" in str(e):
                return None
            time.sleep(2 + 3 * k)
    return None

def soft_blocks(txt, head):
    out, cur = [], None
    for line in txt.splitlines():
        if line.startswith("^" + head):
            cur = {"_id": line.split("=", 1)[1].strip()}; out.append(cur); continue
        if cur is None or not line.startswith("!"):
            continue
        k, _, v = line[1:].partition(" = ")
        cur.setdefault(k.strip(), []).append(v.strip())
    return out

def gse_dir(acc):
    n = acc[3:]
    return f"GSE{n[:-3] or ''}nnn" if len(n) > 3 else "GSEnnn"

def fetch(acc):
    fn = f"{CACHE}/{acc}.json"
    if os.path.exists(fn):
        return
    base = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=%s&targ=%s&form=text&view=brief"
    ser = get(base % (acc, "self"))
    gsm = get(base % (acc, "gsm"))
    flist = get(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_dir(acc)}/{acc}/suppl/filelist.txt")
    if ser is None or gsm is None:
        print("FAIL", acc); return
    s = soft_blocks(ser, "SERIES")[0]
    samples = soft_blocks(gsm, "SAMPLE")
    rec = {"acc": acc, "series": s, "samples": samples, "filelist": flist,
           "queried": time.strftime("%Y-%m-%d"), "urls": [base % (acc, "self"), base % (acc, "gsm")]}
    # sequencing: ENA run report via the BioProject / SRA study in the series relations
    rel = " ".join(s.get("Series_relation", []))
    prj = re.findall(r"(PRJNA\d+)", rel) + re.findall(r"(SRP\d+)", rel)
    if prj:
        q = urllib.parse.urlencode({"accession": prj[0], "result": "read_run", "format": "tsv",
                                    "fields": "run_accession,experiment_accession,sample_alias,library_strategy,instrument_model,read_count,base_count,fastq_bytes,fastq_ftp,submitted_bytes"})
        rec["ena_runs"] = get("https://www.ebi.ac.uk/ena/portal/api/filereport?" + q)
        rec["ena_project"] = prj[0]
    json.dump(rec, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print("ok", acc, len(samples), flush=True)

if __name__ == "__main__":
    accs = [a.strip() for a in open(sys.argv[1]) if a.strip()]
    for a in sorted(accs):
        fetch(a)
