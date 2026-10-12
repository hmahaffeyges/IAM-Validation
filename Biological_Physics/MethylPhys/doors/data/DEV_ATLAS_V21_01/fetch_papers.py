"""Atlas v2.1 prep: for each PubMed id, fetch Europe PMC metadata and (if open access) the full-text XML; keep the methods
sentences that state purification, culture, health status, donors and sex. Papers only; no data files."""
import json, os, re, sys, time, urllib.request, urllib.parse
OUT = "paper_cache"; os.makedirs(OUT, exist_ok=True)
KEY = re.compile(r"(cultur|passag|expan|in vitro|FACS|flow[- ]cytometr|sort|MACS|magnetic|bead|isolat|purif|Ficoll|density gradient|"
                 r"healthy|control subjects|volunteer|donor|post[- ]?mortem|male|female|EasySep|RosetteSep|microbead|laser capture|"
                 r"organoid|fibroblast|smooth muscle|population doubling)", re.I)

def get(url):
    for k in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if "404" in str(e): return None
            time.sleep(3 * (k + 1))
    return None

def one(pmid):
    fn = f"{OUT}/{pmid}.json"
    if os.path.exists(fn): return
    q = urllib.parse.urlencode({"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json", "resultType": "core"})
    meta = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + q)
    rec = {"pmid": pmid, "queried": time.strftime("%Y-%m-%d")}
    try:
        r = json.loads(meta)["resultList"]["result"][0]
    except Exception:
        json.dump(rec, open(fn, "w")); return
    rec.update({k: r.get(k) for k in ("title", "doi", "pmcid", "journalTitle", "pubYear", "authorString", "isOpenAccess")})
    rec["affiliation"] = (r.get("affiliation") or "")[:600]
    rec["abstract"] = re.sub("<[^>]+>", " ", r.get("abstractText") or "")
    if r.get("pmcid"):
        xml = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{r['pmcid']}/fullTextXML")
        if xml and xml.lstrip().startswith('<'):
            m = re.search(r"<sec[^>]*sec-type=\"(materials|methods)[^\"]*\".*?</sec>\s*(?=<sec[^>]*sec-type=\"(results|discussion)|</body>)", xml, re.S)
            body = m.group(0) if m else xml
            txt = re.sub(r"<[^>]+>", " ", body); txt = re.sub(r"\s+", " ", txt)
            sents = re.split(r"(?<=[.;])\s+(?=[A-Z])", txt)
            rec["methods_hits"] = [s[:500] for s in sents if KEY.search(s)][:120]
            rec["fulltext"] = True
        else:
            rec["fulltext"] = False
    json.dump(rec, open(fn, "w"))
    print("ok", pmid, rec.get("pmcid"), len(rec.get("methods_hits", [])), flush=True)

if __name__ == "__main__":
    for p in sorted(set(x.strip() for x in open(sys.argv[1]) if x.strip())):
        one(p)
