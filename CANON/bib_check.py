#!/usr/bin/env python3
"""Bibliography check (LESSONS R6, 2026-10-10): every docs/book/iam.bib entry with a DOI is compared with its CrossRef record: year, first author's
family name, volume, first page, and title (normalised word overlap). CrossRef answers are cached in CANON/bib_crossref_cache.json (committed), so
the check reruns offline and the push gate can run it. Writes CANON/bib_check_report.csv. Exit 1 if any entry is MISMATCH or its DOI does not resolve.
Usage: python3 CANON/bib_check.py [--refresh]"""
import os, re, sys, json, time, unicodedata, requests, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BIB = os.path.join(ROOT, "docs/book/iam.bib")
CACHE = os.path.join(ROOT, "CANON/bib_crossref_cache.json"); OUT = os.path.join(ROOT, "CANON/bib_check_report.csv")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}; refresh = "--refresh" in sys.argv
EXC = json.load(open(os.path.join(ROOT, "CANON/bib_exceptions.json")))   # per-entry, with the evidence for each
def norm(s):
    s = re.sub(r"<[^>]+>", " ", (s or "").replace("\u2019", "'")); s = re.sub(r"\\[`'\"^~=.uvHckr]\s*", "", s)          # tags (MathML), LaTeX accent commands
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\\[a-z]+|[{}]", "", s))
def entries(t):
    for blk in re.split(r"\n(?=@\w+\s*\{)", "\n" + t):                       # one block per entry, however it is closed
        m = re.match(r"\s*@(\w+)\s*\{\s*([^,]+),(.*)", blk, re.S)
        if not m: continue
        f = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", m.group(3))}   # up to 3 brace levels
        yield m.group(2).strip(), m.group(1).lower(), f
rows = []
ALL = list(entries(open(BIB, encoding="utf-8").read())); NAT = len(re.findall(r"(?m)^@\w+\s*\{", open(BIB, encoding="utf-8").read()))
assert len(ALL) == NAT, f"parsed {len(ALL)} entries of {NAT}"
for key, typ, f in ALL:
    doi = f.get("doi", "").strip().lower().replace("https://doi.org/", "")
    if not doi: rows.append(dict(key=key, status="NO DOI", detail=typ)); continue
    if refresh or doi not in cache:
        for t in range(4):
            try:
                r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=60)
                if r.status_code == 404:
                    h = requests.get(f"https://doi.org/api/handles/{doi}", timeout=60)   # DataCite and other agencies (arXiv, Zenodo)
                    cache[doi] = {"agency": "non-CrossRef", "handle_ok": h.ok and h.json().get("responseCode") == 1} if h.ok else None; break
                if r.ok and r.text.startswith("{"):
                    x = r.json()["message"]; cache[doi] = dict(title=(x.get("title") or [""])[0], year=(x.get("issued", {}).get("date-parts") or [[None]])[0][0],
                        first_family=((x.get("author") or [{}])[0]).get("family", ""), volume=x.get("volume"), page=x.get("page"), journal=(x.get("container-title") or [""])[0]); break
            except requests.RequestException: pass
            time.sleep(2)
        time.sleep(0.05)
    c = cache.get(doi)
    if c is None: rows.append(dict(key=key, doi=doi, status="DOI NOT FOUND")); continue
    if "agency" in c: rows.append(dict(key=key, doi=doi, status="OK (resolves, non-CrossRef)" if c["handle_ok"] else "DOI NOT FOUND")); continue
    prob = []
    y = re.sub(r"\D", "", f.get("year", ""))[:4]
    if y and c["year"] and abs(int(y) - int(c["year"])) > 1: prob.append(f"year {y} vs {c['year']}")
    au = norm(f.get("author", "").split(" and ")[0]); fam = norm(c["first_family"]).strip()
    collab = "collaboration" in au or f.get("author", "").startswith("{")
    if fam and not collab and fam.split()[-1] not in au.split(): prob.append(f"first author '{f.get('author','').split(' and ')[0][:30]}' vs '{c['first_family']}'")
    vb, vc = norm(f.get("volume", "")).strip().lstrip("0"), norm(str(c["volume"] or "")).strip().lstrip("0")
    if vb and vc and vb != vc and vc != y: prob.append(f"volume {f['volume']} vs {c['volume']}")
    p0 = re.split(r"[-–]", f.get("pages", ""))[0].strip(); c0 = re.split(r"[-–]", c["page"] or "")[0].strip()
    if p0 and c0 and norm(p0).strip() != norm(c0).strip(): prob.append(f"first page {p0} vs {c0}")
    STOP = {"the", "of", "a", "an", "and", "in", "on", "for", "to", "with", "by", "at", "from", "as", "is"}
    def words(t):
        out, run = [], ""
        for w in norm(t).split():
            if len(w) == 1 and w.isalpha(): run += w; continue        # MathML letters: M I C R O S C O P E -> microscope
            if run: out.append(run); run = ""
            out.append(w)
        if run: out.append(run)
        return {w for w in out if w not in STOP}
    wa, wb = words(f.get("title", "")), words(c["title"]); sh, lo = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    ov = len(sh & lo) / max(len(sh), 1)                                   # words of the shorter title found in the longer (CrossRef may hold the main title only)
    if c["title"] and ov < 0.6: prob.append(f"title overlap {ov:.2f}: '{c['title'][:60]}'")
    ex = EXC.get(key, {}).get("skip", []); kept = [x for x in prob if not any(x.startswith(e) for e in ex)]
    rows.append(dict(key=key, doi=doi, status="MISMATCH" if kept else ("OK (exception recorded)" if prob else "OK"), detail="; ".join(kept or prob)))
json.dump(cache, open(CACHE, "w"), indent=0, sort_keys=True)
R = pd.DataFrame(rows).sort_values(["status", "key"]); R.to_csv(OUT, index=False)
print(R.status.value_counts().to_string())
bad = R[R.status.isin(["MISMATCH", "DOI NOT FOUND"])]
for _, r in bad.iterrows(): print(f"  {r.status:13s} {r.key}: {r.get('detail', '')}"[:220])
sys.exit(1 if len(bad) else 0)
