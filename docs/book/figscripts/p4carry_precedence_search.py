# Bounded precedence search for Part 4 (block P4C-12 in MANIFEST.md): record counts in Europe PMC and arXiv
# for methylome/sky-statistics query pairs. Writes p4carry_precedence_search_result.json beside this script.
import requests, re, json, datetime
out={"run_utc":datetime.datetime.utcnow().isoformat(timespec='seconds'),"arxiv":{},"europepmc":{}}
arx={"HEALPix methylation":'all:HEALPix AND all:methylation',
     "power spectrum methylome":'all:"power spectrum" AND all:methylome',
     "cosmic microwave background epigenome":'all:"cosmic microwave background" AND all:epigenome',
     "needlet biological":'all:needlet AND all:biological',
     "spherical harmonics DNA methylation":'all:"spherical harmonics" AND all:"DNA methylation"'}
for k,q in arx.items():
    try:
        r=requests.get("http://export.arxiv.org/api/query",params={"search_query":q,"max_results":5},timeout=40)
        m=re.search(r"<opensearch:totalResults[^>]*>(\d+)</opensearch:totalResults>",r.text); out["arxiv"][k]=int(m.group(1)) if m else f"HTTP {r.status_code}"
    except Exception as e: out["arxiv"][k]=f"error {type(e).__name__}"
epmc={"spherical harmonic + methylome":'"spherical harmonic" AND methylome',
      "angular power spectrum + (methylation OR epigenome)":'"angular power spectrum" AND (methylation OR epigenome)',
      "needlet + (genome OR methylation)":'needlet AND (genome OR methylation)',
      "HEALPix + methylation":'HEALPix AND methylation',
      "HEALPix + genome":'HEALPix AND genome'}
for k,q in epmc.items():
    try:
        r=requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",params={"query":q,"format":"json","pageSize":5},timeout=40)
        j=r.json(); out["europepmc"][k]={"hits":j.get("hitCount"),"titles":[x.get("title") for x in j.get("resultList",{}).get("result",[])]}
    except Exception as e: out["europepmc"][k]=f"error {type(e).__name__}"
json.dump(out,open(__import__("os").path.join(__import__("os").path.dirname(__file__) or ".","p4carry_precedence_search_result.json"),"w"),indent=1)
print(json.dumps({k:(v if k!="europepmc" else {a:(b["hits"] if isinstance(b,dict) else b) for a,b in v.items()}) for k,v in out.items()},indent=1))
