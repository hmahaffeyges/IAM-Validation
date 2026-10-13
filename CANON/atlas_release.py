"""Writes CANON/atlas_release.json from the public Zenodo record of IAM-Atlas v2 (DOI, file count, total size in bytes), so pages that
state the download size read it from a committed file with a source. Usage: python3 CANON/atlas_release.py [RECORD_ID]"""
import sys, os, json, datetime, urllib.request
REC = sys.argv[1] if len(sys.argv) > 1 else "23325865"
m = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{REC}", timeout=60))
out = {"doi": m["doi"], "record": int(REC), "title": m["metadata"]["title"], "files": len(m["files"]),
       "total_bytes": sum(f["size"] for f in m["files"]), "checked": datetime.date.today().isoformat()}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas_release.json"), "w"), indent=1); print(out)
