"""PHY-092 on GitHub (the last pipeline item): fetch each p092-<GSM> run's chain tiles (<GSM>_tiles.csv.gz) and run a8_where.py on them. Reads only
chain outputs. Writes xout/a8_where_summary.txt."""
import os, sys, io, json, zipfile, urllib.request, urllib.error
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.abspath("xout"); TD = os.path.join(OUT, "tiles"); os.makedirs(TD, exist_ok=True)
TOK = os.environ["GH_TOKEN"]; REPO = os.environ.get("GITHUB_REPOSITORY", "hmahaffeyges/IAM-Validation")
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None
def get(u, raw=False):
    rq = urllib.request.Request(u, headers={"Authorization": "Bearer " + TOK, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.build_opener(_NoRedirect).open(rq, timeout=600) as r: b = r.read()
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308): raise
        b = urllib.request.urlopen(e.headers["Location"], timeout=600).read()
    return b if raw else json.loads(b)
if len(sys.argv) > 1: os.environ["TILE_AGG"] = sys.argv[1]   # amendment 1: counter tiles pooled
sys.path.insert(0, H); import a8_where as A
runs = {}
for page in range(1, 4):
    for w in get(f"https://api.github.com/repos/{REPO}/actions/runs?per_page=100&page={page}")["workflow_runs"]:
        t = w.get("display_title") or ""
        if t.startswith("pq:p092-") and w["conclusion"] == "success" and t[8:] not in runs: runs[t[8:]] = w["id"]
for g in A.WT + A.KO:
    if g not in runs: sys.exit(f"missing recount for {g}")
    z = zipfile.ZipFile(io.BytesIO(get(get(f"https://api.github.com/repos/{REPO}/actions/runs/{runs[g]}/artifacts")["artifacts"][0]["archive_download_url"], raw=True)))
    open(os.path.join(TD, f"{g}_tiles.csv.gz"), "wb").write(z.read(f"eout/{g}_tiles.csv.gz"))
A.main(TD, OUT)
