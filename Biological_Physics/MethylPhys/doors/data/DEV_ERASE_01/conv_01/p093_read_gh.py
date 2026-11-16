"""PHY-093 reading (exploration; register T1), after every p093 library. Reads only chain outputs (<GSM>_q2.json from run_sample.py --pat
--dev-count-only --dev-count-sites islands): the stray-mark rate genome-wide (gain_err) and at the islands territory (count_sites.gain_err), with
their opportunities; each library's meter (conv_fail + tc_err) from PHY-091 (conv_01/a17_read.csv: conv_fail; tc_err from amendment 2's median
where not counted per library). Pooled by cell type (referee_runs.csv titles): islands and genome-wide, each minus the meter, with counting errors.
Null check: genome-wide gain_err here against PHY-091's aligned-file count of the same reads. Writes xout/p093_read_summary.txt, xout/p093_read.csv."""
import os, io, csv, json, math, zipfile, urllib.request, urllib.error, statistics as st
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
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
runs = {}
for page in range(1, 6):
    for w in get(f"https://api.github.com/repos/{REPO}/actions/workflows/erase_test.yml/runs?per_page=100&page={page}")["workflow_runs"]:
        t = w.get("display_title") or ""
        if t.startswith("pq:p093-GSM") and w["conclusion"] == "success" and t[3:] not in runs: runs[t[3:]] = w["id"]
H = os.path.dirname(os.path.abspath(__file__))
A = {r["gsm"]: r for r in csv.DictReader(open(os.path.join(H, "a17_read.csv")))}
R = {r["gsm"]: r for r in csv.DictReader(open(os.path.join(H, "..", "..", "DEV_CHAIN_Q2_01", "referee_runs.csv")))}
TC = 0.00046
def cell(g):
    t = (R.get(g, {}).get("title", "") + " " + R.get(g, {}).get("genotype", "")).lower()
    for k, n in (("cd14", "monocytes"), ("cd19", "B"), ("cd4", "CD4 T"), ("cd8", "CD8 T")):
        if k in t: return n
    return "other"
rows = []
for key, rid in sorted(runs.items()):
    for a in [a for a in get(f"https://api.github.com/repos/{REPO}/actions/runs/{rid}/artifacts")["artifacts"] if a["size_in_bytes"] < 2e7]:
        z = zipfile.ZipFile(io.BytesIO(get(a["archive_download_url"], raw=True)))
        for n in z.namelist():
            if n.endswith("_q2.json"):
                c = json.loads(z.read(n)); g = c["gsm"]; cs = c.get("count_sites") or {}
                m = float(A[g]["conv_fail"]) + TC if g in A else None
                rows.append(dict(gsm=g, cell=cell(g), gain_iso=c.get("gain_iso"), gain_opp=c.get("gain_opp"), gain_err=c.get("gain_err"),
                                 isl_iso=cs.get("gain_iso"), isl_opp=cs.get("gain_opp"), isl_err=cs.get("gain_err"), meter=m,
                                 bam_gain=float(A[g]["gain_err"]) if g in A else None))
with open(os.path.join(OUT, "p093_read.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["gsm"]); w.writeheader(); [w.writerow(r) for r in rows]
L = [f"PHY-093 reading (T1, exploration): {len(rows)} libraries. Meter = PHY-091 conv_fail + tc_err {TC}."]
nr = [r["gain_err"] / r["bam_gain"] for r in rows if r["gain_err"] and r["bam_gain"]]
if nr: L.append(f"null check, molecule file / aligned file genome-wide: median {st.median(nr):.3f} ({min(nr):.3f}-{max(nr):.3f}), n {len(nr)}")
for k in ("B", "CD4 T", "CD8 T", "monocytes", "other"):
    g = [r for r in rows if r["cell"] == k and r["isl_opp"]]
    if not g: continue
    ii, io_ = sum(r["isl_iso"] for r in g), sum(r["isl_opp"] for r in g)
    gw = [r for r in g if r["gain_iso"] is not None and r["gain_opp"]]   # 2026-11-15: genome-wide counts missing in some files
    gi, go = (sum(r["gain_iso"] for r in gw), sum(r["gain_opp"] for r in gw)) if gw else (0, 1)
    m = st.median(r["meter"] for r in g if r["meter"] is not None)
    e_i, e_g = ii / io_, gi / go; s_i = math.sqrt(ii) / io_
    L.append(f"  {k:9s} n {len(g):2d}: islands {e_i:.4f} +/- {s_i:.4f} ({ii} of {io_}); genome-wide {e_g:.4f}; meter {m:.4f}; "
             f"(genome-wide from {len(gw)} of {len(g)} libraries) writer islands {e_i - m:.4f} vs genome-wide {e_g - m:.4f} (ratio {(e_i - m) / (e_g - m) if gw and e_g != m else float('nan'):.2f})")
L.append("Reading guide (exploration): a ratio near 1 means the islands read at the writer's own one-step error plus the meter (T1 from A15 + A17); "
         "well below 1 means an islands shield to price.")
open(os.path.join(OUT, "p093_read_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
