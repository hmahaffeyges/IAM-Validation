#!/usr/bin/env python3
"""Null test of Met-A sequencing's held-site score (A_derived_held, stage_metaa_atlas.derived_held), overhaul step 2 (exploration). Runs ON A GITHUB MACHINE.
The score compares a library's disorder at its cell's held identity sites (atlas level >= 0.80) with the disorder of ONE level, 1 - eps0 (0.955), at every
site. Question: on healthy atlas libraries, does it read about 1 (a reading of the sample) or well above 1 for every type (a reading of the 0.80 cutoff)?
Reads the joint run's kept bundles (artifacts of explore runs since 2026-11-17), the top cell of each library (share >= 0.5): A_derived_held, its ordinary
Met-A A, and, as the cutoff's own size, the atlas's mean disorder at those sites is not in the bundle, so the type medians are compared across types."""
import os, json, subprocess, statistics as st, collections, zipfile, io, glob, tempfile
R = os.environ.get("GITHUB_REPOSITORY", "hmahaffeyges/IAM-Validation")
def gh(*a): return subprocess.run(["gh", "api", *a], capture_output=True, text=True, check=True).stdout
runs = [44025270992, 44025389058, 44025504407, 44025619659, 44025737860, 44025856493, 44025975935, 44026093003, 44026557445, 44027651329, 44027704068, 44027819051, 44027934510, 44028063935, 44028197239, 44028317327, 44028443207, 44028934009, 44029058222, 44030295580, 44030296033, 44030416228, 44030549956, 44030674704, 44030862758, 44030998833, 44031129405, 44031255791, 44031378085, 44031507255, 44032754628, 44032876997, 44033761840, 44033891975, 44034057152, 44034354733, 44034544669, 44034727011, 44035951877, 44036088551, 44036218201, 44036839334, 44041989118, 44042020854, 44042153537, 44042288650, 44042424420, 44042558061, 44042688675, 44042819279, 44042949286, 44055531343, 44055687068, 44055845122, 44055997265, 44056147516, 44056308431, 44056466570, 44056622570, 44056785872, 44056949249, 44058489495, 44058647952, 44058807235, 44058966357, 44059124364, 44059288434, 44059443302, 44059604148, 44059763030, 44059925404, 44061467555, 44061632849, 44061794203, 44062119527, 44062280184, 44062445991, 44062605103, 44062767726, 44062937167, 44064517272, 44064681552, 44064838267, 44064992541, 44065146217, 44065297335, 44065461859, 44065629726, 44065789129, 44065956871, 44067494183, 44067649497, 44067812854, 44067970864, 44068126554, 44068284500, 44068440135, 44068599332, 44068755689, 44068913135, 44070600959, 44070764606, 44070930660, 44071095406, 44071256791, 44071416585, 44071581463, 44071744001, 44071905231, 44073445747, 44073600717, 44073756724, 44073911155, 44074064641, 44074217899, 44074368811, 44074519286, 44074685896, 44074845206, 44076346417, 44076498007, 44076649704, 44076805353, 44076963800, 44077118374, 44077272268, 44077422178, 44077577907, 44077736487, 44079241652, 44079401878, 44079561846, 44079721526, 44079886059, 44080049780, 44080210215, 44080374010, 44080527918, 44080693481, 44082253308, 44082410098, 44082572443, 44082731984, 44082896002, 44083061407, 44083220469, 44083377423, 44083543242, 44083702458, 44085293254, 44085458435, 44085618247, 44085779162, 44085940351, 44086144094, 44086384836]   # the joint run's train (train_readings.json, 156 libraries)
rows = []; seen = set()
import urllib.request, urllib.error
TOK = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
class _NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None
_op = urllib.request.build_opener(_NoRedir)
def get(u):
    q = urllib.request.Request(u, headers={"Authorization": "Bearer " + TOK, "Accept": "application/vnd.github+json"})
    try: return _op.open(q, timeout=120).read()
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308): return urllib.request.urlopen(e.headers["Location"], timeout=120).read()   # storage host: no auth header
        raise
firsterr = []
nfail = 0
for rid in runs:
    try:
        A = json.loads(get(f"https://api.github.com/repos/{R}/actions/runs/{rid}/artifacts"))["artifacts"]
        if not A or A[0]["expired"] or A[0]["size_in_bytes"] > 60_000_000: continue
        z = zipfile.ZipFile(io.BytesIO(get(A[0]["archive_download_url"])))
    except Exception as e:
        nfail += 1; firsterr.append(repr(e)[:200]); continue
    for n in z.namelist():
        if not (n.startswith("joint_GSM") and n.endswith("_bundle.json")): continue
        g = n.split("_")[1]
        if g in seen: continue
        B = json.loads(z.read(n)); m = B.get("met_a_each_cell") or {}; cl = m.get("cells") if isinstance(m, dict) else None
        if not cl: continue
        top = max(cl, key=lambda r: r.get("share") or 0)
        if (top.get("share") or 0) < 0.5 or top.get("A_derived_held") is None: continue
        seen.add(g); rows.append(dict(gsm=g, cell=top["cell"], share=top["share"], A=top.get("A"), A_held=top["A_derived_held"], sites=top.get("derived_held_sites")))
print(f"::notice::MET-A HELD NULL: runs {len(runs)}, download failures {nfail}, libraries kept {len(rows)}; first error {firsterr[:1]}")
if not rows: raise SystemExit(0)
by = collections.defaultdict(list)
for r in rows: by[r["cell"]].append(r)
med = sorted(((c, len(v), st.median(x["A_held"] for x in v), st.median(x["A"] for x in v if x["A"] is not None)) for c, v in by.items()), key=lambda t: t[2])
allh = [r["A_held"] for r in rows]
print(f"::notice::MET-A HELD NULL: {len(rows)} healthy libraries, {len(by)} types; A_held median {st.median(allh):.3f} (range {min(allh):.3f}-{max(allh):.3f}); "
      f"share of libraries above 1.2: {sum(v > 1.2 for v in allh) / len(allh):.2f}; ordinary Met-A A median {st.median(r['A'] for r in rows if r['A'] is not None):.3f}")
print("::notice::MET-A HELD by type (type, n, A_held median, A median): " + "; ".join(f"{c} {n} {h:.2f} {a:.2f}" for c, n, h, a in med)[:3800])
os.makedirs("xout", exist_ok=True); json.dump(rows, open("xout/metheld_null.json", "w"), indent=1)
