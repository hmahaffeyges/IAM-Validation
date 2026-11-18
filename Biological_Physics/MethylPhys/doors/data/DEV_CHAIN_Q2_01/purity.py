#!/usr/bin/env python3
"""Purity of sorted libraries (overhaul E2b; register OVERHAUL plan). Runs ON A GITHUB MACHINE. For each of the joint run's sorted atlas libraries, the
Deconvolver's top share and second share from the kept bundle (no new reading), so the threshold that calls a sample a mixture comes from measurement."""
import os, json, io, zipfile, urllib.request, urllib.error, statistics as st
R = os.environ.get("GITHUB_REPOSITORY", "hmahaffeyges/IAM-Validation"); TOK = os.environ.get("GH_TOKEN")
class NR(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None
op = urllib.request.build_opener(NR)
def get(u):
    q = urllib.request.Request(u, headers={"Authorization": "Bearer " + TOK, "Accept": "application/vnd.github+json"})
    try: return op.open(q, timeout=120).read()
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308): return urllib.request.urlopen(e.headers["Location"], timeout=120).read()
        raise
RUNS = [44025270992, 44025389058, 44025504407, 44025619659, 44025737860, 44025856493, 44025975935, 44026093003, 44026557445, 44027651329, 44027704068, 44027819051, 44027934510, 44028063935, 44028197239, 44028317327, 44028443207, 44028934009, 44029058222, 44030295580, 44030296033, 44030416228, 44030549956, 44030674704, 44030862758, 44030998833, 44031129405, 44031255791, 44031378085, 44031507255, 44032754628, 44032876997, 44033761840, 44033891975, 44034057152, 44034354733, 44034544669, 44034727011, 44035951877, 44036088551, 44036218201, 44036839334, 44041989118, 44042020854, 44042153537, 44042288650, 44042424420, 44042558061, 44042688675, 44042819279, 44042949286, 44055531343, 44055687068, 44055845122, 44055997265, 44056147516, 44056308431, 44056466570, 44056622570, 44056785872, 44056949249, 44058489495, 44058647952, 44058807235, 44058966357, 44059124364, 44059288434, 44059443302, 44059604148, 44059763030, 44059925404, 44061467555, 44061632849, 44061794203, 44062119527, 44062280184, 44062445991, 44062605103, 44062767726, 44062937167, 44064517272, 44064681552, 44064838267, 44064992541, 44065146217, 44065297335, 44065461859, 44065629726, 44065789129, 44065956871, 44067494183, 44067649497, 44067812854, 44067970864, 44068126554, 44068284500, 44068440135, 44068599332, 44068755689, 44068913135, 44070600959, 44070764606, 44070930660, 44071095406, 44071256791, 44071416585, 44071581463, 44071744001, 44071905231, 44073445747, 44073600717, 44073756724, 44073911155, 44074064641, 44074217899, 44074368811, 44074519286, 44074685896, 44074845206, 44076346417, 44076498007, 44076649704, 44076805353, 44076963800, 44077118374, 44077272268, 44077422178, 44077577907, 44077736487, 44079241652, 44079401878, 44079561846, 44079721526, 44079886059, 44080049780, 44080210215, 44080374010, 44080527918, 44080693481, 44082253308, 44082410098, 44082572443, 44082731984, 44082896002, 44083061407, 44083220469, 44083377423, 44083543242, 44083702458, 44085293254, 44085458435, 44085618247, 44085779162, 44085940351, 44086144094, 44086384836]
rows = []
for rid in RUNS:
    try:
        A = json.loads(get(f"https://api.github.com/repos/{R}/actions/runs/{rid}/artifacts"))["artifacts"]
        z = zipfile.ZipFile(io.BytesIO(get(A[0]["archive_download_url"])))
    except Exception: continue
    for n in z.namelist():
        if n.startswith("joint_GSM") and n.endswith("_bundle.json"):
            B = json.loads(z.read(n)); c = sorted(((r["cell"], r["share"]) for r in (B.get("cell_shares") or {}).get("cells", []) if r.get("share") is not None), key=lambda t: -t[1])
            if c: rows.append(dict(gsm=n.split("_")[1], top=c[0][0], top_share=c[0][1], second=c[1][0] if len(c) > 1 else None, second_share=c[1][1] if len(c) > 1 else 0.0))
t = sorted(r["top_share"] for r in rows); s2 = sorted(r["second_share"] for r in rows)
q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
print(f"::notice::PURITY: {len(rows)} sorted libraries; top share min {t[0]:.3f}, 1st pct {q(t, .01):.3f}, 5th pct {q(t, .05):.3f}, median {st.median(t):.3f}; second share median {st.median(s2):.3f}, 95th pct {q(s2, .95):.3f}, max {s2[-1]:.3f}")
low = sorted(rows, key=lambda r: r["top_share"])[:10]
print("::notice::PURITY lowest: " + "; ".join(f"{r['gsm']} {r['top']} {r['top_share']:.3f} / {r['second']} {r['second_share']:.3f}" for r in low))
os.makedirs("xout", exist_ok=True); json.dump(rows, open("xout/purity.json", "w"), indent=1)
