"""PHY-091 amendment 3 reading (exploration; register A17 -> A15), after every p091c library. Reads only chain outputs (<GSM>_conv.json from
run_sample.py --bam --dev-count-only): gain_err, conv_fail (genomic non-CpG cytosines read as methylated, which also counts REAL CpH methylation),
tc_err, and lambda_meter.conv_fail_lambda (every cytosine of the unmethylated lambda spike-in read as C: the meter alone). Per library: the two meters
side by side, lambda reads (the two unspiked libraries are the negative controls: few or none expected), and the writer's error gain_err -
conv_fail_lambda - tc_err against the one-step band. Writes xout/a17c_read_summary.txt and xout/a17c_read.csv."""
import os, io, json, zipfile, urllib.request, urllib.error, statistics as st
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
        if t.startswith("pq:p091c-") and w["conclusion"] == "success":
            key = t[3:]
            if key not in runs: runs[key] = w["id"]
rows = []
for key, rid in sorted(runs.items()):
    arts = [a for a in get(f"https://api.github.com/repos/{REPO}/actions/runs/{rid}/artifacts")["artifacts"] if a["size_in_bytes"] < 2e7]
    for a in arts:
        z = zipfile.ZipFile(io.BytesIO(get(a["archive_download_url"], raw=True)))
        for n in z.namelist():
            if n.endswith("_conv.json"):
                c = json.loads(z.read(n)); c["set"] = "lambda test"
                rows.append(c)
lo, hi = 1 / (1 + 80.0), 1 / (1 + 47.4)
L = [f"PHY-091 amendment 3 reading: the meter on the lambda spike-in, {len(rows)} libraries. One-step band {lo:.4f}-{hi:.4f}."]
with open(os.path.join(OUT, "a17c_read.csv"), "w") as f:
    f.write("gsm,genome,lambda_reads,conv_fail_lambda,conv_fail_nonCpG,tc_err,gain_err,writer_lambda,writer_nonCpG\n")
    for c in sorted(rows, key=lambda c: c["gsm"]):
        lm = c.get("lambda_meter") or {}; cl = lm.get("conv_fail_lambda"); tc = c.get("tc_err") or 0.0
        wl = (c["gain_err"] - cl - tc) if cl is not None else None; wn = c["gain_err"] - c["conv_fail"] - tc
        f.write(f"{c['gsm']},{c['genome']},{lm.get('reads', 0)},{cl if cl is not None else ''},{c['conv_fail']:.6f},{tc:.6f},{c['gain_err']:.6f},{wl if wl is not None else ''},{wn:.6f}\n")
        L.append(f"  {c['gsm']} ({c['genome']}): lambda reads {lm.get('reads', 0)}; meter on lambda {('%.5f' % cl) if cl is not None else 'none'}, meter from non-CpG "
                 f"{c['conv_fail']:.5f}; writer with the lambda meter {('%.4f' % wl) if wl is not None else '-'} ({'in band' if wl is not None and lo <= wl <= hi else 'outside band' if wl is not None else '-'}), with the non-CpG meter {wn:.4f}")
sp = [c for c in rows if (c.get("lambda_meter") or {}).get("reads", 0) >= 1000 and c["lambda_meter"].get("conv_fail_lambda") is not None]
if sp:
    L.append(f"  spiked (>= 1,000 lambda reads), n {len(sp)}: meter on lambda median {st.median(c['lambda_meter']['conv_fail_lambda'] for c in sp):.5f}; "
             f"non-CpG median {st.median(c['conv_fail'] for c in sp):.5f}; writer (lambda meter) median {st.median(c['gain_err'] - c['lambda_meter']['conv_fail_lambda'] - (c.get('tc_err') or 0) for c in sp):.4f}")
L.append("Reading guide (exploration): lambda below non-CpG = real CpH methylation in the cells (the non-CpG route over-subtracts there); equal = the "
         "non-CpG route is the meter. The negative controls must show few lambda reads, or the mismapping guard is not enough.")
open(os.path.join(OUT, "a17c_read_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
