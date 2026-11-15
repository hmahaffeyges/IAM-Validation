"""PHY-091 reading (exploration; register A17 -> A15), the last pipeline item after every p091 library. Reads only chain outputs: each run's
<GSM>_conv.json (run_sample.py --bam --dev-count-only, chain Stage Q2), which carries, from the SAME aligned file and depth:
  gain_err  the stray-mark rate on molecules held unmarked (a lone methylated call between two unmethylated calls), the unmarked side's write error
            as read, INCLUDING the meter's own false marks;
  conv_fail the share of non-CpG cytosines read as methylated (incomplete bisulfite conversion, each library's known-zero channel);
  sub_err   the A/T substitution rate (sequencing error).
Where IAM's Law is: A15 reads the unmarked side as one equilibrium selection step by the writer, stray = 1/(1+D), D measured 47.4-80 (band
0.0123-0.0207; one proofreading step would give 0.00015-0.00043). The meter's false marks at an unmethylated CpG add about conv_fail (a C left
unconverted reads as methylated), so the writer's own error is gain_err - conv_fail; the sequencing term (a T misread as C, at most sub_err) is
reported as a second, upper correction. Exploration: no seal; the corrected rates against the band are read for what they say.
Writes xout/a17_read_summary.txt and xout/a17_read.csv."""
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
        if t.startswith("pq:p091b-") and w["conclusion"] == "success":
            key = t[3:]
            if key not in runs: runs[key] = w["id"]
rows = []
for key, rid in sorted(runs.items()):
    arts = [a for a in get(f"https://api.github.com/repos/{REPO}/actions/runs/{rid}/artifacts")["artifacts"] if a["size_in_bytes"] < 2e7]
    for a in arts:
        z = zipfile.ZipFile(io.BytesIO(get(a["archive_download_url"], raw=True)))
        for n in z.namelist():
            if n.endswith("_conv.json"):
                c = json.loads(z.read(n)); c["set"] = "referee blood (GSE173787)" if key.startswith("p091r-") else "erase libraries"
                rows.append(c)
lo, hi = 1 / (1 + 80.0), 1 / (1 + 47.4); plo, phi_ = lo ** 2, hi ** 2
L = [f"PHY-091 amendment 2 reading (A17 -> A15), {len(rows)} eraser libraries with conv_fail and tc_err. Single selection band {lo:.4f}-{hi:.4f}; one proofreading step {plo:.5f}-{phi_:.5f}."]
with open(os.path.join(OUT, "a17b_read.csv"), "w") as f:
    f.write("gsm,set,genome,gain_err,conv_fail,sub_err,tc_err,corrected,corrected_tc,corrected_upper,gain_opp\n")
    for c in rows:
        c["tc_err"] = c.get("tc_err") or 0.0
        c["corr"] = c["gain_err"] - c["conv_fail"]; c["corr_tc"] = c["gain_err"] - c["conv_fail"] - c["tc_err"]; c["corr_up"] = c["gain_err"] - c["conv_fail"] - c["sub_err"]
        f.write(f"{c['gsm']},{c['set']},{c['genome']},{c['gain_err']:.6f},{c['conv_fail']:.6f},{c['sub_err']:.6f},{c['tc_err']:.6f},{c['corr']:.6f},{c['corr_tc']:.6f},{c['corr_up']:.6f},{c.get('gain_opp')}\n")
for s in sorted({c["set"] for c in rows}):
    g = [c for c in rows if c["set"] == s]
    if not g: continue
    for lab, k in (("as read (gain_err)", "gain_err"), ("minus conversion failure", "corr"), ("minus conversion and tc_err (the measured sequencing false marks)", "corr_tc"), ("minus conversion and sub_err (old upper bound)", "corr_up")):
        v = sorted(c[k] for c in g); inside = sum(lo <= x <= hi for x in v) / len(v)
        L.append(f"  {s}, n {len(v)}, {lab}: median {st.median(v):.4f} (range {v[0]:.4f}-{v[-1]:.4f}); inside the single-selection band {100 * inside:.0f} %; "
                 f"median / proofread level {st.median(v) / phi_:.0f}x")
    L.append(f"  {s}: conv_fail median {st.median(c['conv_fail'] for c in g):.4f}, sub_err median {st.median(c['sub_err'] for c in g):.4f}, tc_err median {st.median(c['tc_err'] for c in g):.5f}")
L.append("Reading guide (exploration): if the corrected rates stay in the band and far above the proofread level, the unmarked side is the writer's one-step "
         "selection error with the meter's false marks removed (A15 from 'accounted after the fact' toward a derived reading); if they fall below the band, "
         "part of what A15 read was the meter.")
open(os.path.join(OUT, "a17b_read_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
