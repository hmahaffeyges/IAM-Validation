#!/usr/bin/env python3
"""Read the derived line, the loss spread and the islands and clock-site counts from finished train jobs' bundles ON A GITHUB MACHINE (the bundles
are each run's own artifact; nothing is downloaded to the laptop). Usage: bundle_fields.py RUN_ID [RUN_ID ...]. Prints one notice line per run:
TRAINFIELDS {json}."""
import os, sys, io, json, zipfile, requests
R = "https://api.github.com/repos/" + os.environ.get("GITHUB_REPOSITORY", "hmahaffeyges/IAM-Validation")
H = {"Authorization": "Bearer " + os.environ.get("GH_TOKEN") or os.environ["GITHUB_TOKEN"], "Accept": "application/vnd.github+json"}
def small(v, n=400):
    s = json.dumps(v, default=str); return v if len(s) <= n else s[:n]
for run in sys.argv[1:]:
    try:
        a = requests.get(f"{R}/actions/runs/{run}/artifacts", headers=H, timeout=60).json()["artifacts"][0]
        z = zipfile.ZipFile(io.BytesIO(requests.get(a["archive_download_url"], headers=H, timeout=120).content))
        n = [x for x in z.namelist() if x.endswith("_bundle.json")][0]; B = json.loads(z.read(n)); ia = B.get("iam_a", {}); c = ia.get("count", {})
        out = dict(run=run, gsm=n.split("joint_")[1].split("_bundle")[0], **{k: ia.get(k) for k in ("A", "A_derived", "predicted_derived", "firm_derived")},
                   loss_dispersion=c.get("loss_dispersion"), loss_dispersion_molecules=c.get("loss_dispersion_molecules"), sites=small(ia.get("sites"), 1500))
    except Exception as e:
        out = dict(run=run, error=str(e)[:200])
    s = json.dumps(out, default=str)
    for i in range(0, len(s), 1900): print(f"::notice::TRAINFIELDS {run} part {i // 1900 + 1}: " + s[i:i + 1900])
