#!/usr/bin/env python3
"""Read the end of a finished run's logs ON A GITHUB MACHINE (2026-11-15; author: no result downloads over the hotspot). Downloads the run's
artifacts here on the runner, finds the chain's *_run_sample.log and the workflow's job log lines with 'exit' / 'REFUSED' / 'Error', and prints
the last lines as notices (a few KB reach the owner's session). Usage (explore.yml): tools/run_log_tail.py <run id> [lines, default 12]"""
import os, sys, io, zipfile, requests
H = {"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"}; R = "https://api.github.com/repos/" + os.environ["GITHUB_REPOSITORY"]
rid = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
for a in requests.get(f"{R}/actions/runs/{rid}/artifacts", headers=H, timeout=60).json().get("artifacts", []):
    z = zipfile.ZipFile(io.BytesIO(requests.get(a["archive_download_url"], headers=H, timeout=600).content))
    for m in z.namelist():
        if m.endswith(("_run_sample.log", "job_exit.txt", "exit_code.txt")):
            for l in z.read(m).decode(errors="replace").splitlines()[-n:]: print(f"::notice::{os.path.basename(m)}: {l[:300]}")
j = requests.get(f"{R}/actions/runs/{rid}/jobs", headers=H, timeout=60).json()["jobs"][0]
t = requests.get(f"{R}/actions/jobs/{j['id']}/logs", headers=H, timeout=120).text.splitlines()
k = [l[29:] for l in t if "36;1m" not in l and any(x in l for x in ("job exit", "exit 3", "REFUSED", "Error", "Killed", "No space", "timed out", "GENOME_COVERAGE"))]
for l in k[-n:]: print(f"::notice::job: {l[:300]}")
