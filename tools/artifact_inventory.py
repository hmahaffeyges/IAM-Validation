#!/usr/bin/env python3
"""Inventory of kept run artifacts (2026-11-15). Lists every artifact named erase-<GSM> (molecule files from erase_test.yml runs keep the
<GSM>.pat.gz in it), newest per library, with size and expiry; writes xout/artifact_inventory.csv and prints a short summary as notices.
Usage (explore.yml): tools/artifact_inventory.py [GSM prefix filter]"""
import os, sys, csv, requests
H = {"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"}; R = "https://api.github.com/repos/" + os.environ["GITHUB_REPOSITORY"]
flt = sys.argv[1] if len(sys.argv) > 1 else ""
rows, page = [], 1
while True:
    A = requests.get(f"{R}/actions/artifacts", headers=H, params={"per_page": 100, "page": page}, timeout=120).json().get("artifacts", [])
    if not A: break
    for a in A:
        if a["name"].startswith("erase-" + flt) and not a["expired"]:
            rows.append(dict(gsm=a["name"][6:], mb=round(a["size_in_bytes"] / 1e6, 1), created=a["created_at"][:10], expires=a["expires_at"][:10], run=a["workflow_run"]["id"]))
    page += 1
os.makedirs("xout", exist_ok=True)
with open("xout/artifact_inventory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["gsm", "mb", "created", "expires", "run"]); w.writeheader(); [w.writerow(r) for r in rows]
big = {}
for r in rows:
    if r["mb"] >= 1.5 and (r["gsm"] not in big or r["mb"] > big[r["gsm"]]["mb"]): big[r["gsm"]] = r
first = min((r["expires"] for r in big.values()), default="-")
print(f"::notice::{len(rows)} erase artifacts kept; {len(big)} libraries with a molecule-sized artifact (>= 1.5 MB); earliest expiry {first}")
for g in sorted(big)[:40]: print(f"::notice::{g} {big[g]['mb']} MB run {big[g]['run']} expires {big[g]['expires']}")
