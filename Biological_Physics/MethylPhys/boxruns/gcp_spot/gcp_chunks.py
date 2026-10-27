"""gcp_chunks.py - the split alignment's chunks on Google Cloud spot machines, beside GitHub's (DEVELOPMENT; written 2026-10-27 for the
reclaim test; nothing has run on it yet).

Each chunk is one spot VM. Its startup script does what one split_align.yml chunk job does (the same erase_gh.sh with CHUNK=k/N, the same
index release, the same requirements and preflight), then copies the chunk's molecule file, run log and exit code to a Cloud Storage bucket
and deletes itself. A chunk whose result is in the bucket is never sent again; a chunk whose VM was reclaimed has no result and is sent again
by the next `launch`. So a reclaim loses only the chunk that was running.

Usage (after `gcloud auth activate-service-account --key-file KEY.json` and `gcloud config set project PROJECT`):
  python3 gcp_chunks.py launch  GSM SAMPLES.csv N BUCKET [--zone us-central1-a] [--type e2-standard-4] [--only 0,4,8]
  python3 gcp_chunks.py status  GSM N BUCKET           # which chunks are done, running, reclaimed or failed
  python3 gcp_chunks.py fetch   GSM N BUCKET DIR       # every finished chunk's files, for the merge
The reclaim test: `launch` one sample's 20 chunks, `status` every 10 minutes, and the ledger (gcp_ledger.csv) records each VM's start and the
state it replaced (reclaimed or not started), so reclaims and cost (machine type x hours at that day's spot price) are measured."""
import sys, os, json, subprocess, csv, datetime as dt

REPO = "https://github.com/hmahaffeyges/IAM-Validation"
HERE = os.path.dirname(os.path.abspath(__file__)); LEDGER = os.path.join(HERE, "gcp_ledger.csv")

STARTUP = r"""#!/bin/bash
# one chunk of the split alignment on a spot VM (gcp_chunks.py); results to gs://{bucket}/{gsm}/c{k}_{n}/
set -u; exec > /var/log/iam_chunk.log 2>&1
B=gs://{bucket}/{gsm}/c{k}_{n}
done_up() {{ gsutil -q cp /root/rc.txt /root/run.log $B/ ; gsutil -q cp /root/eout/*.pat.gz $B/ 2>/dev/null; gsutil -q cp /root/eout/*timing.json $B/ 2>/dev/null; }}
trap 'echo reclaimed > /root/rc.txt; gsutil -q cp /root/rc.txt $B/rc.txt' TERM
apt-get -qq update && apt-get -qq install -y git python3.11 python3.11-venv python3-pip > /dev/null
cd /root && git clone -q --depth 1 {repo}.git repo && cd repo
python3.11 -m venv /root/v && . /root/v/bin/activate && pip install -q -r Biological_Physics/MethylPhys/chain/requirements.txt
python3 Biological_Physics/MethylPhys/chain/preflight.py || {{ echo 1 > /root/rc.txt; done_up; shutdown -h now; }}
fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile > /dev/null && swapon /swapfile
export INDEX_TAG=hg19-bwameth-index-v1 INDEX_TAG_MM10=mm10-bwameth-index-v1 GITHUB_REPOSITORY=hmahaffeyges/IAM-Validation
export MP=$PWD/Biological_Physics/MethylPhys WORK=/root/ew OUT=/root/eout NW={threads} CHUNK={k}/{n} SAMPLES={samples} FULL=0
bash $MP/doors/data/DEV_ERASE_01/erase_gh.sh {gsm} > /root/run.log 2>&1; echo $? > /root/rc.txt
done_up; gcloud compute instances delete $(hostname) --zone {zone} --quiet
"""

def sh(cmd, check=True):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and p.returncode: raise RuntimeError(f"{cmd[:120]}: {p.stderr[-400:]}")
    return p.stdout

def vm(gsm, k, n): return f"iam-{gsm.lower()}-c{k}-{n}"

def state(gsm, n, bucket):
    have = sh(f"gsutil ls gs://{bucket}/{gsm}/ 2>/dev/null || true", check=False)
    running = set(sh("gcloud compute instances list --format='value(name)' --filter='name~^iam-'", check=False).split())
    S = {}
    for k in range(n):
        p = f"gs://{bucket}/{gsm}/c{k}_{n}/"
        if vm(gsm, k, n) in running: S[k] = "running"
        elif p in have:
            rc = sh(f"gsutil cat {p}rc.txt 2>/dev/null || echo none", check=False).strip()
            S[k] = "done" if rc == "0" else ("reclaimed" if rc == "reclaimed" else f"failed rc {rc}")
        else: S[k] = "not started"
    return S

def launch(gsm, samples, n, bucket, zone="us-central1-a", mtype="e2-standard-4", only=None, threads=4):
    S = state(gsm, n, bucket); sent = []
    for k in range(n):
        if only and k not in only: continue
        if S[k] in ("done", "running"): continue
        f = os.path.join(HERE, f".startup_{gsm}_{k}.sh")
        open(f, "w").write(STARTUP.format(bucket=bucket, gsm=gsm, k=k, n=n, zone=zone, repo=REPO, samples=samples, threads=threads))
        sh(f"gcloud compute instances create {vm(gsm, k, n)} --zone {zone} --machine-type {mtype} --provisioning-model SPOT "
           f"--instance-termination-action DELETE --boot-disk-size 100GB --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud "
           f"--scopes cloud-platform --metadata-from-file startup-script={f} --quiet")
        os.remove(f); sent.append(k)
        new = not os.path.exists(LEDGER)
        with open(LEDGER, "a", newline="") as h:
            w = csv.writer(h)
            if new: w.writerow(["vm", "gsm", "chunk", "machine_type", "zone", "started_utc", "previous_state"])
            w.writerow([vm(gsm, k, n), gsm, f"{k}/{n}", mtype, zone, dt.datetime.utcnow().isoformat(timespec="seconds"), S[k]])
    print("sent", sent)

def fetch(gsm, n, bucket, d):
    os.makedirs(d, exist_ok=True); sh(f"gsutil -m -q cp -r gs://{bucket}/{gsm} {d}/"); print(json.dumps(state(gsm, n, bucket)))

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: print(__doc__); raise SystemExit
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    if a[0] == "launch":
        only = [int(x) for x in opt("--only").split(",")] if opt("--only") else None
        launch(a[1], a[2], int(a[3]), a[4], opt("--zone", "us-central1-a"), opt("--type", "e2-standard-4"), only)
    elif a[0] == "status": print(json.dumps(state(a[1], int(a[2]), a[3]), indent=1))
    elif a[0] == "fetch": fetch(a[1], int(a[2]), a[3], a[4])
