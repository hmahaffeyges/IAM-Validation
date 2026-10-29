"""Timed test of the sequencing atlas's real step on Google spot machines: one healthy input (spent) read by the chain exactly as GitHub's
geo_pat.yml reads it (erase_gh.sh with PATFILE and FULL=1, the one command run_sample.py). Each machine records its phase times, uploads its
results to the bucket, and deletes itself; Google deletes it anyway at a 2-hour limit. Every launch is written to gcp_ledger.csv.
Usage (python kernel, with S an authorized session and P the project): exec(open('gcp_test.py').read()); launch(name, zone, machine, price)"""
import json, time, csv, os, datetime
BUCKET = f"{P}-iam-results"; GSM = "GSM5652317"; SPEC = "sorted b cells"; MAXS = 7200
SAMPLES = "Biological_Physics/MethylPhys/doors/data/DEV_ALL_INFO_01/coupling_02/healthy_seq.csv"
STARTUP = r'''#!/bin/bash
set -u
B=gs://__BUCKET__/timing_test/$(hostname); W=/root/w; mkdir -p $W; L=$W/progress.log
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
MT=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type | awk -F/ '{print $NF}')
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; gcloud storage cp $L $B/progress.log -q > /dev/null 2>&1; }
finish(){ log "finish rc $(cat $W/rc.txt 2>/dev/null)"; gcloud storage cp $W/timing.json $W/run.log $W/rc.txt $B/ -q > /dev/null 2>&1
  [ -d $W/eout ] && gcloud storage cp -r $W/eout $B/ -q > /dev/null 2>&1; log "uploaded; deleting myself"; gcloud compute instances delete $(hostname) --zone=$ZONE -q; }
trap finish EXIT
T0=$(date +%s); log "start $MT $ZONE"
apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq git curl bzip2 python3-venv python3-pip > /dev/null 2>&1; T1=$(date +%s); log "packages"
cd $W && git clone -q --depth 1 https://github.com/hmahaffeyges/IAM-Validation.git repo; T2=$(date +%s); log "repo $(git -C repo rev-parse --short HEAD) $(du -sh repo | cut -f1)"
python3 -m venv /root/venv && . /root/venv/bin/activate && pip install -q -r repo/Biological_Physics/MethylPhys/chain/requirements.txt > /dev/null 2>&1 && python3 repo/Biological_Physics/MethylPhys/chain/preflight.py >> $L 2>&1 || { echo 2 > $W/rc.txt; exit 1; }
T3=$(date +%s); log "python ready $(python3 --version)"
G=__GSM__; U="https://ftp.ncbi.nlm.nih.gov/geo/samples/${G:0:7}nnn/$G/suppl/"
F=$(curl -sfL --retry 5 "$U" | grep -o 'href="[^"]*\.pat\.gz"' | sed 's/href="//; s/"$//' | grep -v '\.hg38\.' | head -1)
curl -sfL --retry 5 -o $W/$G.pat.gz "$U$F"; T4=$(date +%s); log "molecules $(du -h $W/$G.pat.gz | cut -f1)"
cd $W/repo && export SPECIMEN="__SPEC__" INDEX_TAG=hg19-bwameth-index-v1 GITHUB_REPOSITORY=hmahaffeyges/IAM-Validation MP=$PWD/Biological_Physics/MethylPhys WORK=/root/ew OUT=$W/eout NW=4 FULL=1 PATFILE=$W/$G.pat.gz SAMPLES=__SAMPLES__
bash $MP/doors/data/DEV_ERASE_01/erase_gh.sh $G > $W/run.log 2>&1; echo $? > $W/rc.txt; T5=$(date +%s); log "chain read rc $(cat $W/rc.txt)"
python3 -c "import json,sys; t=list(map(int,sys.argv[3:])); json.dump(dict(machine=sys.argv[1], zone=sys.argv[2], packages_s=t[1]-t[0], repo_s=t[2]-t[1], python_s=t[3]-t[2], download_s=t[4]-t[3], chain_read_s=t[5]-t[4], total_s=t[5]-t[0]), open('$W/timing.json','w'), indent=1)" $MT $ZONE $T0 $T1 $T2 $T3 $T4 $T5
'''.replace("__BUCKET__", BUCKET).replace("__GSM__", GSM).replace("__SPEC__", SPEC).replace("__SAMPLES__", SAMPLES)
SA = f"iam-runner@{P}.iam.gserviceaccount.com"
def launch(name, zone, machine, price_h, purpose=None):
    import glob as _g   # price guard (2026-10-29): refuse a launch above twice the morning's price for that machine and region
    _f = sorted(_g.glob("gcp_prices_*.csv"))
    if not _f: return 0, "no price check today: run gcp_prices.py first"
    _T = pd.read_csv(_f[-1]); _m = _T[(_T.machine == machine) & (_T.region == zone.rsplit("-", 1)[0])]
    if _m.empty or not bool(_m.offered.iloc[0]): return 0, f"{machine} not offered in {zone} by the last price check"
    if price_h > 2 * float(_m.spot.iloc[0]): return 0, f"refused: {price_h} is above twice the checked price {float(_m.spot.iloc[0])}"
    body = dict(name=name, machineType=f"zones/{zone}/machineTypes/{machine}",
        disks=[dict(boot=True, autoDelete=True, initializeParams=dict(sourceImage="projects/debian-cloud/global/images/family/debian-12", diskSizeGb="50", diskType=f"zones/{zone}/diskTypes/pd-balanced"))],
        networkInterfaces=[dict(network="global/networks/default", accessConfigs=[dict(type="ONE_TO_ONE_NAT", name="external")])],
        serviceAccounts=[dict(email=SA, scopes=["https://www.googleapis.com/auth/cloud-platform"])],
        scheduling=dict(provisioningModel="SPOT", instanceTerminationAction="DELETE", maxRunDuration=dict(seconds=str(MAXS)), automaticRestart=False, onHostMaintenance="TERMINATE"),
        metadata=dict(items=[dict(key="startup-script", value=STARTUP)] + ([dict(key="shutdown-script", value=globals()["SHUTDOWN_SCRIPT"])] if globals().get("SHUTDOWN_SCRIPT") else [])),
        labels=dict(purpose="iam-timing-test"))
    r = S.post(f"https://compute.googleapis.com/compute/v1/projects/{P}/zones/{zone}/instances", json=body)
    new = not os.path.exists("gcp_ledger.csv")
    with open("gcp_ledger.csv", "a", newline="") as h:
        w = csv.writer(h)
        if new: w.writerow(["name", "zone", "machine", "spot_usd_per_h", "launched_utc", "max_s", "status", "purpose"])
        w.writerow([name, zone, machine, price_h, datetime.datetime.utcnow().isoformat(timespec="seconds"), MAXS, r.status_code, purpose or f"timing test: {GSM} read by the chain"])
    return r.status_code, (r.json().get("error") or {}).get("message", "")[:300] if not r.ok else "launched"
