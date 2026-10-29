"""IAM Sequencing Atlas on Google spot machines (plan 35.30). Each machine builds a list of blocks of the hg19 CpG numbering by the one command
(run_sample.py --build seq-atlas), in pieces of PIECE CpGs; each piece is uploaded to the Toronto bucket the moment it is built, and a piece
already there is skipped, so a machine sent after a reclaimed one finishes only what is missing. On Google's reclaim warning the shutdown
script uploads the machine's log. Each machine deletes itself when done; Google deletes it anyway at the time limit. Every launch goes through
the price guard (gcp_test.launch) and into gcp_ledger.csv.
Usage (python kernel, with S and P): exec(open('gcp_test.py').read()); exec(open('gcp_seqatlas.py').read()); launch_machine(m, blocks)"""
BLOCK = 1_000_000; NBLOCKS = 29; PIECE = 250_000; HG19 = 28217448
SBUCKET = f"{P}-iam-ca-toronto"; ZONE = "northamerica-northeast2-a"
INPUTS = "Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv"
BSTART = r'''#!/bin/bash
set -u
M=__M__; KS="__KS__"; B=gs://__BUCKET__/seq_atlas; W=/root/w; mkdir -p $W; L=$W/machine_$M.log
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
log(){ echo "$(date -u +%H:%M:%S) $*" >> $L; gcloud storage cp $L $B/logs/machine_$M.log -q > /dev/null 2>&1; }
finish(){ log "finish rc $(cat $W/rc.txt 2>/dev/null)"; gcloud storage cp $W/machine_$M.timing.json $B/logs/ -q > /dev/null 2>&1
  log "deleting myself"; touch $W/self_delete; gcloud compute instances delete $(hostname) --zone=$ZONE -q; }
trap finish EXIT
T0=$(date +%s); log "start machine $M, blocks $KS"
apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq git python3-venv python3-pip > /dev/null 2>&1
cd $W && git clone -q --depth 1 https://github.com/hmahaffeyges/IAM-Validation.git repo && log "repo $(git -C repo rev-parse --short HEAD)"
python3 -m venv /root/venv && . /root/venv/bin/activate && pip install -q -r repo/Biological_Physics/MethylPhys/chain/requirements.txt > /dev/null 2>&1 || { echo 2 > $W/rc.txt; exit 1; }
T1=$(date +%s); log "python ready"; echo 0 > $W/rc.txt; NP=0
cd repo/Biological_Physics/MethylPhys/chain/MethylPhys_Interface
for K in $KS; do
  S0=$((K * __BLOCK__)); S1=$(( (K + 1) * __BLOCK__ )); [ $S1 -gt __HG19__ ] && S1=__HG19__
  for P0 in $(seq $S0 __PIECE__ $((S1 - 1))); do
    P1=$((P0 + __PIECE__)); [ $P1 -gt $S1 ] && P1=$S1; N=piece_$(printf %08d $P0)_$(printf %08d $P1)
    gcloud storage ls $B/pieces/$N.parquet > /dev/null 2>&1 && { log "$N already built; skipped"; continue; }
    if python3 run_sample.py --build seq-atlas --build-samples $W/repo/__INPUTS__ --build-block $P0:$P1 --build-out $W/$N.parquet --bundle $W/$N.json > $W/$N.log 2>&1 \
       && gcloud storage cp $W/$N.json $B/pieces/ -q > /dev/null 2>&1 && gcloud storage cp $W/$N.parquet $B/pieces/ -q > /dev/null 2>&1; then
      NP=$((NP + 1)); log "$N built and uploaded"
    else log "$N FAILED: $(tail -2 $W/$N.log | tr '\n' ' ')"; echo 1 > $W/rc.txt; fi
    rm -f $W/$N.parquet
  done
done
T2=$(date +%s); log "machine done, $NP pieces built, rc $(cat $W/rc.txt)"
python3 -c "import json,sys; t=list(map(int,sys.argv[3:])); json.dump(dict(machine=int(sys.argv[1]), pieces=int(sys.argv[2]), setup_s=t[1]-t[0], build_s=t[2]-t[1], total_s=t[2]-t[0]), open('$W/machine_$M.timing.json','w'), indent=1)" $M $NP $T0 $T1 $T2
'''
SHUTDOWN = r'''#!/bin/bash
# Google's warning before it takes the machine back: the log goes up so the loss is visible (2026-10-29)
[ -e /root/w/self_delete ] && exit 0   # its own delete after finishing, not a reclaim
for f in /root/w/machine_*.log; do echo "$(date -u +%H:%M:%S) machine reclaimed by Google" >> $f; gcloud storage cp $f gs://__BUCKET__/seq_atlas/logs/ -q; done
'''
def launch_machine(m, blocks, zone=ZONE, machine="n2-standard-4", price_h=0.0192, maxs=10800):
    global STARTUP, MAXS, SHUTDOWN_SCRIPT
    STARTUP = (BSTART.replace("__M__", str(m)).replace("__KS__", " ".join(map(str, blocks))).replace("__BLOCK__", str(BLOCK)).replace("__HG19__", str(HG19))
               .replace("__BUCKET__", SBUCKET).replace("__INPUTS__", INPUTS).replace("__PIECE__", str(PIECE)))
    SHUTDOWN_SCRIPT = SHUTDOWN.replace("__BUCKET__", SBUCKET); MAXS = maxs
    return launch(f"iam-seqatlas-m{m}", zone, machine, price_h, purpose=f"IAM Sequencing Atlas blocks {blocks}")
