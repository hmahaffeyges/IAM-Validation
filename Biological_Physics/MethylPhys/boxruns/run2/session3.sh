#!/bin/bash
# Box Run 2 session 3 (doors/DEV_IAMA_XCELL_01.md): P_CD4 on Loyfer CD4, laboratory-G CD4 runs, pUC19 spike in the neutrophil BAMs.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/BOXRUN2/session3
S=/mnt/scratch/run2; O=$S/out3; mkdir -p $O $S/ref $S/loyfer $S/nbam; cd $S
PYB=/home/ubuntu/env/bin/python; HERE=$(cd "$(dirname "$0")" && pwd); CH=$HERE/../../chain
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/session3.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
get(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').download_file('$B',sys.argv[1],sys.argv[2])" "$1" "$2"; }
fin(){ log "END $1"; for f in $O/*; do [ -f "$f" ] && up "$f" "$P/$(basename $f)"; done; exit 0; }
log "STEP tools (as session 1)"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba || fin FAIL_micromamba
MM="$S/bin/micromamba -r $S/mamba"
$MM create -y -q -p $S/e_bm -c conda-forge -c bioconda python=3.6 bwameth=0.2.0 bwa > $O/install.log 2>&1 || fin FAIL_install_bwameth
$MM create -y -q -p $S/e_wgbs -c conda-forge -c bioconda python=3.8 numpy pandas scipy htslib bedtools >> $O/install.log 2>&1 || fin FAIL_install_wgbs
$MM create -y -q -p $S/e_st19 -c conda-forge -c bioconda samtools=1.9 >> $O/install.log 2>&1 || fin FAIL_install_samtools
$MM create -y -q -p $S/e_sra -c conda-forge -c bioconda sra-tools >> $O/install.log 2>&1 || fin FAIL_install_sra
$MM create -y -q -p $S/e_sbb -c conda-forge -c bioconda sambamba=0.6.5 >> $O/install.log 2>&1 || fin FAIL_install_sambamba
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_sra/bin:$S/e_sbb/bin:$PATH
command -v g++ >/dev/null || sudo apt-get install -y -q build-essential >> $O/install.log 2>&1
[ -d wgbs_tools ] || git clone -q https://github.com/nloyfer/wgbs_tools.git
(cd wgbs_tools && git checkout -q 0.1.0 && python setup.py >> $O/install.log 2>&1)
export PATH=$S/wgbs_tools:$PATH
log "STEP restore index and hg19 references from S3"
for f in hg19_lambda_puc19.fa hg19_lambda_puc19.fa.bwameth.c2t hg19_lambda_puc19.fa.bwameth.c2t.amb hg19_lambda_puc19.fa.bwameth.c2t.ann hg19_lambda_puc19.fa.bwameth.c2t.bwt hg19_lambda_puc19.fa.bwameth.c2t.pac hg19_lambda_puc19.fa.bwameth.c2t.sa; do
  [ -s ref/$f ] || get reference/hg19_bwameth/$f ref/$f || fin FAIL_restore_$f; done
samtools faidx ref/hg19_lambda_puc19.fa || fin FAIL_faidx
get results/BOXRUN2/session1/wgbstools_hg19_references.tgz wgbs_ref.tgz && mkdir -p wgbs_tools/references && tar xzf wgbs_ref.tgz -C wgbs_tools/references || fin FAIL_restore_wgbs_refs

log "STEP (c) pUC19 in the six readable neutrophil BAMs"
NB=""; for r in SRR9888330 SRR9888332 SRR9888333 SRR9888334 SRR9888336 SRR9888337; do get results/BOXRUN2/session2/bam/$r.bam nbam/$r.bam && samtools index -@ 8 nbam/$r.bam && NB="$NB nbam/$r.bam"; done
$PYB $HERE/puc19.py ref/hg19_lambda_puc19.fa $O/puc19_neutrophils.json $NB >> $O/session3.log 2>&1 || log "puc19 step failed"
rm -f nbam/*.bam nbam/*.bai

log "STEP (a) P_CD4: three whole Loyfer Blood-T-CD4 files"
G=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5652nnn
for x in GSM5652279_Blood-T-CD4-Z000000TT GSM5652280_Blood-T-CD4-Z000000U7 GSM5652281_Blood-T-CD4-Z000000UM; do
  g=${x%%_*}; [ -s loyfer/$x.pat.gz ] || curl -sfL -o loyfer/$x.pat.gz $G/$g/suppl/$x.pat.gz || log "download failed $x"; done
$PYB $HERE/eps_of.py $CH $O/loyfer_cd4_eps.json loyfer/*.pat.gz >> $O/session3.log 2>&1 || log "loyfer eps failed"

log "STEP (b) laboratory-G CD4 runs through session 2's pipeline"
RUNS="SRR9888326 SRR9888328 SRR9888329 SRR9888338 SRR9888340 SRR9888341" OUTP=$P OUTD=out3cd4 N_PAIRS=${N_PAIRS:-25000000} THREADS=${THREADS:-120} \
  SKIP_FORMAT=1 bash $HERE/session2.sh >> $O/session2_cd4.log 2>&1
log "session 2 pipeline exit $?"
fin DONE
