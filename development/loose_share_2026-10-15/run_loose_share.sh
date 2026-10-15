#!/bin/bash
# Two tests, one job (2026-10-15). 1) arch_cut.py at CUT=0.70 must reproduce DEV-ARCH-01's loose shares exactly (reproduction check),
# then at CUT=2/3 (the one-Landauer-bit cut). 2) own_reads_loose.py on each Loyfer sample's own molecules in the 399 windows.
set -uo pipefail
P=~/env/bin/python
CUT=0.70 $P arch_cut.py > arch_070.log 2>&1 && CUT=0.6666666667 $P arch_cut.py > arch_0667.log 2>&1; echo "arch rc $?"
$P own_reads_loose.py real /home/ubuntu/data/lines/loyfer_pat_windows windows_hg19_cpgidx.bed > own.log 2>&1; echo "own rc $?"
for f in arch_01_cut*.csv own_reads_loose.csv arch_070.log arch_0667.log own.log; do $P -c "import boto3;boto3.client('s3').upload_file('$f','methylphys-data-945451304272-us-west-2-an','results/LOOSE_SHARE_2026-10-15/$f')"; done
ls arch_01_cut*.csv own_reads_loose.csv
