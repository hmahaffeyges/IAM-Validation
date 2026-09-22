#!/bin/sh
# Re-run the H_min calibration and print the posterior floors.
# Takes about 15 seconds: 32 walkers x (500 burn-in + 5,000 production) x 5 chains over 37 reference cells.
set -e
python3 -m pip install -r requirements.txt
cd samplers
python3 gape_mcmc_g002.py | tee ../reproduction/g002_rerun_$(date +%Y-%m-%d).log
echo
echo "Compare the _H_MIN_REGISTRY_POSTERIOR block against reproduction/REPRODUCTION.md."
echo "The script's own 'All consistent' line compares with PRE-calibration values - read REPRODUCTION.md first."
