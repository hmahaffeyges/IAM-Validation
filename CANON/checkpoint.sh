#!/usr/bin/env bash
# Checkpoint (author, 2026-10-14: "push to the repo every so often ... so nothing is missed"). One command, run after every step of the
# atlas work and on a timer while box jobs run: verifies every registered box job's code, log and outputs in S3, regenerates the run
# ledger, the readiness board, the release checklist, the manifest and the day log, commits whatever changed and pushes THROUGH the gate.
# Nothing here bypasses checked_push.sh; a refusal is printed and nothing is pushed.  Usage: bash CANON/checkpoint.sh "what just happened"
cd "$(dirname "$0")/.." || exit 1
step(){ "$@" > /tmp/checkpoint_step.txt 2>&1 || { echo "checkpoint: STEP FAILED: $*"; tail -5 /tmp/checkpoint_step.txt; exit 1; }; }
git pull -q --ff-only origin main 2>/dev/null || true
# verify first (it records what S3 holds; its refusals are re-checked by the gate after the ledger is regenerated)
python3 CANON/box_code_check.py --verify > /dev/null 2>&1 || true
step python3 CANON/chain_gate.py register
step python3 CANON/not_chain_verified.py
step python3 CANON/replicate.py
step python3 Biological_Physics/MethylPhys/boxruns/make_runs_ledger.py
step python3 CANON/box_code_check.py
python3 CANON/fit_progress.py > /dev/null 2>&1 || true
step python3 CANON/v21_readiness.py; step python3 CANON/v21_release.py --verify
step python3 CANON/ascore_db.py; step python3 CANON/instruments_board.py;
step python3 CANON/book_toolbox.py;
step python3 CANON/book_bits.py; python3 CANON/bit_tables.py; python3 CANON/four_domains.py
step python3 CANON/commissioning_report.py
python3 CANON/costs_snapshot.py > /dev/null 2>&1 || echo "checkpoint: costs not refreshed (no AWS billing access in this shell); the page keeps the last snapshot"
step python3 CANON/catalogue.py; step python3 CANON/status_md.py; step python3 CANON/status_lines.py; step python3 CANON/plan_md.py; python3 CANON/now_running.py > /dev/null 2>&1 || true
step python3 CANON/four_domains.py; python3 CANON/bit_tables.py; step python3 CANON/chain_events.py; step python3 CANON/manifest.py; step python3 CANON/master_inventory.py; python3 CANON/daylog.py > /dev/null 2>&1 || true
git add -A --sparse .
if git diff --cached --quiet; then echo "checkpoint: nothing changed"; exit 0; fi
git -c user.name="Heath W. Mahaffey" -c user.email="hmahaffeyges@users.noreply.github.com" commit -q -m "Checkpoint: ${1:-records regenerated} (CANON/checkpoint.sh)"
GIT_TERMINAL_PROMPT=0 bash CANON/checked_push.sh -q origin HEAD:main 2>&1 | grep -E "check: |REFUSED|FAILED|untracked" || true
git fetch -q; echo "checkpoint: origin at $(git log --oneline -1 origin/main | cut -c1-8)"
