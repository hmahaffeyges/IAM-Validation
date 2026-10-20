#!/bin/bash
# Push only if the canon check passes. Use this (or MethylPhys/chain/guarded_push.sh) for every push; a plain `git push` skips the check.
# temp files of every check go to one folder that is removed when the gate exits (2026-10-17: leftover check folders filled the laptop disk)
_GATE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/gate.XXXXXX")"; export TMPDIR="$_GATE_TMP"; trap 'rm -rf "$_GATE_TMP"' EXIT
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
python3 CANON/canon_check.py || { echo "checked_push.sh: CANON CHECK FAILED - push refused"; exit 1; }
git fetch -q origin main
python3 CANON/repro_check.py || { echo "checked_push.sh: REPRODUCIBILITY CHECK FAILED - push refused"; exit 1; }
python3 CANON/status_check.py || { echo "checked_push.sh: STATUS CHECK FAILED [failed: CANON/status_check.py; each finding is printed above with its check name in brackets] - push refused"; exit 1; }
python3 CANON/book_toolbox.py --check || { echo "checked_push.sh: BOOK TOOLBOX CHECK FAILED - run python3 CANON/book_toolbox.py - push refused"; exit 1; }
python3 CANON/chain_gate_controls.py > /dev/null || { echo "checked_push.sh: CHAIN GATE CONTROLS FAILED - CANON/chain_gate_controls.py (the gate's own planted cases) did not pass; its output is above - push refused"; exit 1; }
python3 CANON/chain_gate.py check || { echo "checked_push.sh: CHAIN GATE FAILED - CANON/chain_gate.py: a logged test did not run through the chain (rule 23b); the file and rule are named above - push refused"; exit 1; }
python3 CANON/replicate.py --check || { echo "checked_push.sh: REPLICATE.md OUT OF DATE - run python3 CANON/replicate.py and commit the result - push refused"; exit 1; }
python3 CANON/gauge_words.py controls > /dev/null || { echo "checked_push.sh: GAUGE WORDS CHECK FAILED - the gauge reads writing: the floor is the least writing an X cell can run on; cancer moves right and has not lost its identity [failed: CANON/gauge_words.py; the check's own planted cases did not pass] - push refused"; exit 1; }
python3 CANON/gauge_words.py check || { echo "checked_push.sh: GAUGE WORDS CHECK FAILED - the gauge reads writing: the floor is the least writing an X cell can run on; cancer moves right and has not lost its identity [failed: CANON/gauge_words.py; its findings are printed above] - push refused"; exit 1; }
python3 CANON/commissioning_check.py || { echo "checked_push.sh: COMMISSIONING CHECK FAILED - a commissioned instrument is missing from a required document, or a stale statement remains - push refused"; exit 1; }
python3 CANON/ascore_db.py --check || { echo "checked_push.sh: A-SCORE DATABASE CHECK FAILED - run python3 CANON/ascore_db.py - push refused"; exit 1; }
python3 CANON/bit_tables.py --check || { echo "checked_push.sh: BIT TABLES differ from CANON/four_domains.json - push refused"; exit 1; }
python3 CANON/manifest_fresh_controls.py > /dev/null || { echo "checked_push.sh: MANIFEST FRESHNESS CHECK FAILED - work changed without its Master Testing Manifest row [failed: CANON/manifest_fresh_controls.py; the check's own planted cases did not pass] - push refused"; exit 1; }
python3 CANON/manifest_fresh.py check || { echo "checked_push.sh: MANIFEST FRESHNESS CHECK FAILED - work changed without its Master Testing Manifest row [failed: CANON/manifest_fresh.py; its findings are printed above] - push refused"; exit 1; }
python3 CANON/status_md.py --check || { echo "checked_push.sh: STATUS.md or Biological_Physics/MethylPhys/COMMISSIONING_PLAN.md differs from its sources [failed: CANON/status_md.py; its findings are printed above] - push refused"; exit 1; }
python3 CANON/status_lines.py --check || { echo "checked_push.sh: STATUS.md or Biological_Physics/MethylPhys/COMMISSIONING_PLAN.md differs from its sources [failed: CANON/status_lines.py; its findings are printed above] - push refused"; exit 1; }
python3 CANON/plan_md.py --check || { echo "checked_push.sh: STATUS.md or Biological_Physics/MethylPhys/COMMISSIONING_PLAN.md differs from its sources [failed: CANON/plan_md.py; its findings are printed above] - push refused"; exit 1; }
python3 CANON/kiss_check_controls.py > /dev/null || { echo "checked_push.sh: KISS CHECK FAILED - a sealed rule needs the instrument's job, at most 3 bars and public sources [failed: CANON/kiss_check_controls.py; the check's own planted cases did not pass] - push refused"; exit 1; }
python3 CANON/kiss_check.py check || { echo "checked_push.sh: KISS CHECK FAILED - a sealed rule needs the instrument's job, at most 3 bars and public sources [failed: CANON/kiss_check.py; its findings are printed above] - push refused"; exit 1; }
python3 CANON/costs_fresh.py || { echo "checked_push.sh: COSTS ON THE PAGE ARE STALE - push refused"; exit 1; }
python3 website/build/build_home.py --check || { echo "checked_push.sh: THE PAGE DOES NOT BUILD from this commit - push refused (LESSONS C22)"; exit 1; }
python3 CANON/site_build_check.py || { echo "checked_push.sh: WEBSITE BUILD IS FAILING - push refused"; exit 1; }
python3 CANON/commissioning_report.py --check || { echo "checked_push.sh: Biological_Physics/MethylPhys/COMMISSIONING_REPORT.md differs from the records - push refused"; exit 1; }
python3 CANON/earlier_results.py --check || { echo "checked_push.sh: Biological_Physics/MethylPhys/EARLIER_RESULTS.md differs from the register - push refused"; exit 1; }
python3 Biological_Physics/MethylPhys/sop/build_sop_v4.py --check || { echo "checked_push.sh: the chain v4 SOP differs from the chain's files - push refused"; exit 1; }
python3 CANON/atlas_story.py --check || { echo "checked_push.sh: ATLAS STORY (README or book) differs from CANON/atlas_story.json - push refused"; exit 1; }
python3 CANON/book_bits.py --check || { echo "checked_push.sh: BOOK BITS CHAPTER differs from CANON/four_domains.json - push refused"; exit 1; }
M_=Biological_Physics/MethylPhys; python3 CANON/relink.py check README.md CONTRIBUTING.md REFERENCE.md FOLLOW.md FOUR_DOMAINS.md MANIFEST.md Biological_Physics/README.md $M_/README.md $M_/START_HERE.md $M_/QUESTIONS_AND_ANSWERS.md $M_/RESEARCHER_PACKAGE.md $M_/COMMISSIONING_PLAN.md $M_/COMMISSIONING_REPORT.md $M_/EARLIER_RESULTS.md $M_/FIELD_NOTES.md $M_/INSTRUMENTS.md $M_/reproduce/README.md $M_/ASCORES/README.md $M_/sop/MethylPhys_CPG_SOP_v4.md $M_/STATUS.md $M_/RUNS.md || { echo "checked_push.sh: A DOCUMENT HAS A BROKEN LINK (links resolve from the document's own folder, as on GitHub) - push refused"; exit 1; }
python3 CANON/number_trace.py --check || { echo "REFUSED: a public document prints a number no measurement record holds (CANON/number_trace.py)"; exit 1; }
python3 CANON/canon_tree.py HEAD || { echo "REFUSED: a retired value in the committed tree (folders this checkout may not hold)"; exit 1; }
python3 CANON/data_homes.py --check || { echo "REFUSED: an input of the chain or a reproduction test has no home outside our S3 bucket, or DATA_HOMES.md is stale"; exit 1; }
python3 CANON/doc_audit.py --check || { echo "REFUSED: the documentation audit has findings (CANON/doc_audit.py)"; exit 1; }
python3 docs/book/figscripts/make_app_E.py --check || { echo "REFUSED: Appendix E differs from what make_app_E.py writes; edit docs/book/figscripts/app_E_overrides.json, not the appendix"; exit 1; }
python3 docs/book/stale_checks.py --check || { echo "REFUSED: a book check holds a number its chapter no longer prints (docs/book/stale_checks.py)"; exit 1; }
python3 docs/book/build_repro_chapter.py --check || { echo "checked_push.sh: THE REPRODUCED CHAPTER differs from Biological_Physics/MethylPhys/reproduce/tests.json or verified.json - run python3 docs/book/build_repro_chapter.py - push refused"; exit 1; }
python3 CANON/catalogue.py --check || { echo "checked_push.sh: CATALOGUE CHECK FAILED - Biological_Physics/MethylPhys/INSTRUMENTS.md or the README catalogue differs from CANON/catalogue.json - push refused"; exit 1; }
python3 CANON/bib_check.py > /dev/null || { python3 CANON/bib_check.py | tail -20; echo "checked_push.sh: BIBLIOGRAPHY CHECK FAILED - a citation does not match its CrossRef record - push refused"; exit 1; }
n=$(grep -rl "COMMISSIONING-RETURN" docs/book --include=*.tex 2>/dev/null | wc -l | tr -d " ")
[ "$n" -gt 0 ] && echo "REMINDER: $(grep -rh "COMMISSIONING-RETURN" docs/book --include=*.tex | wc -l | tr -d " ") figure(s) wait to go back into the book at commissioning (development/COMMISSIONING_RETURNS.md)."
git push "$@"
