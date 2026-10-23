#!/usr/bin/env python3
"""CANON/build_all.py - one command that rebuilds every generated document from its one source, in order (2026-10-23, plan 39.16; author: "All
of this could be avoided by using code and keeping the master data source in ONE place"). The sources are the canon's files (CANON/*.json, the
results register, the manifest, the commissioning registry) and the chain's own files; every document below is written from them and never
edited by hand. Run it before every push:  python3 CANON/build_all.py        (writes, then runs the cross-document checks)
                                            python3 CANON/build_all.py --check (writes nothing; exit 1 naming each document that differs)
The push gate runs the --check form first, so a forgotten generator is one message, not a chain of refusals."""
import os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MP = "Biological_Physics/MethylPhys/"
# (script, takes --check): the order is the order the data flows - the manifest first, the pages that read everything last
STEPS = [("CANON/chain_events.py", False), ("CANON/manifest.py", False), ("CANON/ascore_db.py", True), ("CANON/status_lines.py", True),
         ("CANON/four_domains.py", True), ("CANON/bit_tables.py", True), ("CANON/book_bits.py", True), ("CANON/atlas_story.py", True),
         ("CANON/results_to_tex.py", False), ("CANON/canon_to_tex.py", False), ("CANON/commissioning_report.py", True),
         ("CANON/earlier_results.py", True), ("CANON/data_homes.py", True), ("CANON/now_running.py", True), ("CANON/catalogue.py", True),
         ("CANON/sex_chromosomes.py", True), ("CANON/book_toolbox.py", True), ("CANON/replicate.py", True), (MP + "reproduce/readme.py", True),
         ("docs/book/build_repro_chapter.py", True), ("docs/book/figscripts/make_app_E.py", True), (MP + "manual/decoder_inventory.py", True),
         (MP + "manual/build_decoder.py", True), (MP + "sop/build_sop_v4.py", True), ("CANON/daylog.py", False), ("CANON/plan_md.py", True),
         ("CANON/status_md.py", True), ("CANON/master_inventory.py", False)]
CHECKS = ["CANON/no_dates.py --check", "CANON/status_claims.py --check", "CANON/doc_audit.py --check", "CANON/number_trace.py --check",
          "CANON/compile_check.py", "CANON/commissioning_check.py"]
def run(cmd):
    r = subprocess.run([sys.executable] + cmd.split(), cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip().split("\n")[-1][:220]
if __name__ == "__main__":
    check = "--check" in sys.argv; bad = []; t0 = time.time()
    for s, has in STEPS:
        if not os.path.exists(os.path.join(ROOT, s)): bad.append(f"{s}: missing"); continue
        if check and not has: continue
        rc, last = run(s + (" --check" if check else ""))
        if rc: bad.append(f"{s}: {last}")
    if not check:
        for c in CHECKS:
            rc, last = run(c)
            if rc: bad.append(f"{c}: {last}")
    for b in bad: print("  " + b)
    print(f"build_all: {'REFUSED (%d)' % len(bad) if bad else 'ok'} ({len(STEPS)} generators{'' if check else ', ' + str(len(CHECKS)) + ' checks'}, {time.time() - t0:.0f} s)"
          + (" - run python3 CANON/build_all.py" if bad and check else ""))
    sys.exit(1 if bad else 0)
