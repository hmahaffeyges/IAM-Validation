#!/usr/bin/env python3
"""Batch runner — process every patient visit that does not yet have a report.

Folder layout it expects (you create the patient + dated visit folders; the
baselines/ folder is created for you):

  patients/
    ANON_0001/
      baselines/                       (created automatically; holds the vectors)
      2026-06-22/                      one folder per blood draw, named by date
        <sample>_Grn.idat[.gz]
        <sample>_Red.idat[.gz]
        questionnaire.json
        CPG_report_ANON_0001_...html   (written here after the run)
      2027-01-10/
        ...second draw...
    ANON_0002/
      2026-06-22/ ...

Run:
    python run_batch.py --patients /path/to/patients

For each patient, every dated visit folder WITHOUT a report is run, in date
order, so a later draw always compares against the earlier one (trajectory).
Already-reported visits are skipped unless you pass --force. A summary table
prints at the end.
"""
import sys, argparse, glob, json, os
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import walther_clinical as wc


def _has_report(visit):
    return bool(glob.glob(str(visit / "CPG_report_*.html")))


def _is_visit_dir(p):
    return p.is_dir() and p.name != "baselines"


def main():
    ap = argparse.ArgumentParser(description="CPG batch runner over a patients/ tree.")
    ap.add_argument("--patients", required=True, help="root folder holding one subfolder per patient")
    ap.add_argument("--force", action="store_true", help="re-run visits that already have a report")
    args = ap.parse_args()

    root = Path(args.patients)
    if not root.exists():
        sys.exit(f"patients root not found: {root}")

    summary = []
    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir():
            continue
        visits = sorted([v for v in pdir.iterdir() if _is_visit_dir(v)], key=lambda v: v.name)
        for v in visits:
            if _has_report(v) and not args.force:
                print(f"-- skip {pdir.name}/{v.name} (already has a report)")
                continue
            print(f"\n=== {pdir.name} / {v.name} ===")
            try:
                wc.run_from_folder(str(v), str(v))
                row = {"patient": pdir.name, "visit": v.name}
                bls = glob.glob(str(pdir / "baselines" / "baseline_*.json"))
                if bls:
                    b = json.load(open(max(bls, key=os.path.getmtime)))  # the one just written
                    row["flagged"] = b.get("flagged_disease") or "-"
                    row["maha_d"] = b.get("mahalanobis_distance")
                    row["beyond"] = b.get("mahalanobis_beyond_band")
                summary.append(row)
            except Exception as e:
                summary.append({"patient": pdir.name, "visit": v.name, "error": str(e)})
                print(f"   ERROR: {e}")

    print("\n" + "=" * 74)
    print("BATCH SUMMARY")
    print("=" * 74)
    for r in summary:
        if "error" in r:
            print(f"  {r['patient']:16} {r['visit']:12} ERROR: {r['error'][:40]}")
        else:
            md = f"Mahalanobis d={r['maha_d']}" if r.get("maha_d") is not None else "no flag raised"
            band = ""
            if r.get("beyond") is True:
                band = " (BEYOND healthy band)"
            elif r.get("beyond") is False:
                band = " (within healthy band)"
            print(f"  {r['patient']:16} {r['visit']:12} flag={str(r.get('flagged','-')):16} {md}{band}")
    print(f"\n{len(summary)} visit(s) processed.")


if __name__ == "__main__":
    main()
