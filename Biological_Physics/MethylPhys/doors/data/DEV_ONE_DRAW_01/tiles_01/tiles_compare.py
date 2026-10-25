"""Plan 14.9 (development): Stage Q2's count on one spent healthy plasma (SRR23279758) with the commissioned tiles and with tiles built from the
genome dictionary's chromosome ranges, both through run_sample.py --dev-count-only (--dev-tiles table | dictionary); this script runs both and
copies the count each bundle holds. Usage: python3 tiles_compare.py PAT.gz WORKDIR OUT.txt"""
import sys, os, json, subprocess
P, W, OUT = sys.argv[1:4]; os.makedirs(W, exist_ok=True); RS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain", "MethylPhys_Interface")
L = ["tiles from | tiles counted | loose share | held error | held opportunities"]
for t in ("table", "dictionary"):
    b = os.path.join(W, t + ".json")
    subprocess.run([sys.executable, "run_sample.py", "--pat", P, "--specimen", "plasma", "--id", "tl_" + t, "--dev-count-only", "--dev-tiles", t,
                    "--out", os.path.join(W, t + ".html"), "--bundle", b], cwd=RS, capture_output=True, text=True, check=True)
    c = json.load(open(b))["iam_a_count_only"]["count"]; L.append(f"{t} | {c['tiles']} | {c['loose_share']:.6f} | {c['held_err']:.6f} | {c['held_opp']}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
