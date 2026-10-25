"""DEV-ONE-DRAW-01 provisional reference, synthetic check 1 (2026-10-24, before provisional reference test 1): the machinery on the donor the
column was built from (GSE87197 donor 1, spent), so it should pass easily; a failure means the wiring is wrong, not the idea. Absorption
test 1's own files are read again through the one command with the column added: the pure donor 1 pool, and the plasma with 0 % and 40 %
planted. Expected: the pool reads megakaryocytes (provisional) as its largest share; with 40 % planted the megakaryocyte share rises and
the four stand-ins do not.  Usage: python3 provisional_synthetic_01.py OUT.txt POOL.lbeta PLASMA00.lbeta PLASMA40.lbeta [COLUMN, default megakaryocytes; megakaryocytes@2
for version 2]"""
import sys, os, json, subprocess
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(H, "..", "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py")
SI = ("lung interstitial macrophages", "erythrocyte progenitors", "MEP (bone marrow)", "eosinophils"); L = ["provisional reference, synthetic check 1 (donor 1, in-sample)"]
for tag, p in zip(("donor1_pool", "plasma_00pct", "plasma_40pct"), sys.argv[2:5]):
    b = f"/tmp/ps_{tag}.json"
    subprocess.run([sys.executable, RS, "--dev-seq-beta", p, "--dev-seq-values", "1", "--specimen", "plasma", "--id", tag, "--dev-provisional", (sys.argv[5] if len(sys.argv) > 5 else "megakaryocytes"),
                    "--out", f"/tmp/ps_{tag}.html", "--bundle", b], capture_output=True, text=True)
    B = json.load(open(b)); A = {x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells") or []}; P = {x["cell"]: x["share"] for x in (B.get("cell_shares_provisional") or {}).get("cells") or []}
    L.append(f"{tag}: atlas only, stand-ins summed {sum(A.get(x, 0) for x in SI):.4f} | with the column: megakaryocytes {P.get('megakaryocytes (provisional)')}, "
             f"stand-ins summed {sum(P.get(x, 0) for x in SI):.4f}, largest {max(P, key=P.get) if P else None} | markers {(B.get('cell_shares_provisional') or {}).get('markers')}")
    print(L[-1], flush=True)
open(sys.argv[1], "w").write("\n".join(L) + "\n")
