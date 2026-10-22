import numpy as np
"""The Globe Explorer's globes beyond the atlas (2026-10-21, to-do step 28; author: "the webpage to show ALL the versions").
Writes website/art/out/explorer/<id>.json (a globe spec, chain/MethylPhys_Interface/globe_viewer.py) and its pictures beside it:
  purified   Met-A's own healthy reference: the purified-array group profiles at its 6,000 neutrophil identity sites
             (Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json), one layer per group, drawn as dots on the atlas's layout
  samples    the report globe of each example sample, from the one command's own globe module (report_globe_v4.build), unchanged
Usage: python3 website/art/make_explorer.py purified
       python3 website/art/make_explorer.py noise NOISE_DIR   (output of atlas/tools/noise_maps_v21.py combine)
       python3 website/art/make_explorer.py sample ID "TITLE" BETAS.csv BUNDLE.json [planted]"""
import os, sys, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "..", ".."))
CH = os.path.join(R, "Biological_Physics", "MethylPhys", "chain"); sys.path[:0] = [CH, os.path.join(CH, "MethylPhys_Interface")]
import globe_viewer as V   # presentation only: writes the spec the one command made
OUT = os.path.join(H, "out", "explorer"); URL = "../art/explorer"
GROUP = {"NEU": "neutrophils", "MONO": "monocytes", "B": "B cells", "CD4T": "CD4 T cells", "CD8T": "CD8 T cells", "NK": "NK cells", "BASO": "basophils",
         "EOS": "eosinophils", "BAS": "basophils", "TREG": "regulatory T cells", "GRAN": "granulocytes"}

def _build(kind, inp=None):
    """A reference globe exactly as the one command draws it (2026-10-22, rule 23b-f): run_sample.py --build KIND writes the spec; nothing is drawn here."""
    import subprocess, tempfile
    sp = os.path.join(tempfile.mkdtemp(), "spec.json"); rs = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
    r = subprocess.run([sys.executable, rs, "--build", kind, "--build-out", sp] + (["--build-input", inp] if inp else []), capture_output=True, text=True)
    if r.returncode or not os.path.exists(sp): sys.exit(f"run_sample.py --build {kind} failed: {r.stdout[-400:]} {r.stderr[-600:]}")
    return json.load(open(sp))

def purified(): return _build("explorer-purified")

def noise(d): return _build("explorer-noise", d)

def sample(sid, title, betas, bundle=None, planted=False):
    """A sample globe exactly as the one command draws it (2026-10-22, rule 23b-f): runs chain/MethylPhys_Interface/run_sample.py on the
    sample's values with --explorer-spec and only renames the spec; nothing is read or drawn here. bundle is kept so recorded commands run."""
    import subprocess, tempfile
    d = tempfile.mkdtemp(); sp = os.path.join(d, "spec.json")
    rs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Biological_Physics", "MethylPhys", "chain", "MethylPhys_Interface", "run_sample.py")
    r = subprocess.run([sys.executable, rs, "--betas", betas, "--specimen", "whole blood", "--id", sid, "--out", os.path.join(d, sid + ".html"),
                        "--explorer-spec", sp] + (["--banner", "PLANTED: a constructed example, not a person."] if planted else []), capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(sp): sys.exit(f"run_sample.py did not write the globe ({r.returncode}): {r.stdout[-400:]} {r.stderr[-400:]}")
    s = json.load(open(sp)); s["id"] = sid; s["name"] = title
    if planted: s["about"] = "<b>PLANTED: a constructed example, not a person.</b> A clustered loss was planted in a healthy array. " + s["about"]
    return s

if __name__ == "__main__":
    spec = purified() if sys.argv[1] == "purified" else noise(sys.argv[2]) if sys.argv[1] == "noise" else sample(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], len(sys.argv) > 6)
    os.makedirs(OUT, exist_ok=True); s = V.write_files(spec, OUT, URL)
    json.dump(s, open(os.path.join(OUT, spec["id"] + ".json"), "w"), indent=1); print("wrote", spec["id"], len(s["assets"]), "pictures")
