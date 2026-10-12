"""Checks every number on the live fundraiser page (GOFUNDME_STORY_LIVE.txt, the text as posted at gofund.me/f83e34950) against the records,
computed by CANON/results_to_tex.compute from the committed scored files (2026-10-12; author: 'we must use code always to demand accuracy').
Each check pairs a phrase that must be on the page with the record value behind it. Run by push-gate rule 15. Usage: python3 fundraiser/check_story.py"""
import os, sys, json, importlib.util as u
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); D = "Biological_Physics/MethylPhys/doors/"
sp = u.spec_from_file_location("r2t", "CANON/results_to_tex.py"); m = u.module_from_spec(sp); sp.loader.exec_module(m)
V = {e["id"]: m.compute(e) for e in json.load(open("CANON/results_register.json", encoding="utf-8"))["entries"] if e.get("compute")}
S = open("fundraiser/GOFUNDME_STORY_LIVE.txt", encoding="utf-8").read()
b, c, s, ep, mo, fp, fb = (V[k] for k in ("DEV-TISSUE-BREAST-01", "DEV-TISSUE-COLON-01", "DEV-SAM-LEVER-01", "DEV-EPIQC-ARRAY-01", "DEV-CSCORE-MOSS-01",
                                          "DEV-FINGERPRINT-02-PROSTATE", "DEV-FINGERPRINT-02-BREAST"))
atlas = open("docs/book/part6/p6_14_atlas.tex", encoding="utf-8").read(); cn = open(D + "COMMISSIONING_NOTE_METAA_NEUTROPHILS.md", encoding="utf-8").read()
stages = {r.get("stage") for r in json.load(open(D + "data/DEV_TISSUE_COLON_01/scored_colon_tissue.json"))["rows"]}
C = [(f"Breast cancer, {b['n']} patients: {b['k']} tumors crossed the strict bar, where chance alone would give about {round(.05 * b['n'])}, and {b['up']} of {b['n']}", b["inside"]),
     (f"Colorectal cancer, {c['n']} patients (stage II): {c['k']} crossed, where chance would give about {round(.05 * c['n'])}, and {c['up']} of {c['n']}", c["inside"] and all(str(x).startswith("II") for x in stages)),
     (f"(p = {s['p']:.4f})", s["inside"]), (f"within {ep['raw'] * 100:.1f} %, before any correction", True),
     (f"{mo['ok']} of {mo['n']} readings stayed inside the healthy bar", mo["inside"]),
     (f"{fp['above']} of {fp['n']} cancer samples, the weakest still more than twice", fp["inside"] and fp["above"] == fp["n"] and fp["ratio"] > 2),
     ("We found it in every prostate cancer sample from one laboratory", "every LNCaP run above the largest PrEC run" in open(D + "DEV_FINGERPRINT_01_OUTCOME.md", encoding="utf-8").read()),
     (f"found in all {fb['n']} comparisons with normal breast cells", fb["gain"] == fb["n"]),
     ("74 purified human cell types", "74" in atlas), ("342,716 measurements", "342{,}716" in atlas or "342,716" in atlas), ("caught 92.7 %", "92.7" in atlas),
     ("(a 2 % loss of the pattern)", "a loss of 2 % of the neutrophil pattern (purified neutrophils)" in cn)]
bad = [p for p, ok in C if p not in S or not ok]
print("\n".join(f"{'OK  ' if p not in bad else 'FAIL'} {p}" for p, _ in C)); print("FUNDRAISER CHECK " + ("PASSED" if not bad else "FAILED"))
sys.exit(1 if bad else 0)
