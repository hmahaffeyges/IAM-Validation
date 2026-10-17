"""PRED-TEMPERATURE-01 stage T3: one plant sample -> chain stage TL counts (chain-stamped out/<gsm>.json). Run on the box, never on the laptop.
usage: count_plant.py <bam> <ref.fa> <gsm> <accession> <temp> <runs> <workers>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "chain"))
import stage_temperature as TL
bam, fa, gsm, acc, temp, runs, w = sys.argv[1:8]
o = TL.count_plant(bam, fa, workers=int(w)); o.update(gsm=gsm, accession=acc, temp=temp, runs=runs.split(";"))
os.makedirs("out", exist_ok=True); TL.Q2.write_json(o, os.path.join("out", gsm + ".json"), manifest=["PHY-057"])
print(gsm, acc, temp, o["qualifying"], round(o["held_err"], 5), o["eps_corr"], o["conv_fail"], o["usable"], o["reason"], flush=True)
