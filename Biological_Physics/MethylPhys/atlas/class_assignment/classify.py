#!/usr/bin/env python3
"""Architecture-class assignment by rule (DRAFT 2026-09-28, not in the chain). The existing classes were assigned by lists of
examples (G-002's 37 reference cells, Issue 002's 'what_includes', the v1 115-cell map) with no written criterion. This file
states the criterion those lists imply and TESTS it against all three records before it may be applied to any new cell.

Facts per cell concept (biology only - no methylation data enters):
  lineage    pluripotent | haematopoietic | mesenchymal (connective, vascular, smooth muscle, adipose, pericyte/VLMC) |
             neural | epithelial | striated_muscle | neural_crest
  potency    pluripotent | tissue_stem (self-renewing, multipotent) | progenitor (committed, dividing precursor) | differentiated
  post_mitotic   does not divide under normal adult conditions (True/False)
  secretory      defining function is producing a secreted product (enzyme, hormone, bile, milk, pigment) (True/False)
Rule, first match wins:
  R1 potency pluripotent            -> stem_pluri
  R2 potency tissue_stem            -> stem_adult
  R3 potency progenitor             -> progenitor
  R4 lineage haematopoietic         -> immune
  R5 lineage mesenchymal            -> stromal
  R6 post_mitotic                   -> terminal
  R7 secretory                      -> secretory
  R8 otherwise (dividing epithelium) -> cycling
Composite labels (mixtures, tissues) are not cells and are reported, never classified."""
import csv, json, os, sys, collections
H=os.path.dirname(os.path.abspath(__file__))
F={r["concept"]:r for r in csv.DictReader(open(f"{H}/cell_facts_v0.csv"))}
def rule(c):
    f=F[c]
    if f["lineage"]=="composite": return "COMPOSITE"
    if f["potency"]=="pluripotent": return "stem_pluri"
    if f["potency"]=="tissue_stem": return "stem_adult"
    if f["potency"]=="progenitor": return "progenitor"
    if f["lineage"]=="haematopoietic": return "immune"
    if f["lineage"]=="mesenchymal": return "stromal"
    if f["post_mitotic"]=="True": return "terminal"
    if f["secretory"]=="True": return "secretory"
    return "cycling"
MAP=json.load(open(f"{H}/label_to_concept_v0.json"))
out=[]; 
for rec in ("G002_reference_37","issue002_what_includes","v1_map_115","v2_candidates"):
    for lab,(concept,recorded) in MAP[rec].items():
        got=rule(concept)
        status="new" if recorded is None else ("composite" if got=="COMPOSITE" else ("agree" if got==recorded else "CONFLICT"))
        out.append(dict(record=rec,label=lab,concept=concept,recorded=recorded or "",rule=got,status=status,basis=F[concept]["basis"]))
with open(f"{H}/class_test_report.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
c=collections.Counter((o["record"],o["status"]) for o in out)
for rec in ("G002_reference_37","issue002_what_includes","v1_map_115","v2_candidates"):
    print(rec, {k[1]:v for k,v in c.items() if k[0]==rec})
print("\nCONFLICTS:"); [print(f"  {o['record']:<24} {o['label']:<28} recorded {o['recorded']:<11} rule {o['rule']:<11} | {o['basis']}") for o in out if o["status"]=="CONFLICT"]
print("\nCOMPOSITE (not cells):", sorted({o["label"] for o in out if o["status"]=="composite"}))
print("\nNEW CELLS:"); [print(f"  {o['label']:<34} -> {o['rule']:<11} | {o['basis']}") for o in out if o["status"]=="new"]
