#!/usr/bin/env python3
"""A8S step 2 (register A8, candidate iv: traffic; exploration, 2026-11-16). Runs ON A GITHUB MACHINE. For each cell group of the names table
(written before any number was read), reads the open per-type cCRE calls of Zhang et al. 2021 (Cell 184:5985; open server
decoder-genetics.wustl.edu/catlasv1/humanenhancer/data/cCREs/<type>.bed, hg38) and reports T2: the number of open cCREs and the open base pairs
(union of the group's types, merged). Only these few numbers come back as notice lines; nothing is kept but a small JSON."""
import os, json, urllib.request
N = {"adipocytes": ["Adipocyte"], "cardiomyocyte": ["Ventricular_Cardiomyocyte", "Atrial_Cardiomyocyte"], "hepatocyte": ["Hepatocyte"], "cortical neurons": ["Glutaminergic_Neuron_1", "Glutaminergic_Neuron_2", "GABAergic_Neuron_1", "GABAergic_Neuron_2"], "oligodendrocytes": ["Oligodendrocyte"], "smooth muscle": ["Smooth_Muscle_General"], "striated muscle": ["Type_I_Skeletal_Myocyte", "Type_II_Skeletal_Myocyte"], "naive cd4 t cells": ["Naive_T_cell"], "naive cd8 t cells": ["Naive_T_cell"], "t central memory cd4": ["T_lymphocyte_2_CD4+"], "t effector memory cd4": ["T_lymphocyte_2_CD4+"], "t effector cell cd8": ["T_Lymphocyte_1_CD8+"], "effector memory cd8 t cells": ["T_Lymphocyte_1_CD8+"]}
B = "https://decoder-genetics.wustl.edu/catlasv1/humanenhancer/data/cCREs/"
cache, out = {}, {}
def bed(t):
    if t not in cache:
        txt = urllib.request.urlopen(B + urllib.request.quote(t + ".bed"), timeout=300).read().decode()
        cache[t] = [(l.split("\t")[0], int(l.split("\t")[1]), int(l.split("\t")[2])) for l in txt.splitlines() if l and not l.startswith(("#", "track"))]
    return cache[t]
for group, types in N.items():
    iv = sorted(x for t in types for x in bed(t)); bp = n = 0; cur = None
    for c, s, e in iv:
        if cur and c == cur[0] and s <= cur[2]: cur[2] = max(cur[2], e)
        else:
            if cur: bp += cur[2] - cur[1]; n += 1
            cur = [c, s, e]
    if cur: bp += cur[2] - cur[1]; n += 1
    out[group] = dict(types=types, n_cCRE_merged=n, open_bp=bp, n_calls_by_type={t: len(bed(t)) for t in types})
    print("::notice::A8S T2 " + json.dumps(dict(group=group, **out[group])))
os.makedirs("xout", exist_ok=True); json.dump(out, open("xout/a8s_traffic_t2.json", "w"), indent=1)
