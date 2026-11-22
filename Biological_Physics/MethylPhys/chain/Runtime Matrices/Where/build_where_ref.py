#!/usr/bin/env python3
"""Plan step 12, the positions reference for the 'where' stage (design PRIVATE-WHERE). For every Law Sphere unit (27,718 islands, then 50-CpG
blocks; stage_law_sphere.unit_map) and every Sequencing Globe region (1,000 consecutive CpG indexes; stage_q2_globe.REGION) it writes: chromosome,
first and last CpG position (hg19, 1-based), the UCSC CpG island it lies in or overlaps (cpgIslandExt, named chr:start-end), the nearest gene by
transcription start (refGene; name, strand, signed distance), and how many array probes it holds (the chain's sequencing key: 0 = arrays blind).
Check: every array probe's recorded loyfer_cpg_index must land on the same chromosome and position as its key row (all 813,951 or the job stops).
Inputs: the UCSC hg19 sequence (hg19.fa.gz, public), cpgIslandExt and refGene (public), the chain's key and units.npz. The CpG index is built by us,
the way wgbstools numbers it: every CG in chromosome order chr1..chr22, chrX, chrY, chrM, 1-based position of the C; the key check proves it.
Usage: where_ref.py hg19.fa.gz cpgIslandExt.txt.gz refGene.txt.gz key.parquet units.npz out_dir"""
import sys, os, json, numpy as np, pandas as pd
bed, isl, ref, key, units, out = sys.argv[1:7]; os.makedirs(out, exist_ok=True)
import gzip, re
ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]; GOT = {}; cur = None; buf = []
def _done(c, b):
    if c in ORDER: q = "".join(b).upper(); GOT[c] = np.array([m.start() + 1 for m in re.finditer("(?=CG)", q)], np.int64)
with gzip.open(bed, "rt") as f:   # 'bed' is the hg19 sequence here; each chromosome is counted as soon as it is read
    for line in f:
        if line.startswith(">"): _done(cur, buf); cur = line[1:].split()[0]; buf = []
        elif cur in ORDER: buf.append(line.rstrip())
    _done(cur, buf)
CH = [np.full(len(GOT[c]), ORDER.index(c), np.int8) for c in ORDER if c in GOT]; PO = [GOT[c] for c in ORDER if c in GOT]
chr_code = np.concatenate(CH); pos_of = np.concatenate(PO); n = len(pos_of); chr_of = np.array(ORDER, dtype=object)[chr_code]
print(json.dumps(dict(cpgs=int(n))))
K = pd.read_parquet(key); K = K.dropna(subset=["loyfer_cpg_index"]); ki = K.loyfer_cpg_index.astype(int).values
cc = [c for c in ("chrom_hg19", "chr_hg19", "chrom") if c in K][0]; pc = [c for c in ("pos_hg19", "start_hg19", "pos", "position") if c in K]
ok_chr = (np.asarray(chr_of[ki - 1]).astype(str) == K[cc].astype(str).str.replace("^(?!chr)", "chr", regex=True).values)
chk = dict(key_rows=int(len(K)), chromosome_agrees=int(ok_chr.sum()))
if pc: d = np.abs(pos_of[ki - 1] - K[pc[0]].astype(np.int64).values); chk.update(position_within_1bp=int((d <= 1).sum()), position_column=pc[0])
print(json.dumps(chk)); assert chk["chromosome_agrees"] == chk["key_rows"] and (not pc or chk["position_within_1bp"] == chk["key_rows"]), "key check failed"
probes = np.bincount(ki, minlength=n + 2)
I = pd.read_csv(isl, sep="\t", header=None, usecols=[1, 2, 3], names=["chr", "s0", "e"]); I["name"] = I.chr + ":" + (I.s0 + 1).astype(str) + "-" + I.e.astype(str)
G = pd.read_csv(ref, sep="\t", header=None, usecols=[2, 3, 4, 5, 12], names=["chr", "strand", "txs", "txe", "gene"]); G["tss"] = np.where(G.strand == "+", G.txs + 1, G.txe)
G = G.drop_duplicates(["chr", "tss", "gene"]); TSS = {c: (g.tss.values[np.argsort(g.tss.values)], g.gene.values[np.argsort(g.tss.values)], g.strand.values[np.argsort(g.tss.values)]) for c, g in G.groupby("chr")}
ISL = {c: (g.s0.values + 1, g.e.values, g.name.values) for c, g in I.sort_values("s0").groupby("chr")}
def annotate(first, last):
    rows = []
    for a, b in zip(first, last):
        a, b = int(max(1, a)), int(min(n, b)); ca, cb = str(chr_of[a - 1]), str(chr_of[b - 1])
        if ca != cb:   # a span crossing a chromosome end: keep the first chromosome's part
            j = a
            while j < b and str(chr_of[j]) == ca: j += 1
            b = j
        s, e = int(pos_of[a - 1]), int(pos_of[b - 1]); r = dict(chr=ca, start=s, end=e + 1, cpgs=b - a + 1, array_probes=int(probes[a:b + 1].sum()))
        il = ISL.get(ca)
        if il is not None:
            k = np.searchsorted(il[0], e + 1, "right"); hit = [il[2][q] for q in range(max(0, k - 3), k) if il[1][q] >= s]
            r["island"] = ";".join(hit)
        t = TSS.get(ca)
        if t is not None and len(t[0]):
            m = (s + e) // 2; k = np.searchsorted(t[0], m); c = [q for q in (k - 1, k) if 0 <= q < len(t[0])]; q = min(c, key=lambda q: abs(t[0][q] - m))
            dd = int(m - t[0][q]); r.update(gene=t[1][q], gene_distance=dd if t[2][q] == "+" else -dd)
        rows.append(r)
    return pd.DataFrame(rows)
Z = np.load(units); nI = len(Z["isl_first"]); blk = int(Z["block"][0])
U = annotate(Z["isl_first"], Z["isl_last"]); U.insert(0, "unit", np.arange(nI)); U["kind"] = "island"
nb = (n - 1) // blk + 1; bf = np.arange(nb) * blk + 1; Bk = annotate(bf, bf + blk - 1); Bk.insert(0, "unit", nI + np.arange(nb)); Bk["kind"] = "block"
UU = pd.concat([U, Bk], ignore_index=True); UU.to_parquet(os.path.join(out, "where_units_v1.parquet"), index=False)
R = 1000; nr = n // R + 1; rf = np.arange(nr) * R + 1; RG = annotate(rf, np.minimum(rf + R - 1, n)); RG.insert(0, "region", np.arange(nr)); RG.to_parquet(os.path.join(out, "where_regions_v1.parquet"), index=False)
meta = dict(kind="IAM 'where' positions reference v1", build="hg19", cpgs=n, units=int(len(UU)), regions=int(len(RG)), key_check=chk,
            sources=dict(cpg_index="built from the UCSC hg19 sequence in wgbstools numbering (Loyfer et al. 2023), checked against the chain key", islands="UCSC hg19 cpgIslandExt", genes="UCSC hg19 refGene (nearest transcription start)"),
            array_blind_islands=int(((U.array_probes == 0)).sum()))
json.dump(meta, open(os.path.join(out, "where_ref_v1.json"), "w"), indent=1); print(json.dumps(meta))
