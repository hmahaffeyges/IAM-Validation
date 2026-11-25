#!/usr/bin/env python3
"""The Locator's gene relation and CpG-island context (positions reference v2), added to every place of v1.

For each place (every Law Sphere unit and every Globe region of where_units_v1 / where_regions_v1):
  - promoter genes: genes with a transcription start within 2 kb of the place (a 4-kb window centred on the start, the definition used by
    Koldobskiy et al. 2021, Nat Biomed Eng, Methods, "Genomic features and annotations"), so our gene lists compare with theirs directly;
  - gene-body genes: genes whose transcript (start to end) overlaps the place outside any of their promoter windows;
  - nearest gene by transcription start and its signed distance (upstream negative, on the gene's strand), kept from v1 for every place;
  - relation: promoter, gene body or intergenic (the closest of these, in that order);
  - CpG context: island, shore (within 2 kb of an island), shelf (2-4 kb) or open sea, by the same definitions;
  - each relation from two gene tables: NCBI RefSeq (UCSC refGene) and UCSC known genes (knownGene with kgXref symbols; the gene table the
    Hopkins study used through TxDb.Hsapiens.UCSC.hg19.knownGene).
Inputs are public UCSC hg19 files. Usage: build_where_genes_v2.py where_dir refGene.txt.gz knownGene.txt.gz kgXref.txt.gz cpgIslandExt.txt.gz out_dir"""
import sys, os, json, gzip, numpy as np, pandas as pd
W, REF, KG, KX, ISL, OUT = sys.argv[1:7]; os.makedirs(OUT, exist_ok=True)
PROM = 2000; SHORE = 2000; SHELF = 4000

def refgene(p):
    G = pd.read_csv(p, sep="\t", header=None, usecols=[2, 3, 4, 5, 12], names=["chr", "strand", "txs", "txe", "gene"])
    return G
def knowngene(p, x):
    G = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2, 3, 4], names=["kg", "chr", "strand", "txs", "txe"])
    X = pd.read_csv(x, sep="\t", header=None, usecols=[0, 4], names=["kg", "gene"]); G = G.merge(X, on="kg", how="left").dropna(subset=["gene"])
    return G[["chr", "strand", "txs", "txe", "gene"]]

def index(G):
    """per chromosome: transcription starts (1-based) and transcript spans (1-based, inclusive), each with its gene and strand."""
    G = G[G.chr.str.match(r"^chr([0-9]+|X|Y|M)$")].copy()
    G["tss"] = np.where(G.strand == "+", G.txs + 1, G.txe); G["s"] = G.txs + 1; G["e"] = G.txe
    out = {}
    for c, g in G.groupby("chr"):
        t = g.drop_duplicates(["tss", "gene"]).sort_values("tss"); b = g.drop_duplicates(["s", "e", "gene"]).sort_values("s")
        out[c] = dict(tss=t.tss.values, tg=t.gene.values, ts=t.strand.values, bs=b.s.values, be=b.e.values, bg=b.gene.values, bmaxe=np.maximum.accumulate(b.e.values))
    return out

def relate(P, IX):
    """P: places (chr, start, end; end exclusive as in v1). Returns promoter genes, body genes, nearest gene, distance, relation."""
    prom, body, near, dist, rel = [], [], [], [], []
    for c, s, e in zip(P.chr.values, P.start.values.astype(np.int64), P.end.values.astype(np.int64) - 1):
        x = IX.get(c)
        if x is None: prom.append(""); body.append(""); near.append(""); dist.append(np.nan); rel.append("intergenic"); continue
        i0, i1 = np.searchsorted(x["tss"], s - PROM, "left"), np.searchsorted(x["tss"], e + PROM, "right")
        pg = sorted(set(x["tg"][i0:i1]))
        j1 = np.searchsorted(x["bs"], e, "right"); j0 = np.searchsorted(x["bmaxe"], s, "left")   # transcripts starting by e and (by running max) possibly ending after s
        bg = sorted({x["bg"][j] for j in range(j0, j1) if x["be"][j] >= s} - set(pg))
        m = (s + e) // 2; k = np.searchsorted(x["tss"], m); cand = [q for q in (k - 1, k) if 0 <= q < len(x["tss"])]
        q = min(cand, key=lambda q: abs(x["tss"][q] - m)); d = int(m - x["tss"][q]); d = d if x["ts"][q] == "+" else -d
        prom.append(";".join(pg)); body.append(";".join(bg)); near.append(x["tg"][q]); dist.append(d)
        rel.append("promoter" if pg else "gene body" if bg else "intergenic")
    return prom, body, near, dist, rel

def context(P, isl):
    I = pd.read_csv(isl, sep="\t", header=None, usecols=[1, 2, 3], names=["chr", "s0", "e"]); I["s"] = I.s0 + 1
    D = {c: (g.sort_values("s").s.values, g.sort_values("s").e.values) for c, g in I.groupby("chr")}
    out = []
    for c, s, e in zip(P.chr.values, P.start.values.astype(np.int64), P.end.values.astype(np.int64) - 1):
        x = D.get(c)
        if x is None: out.append("open sea"); continue
        k = np.searchsorted(x[0], e + SHELF, "right"); best = None
        for q in range(max(0, k - 6), k):
            gap = 0 if (x[0][q] <= e and x[1][q] >= s) else min(abs(s - x[1][q]), abs(x[0][q] - e))
            best = gap if best is None else min(best, gap)
        out.append("open sea" if best is None or best > SHELF else "island" if best == 0 else "shore" if best <= SHORE else "shelf")
    return out

IXR = index(refgene(REF)); IXK = index(knowngene(KG, KX)); stats = {}
for nm in ("where_units_v1", "where_regions_v1"):
    P = pd.read_parquet(os.path.join(W, nm + ".parquet"))
    for tag, IX in (("refseq", IXR), ("known", IXK)):
        pg, bg, ng, d, r = relate(P, IX)
        P[f"promoter_genes_{tag}"] = pg; P[f"body_genes_{tag}"] = bg; P[f"nearest_gene_{tag}"] = ng; P[f"tss_distance_{tag}"] = d; P[f"relation_{tag}"] = r
    P["cpg_context"] = context(P, ISL)
    # consistency with v1: v1's nearest gene (refGene) must equal this build's refseq nearest gene at every place
    agree = float((P.gene.fillna("") == P.nearest_gene_refseq.fillna("")).mean()) if "gene" in P else None
    dagree = float((P.gene_distance.abs().fillna(-1) == pd.Series(P.tss_distance_refseq).abs().fillna(-1)).mean()) if "gene_distance" in P else None   # ties: two genes sharing one start
    v2 = nm.replace("_v1", "_v2"); P.to_parquet(os.path.join(OUT, v2 + ".parquet"), index=False)
    stats[v2] = dict(places=int(len(P)), nearest_gene_agrees_with_v1=agree, nearest_distance_agrees_with_v1=dagree,
                     relation_refseq=P.relation_refseq.value_counts().to_dict(), relation_known=P.relation_known.value_counts().to_dict(),
                     cpg_context=P.cpg_context.value_counts().to_dict(),
                     array_blind_promoter_places=int(((P.array_probes == 0) & (P.relation_refseq == "promoter")).sum()))
meta = dict(kind="IAM Locator positions reference v2 (gene relation and CpG-island context added to v1)", build="hg19",
            definitions=dict(promoter="transcription start +/- 2 kb (4-kb window centred on the start)", gene_body="transcript span outside the gene's promoter windows",
                             shore="within 2 kb of a CpG island", shelf="2 to 4 kb from a CpG island", open_sea="farther than 4 kb", source_of_definitions="Koldobskiy et al. 2021, Nature Biomedical Engineering, Methods"),
            sources=dict(refseq="UCSC hg19 refGene (NCBI RefSeq)", known="UCSC hg19 knownGene with kgXref gene symbols", islands="UCSC hg19 cpgIslandExt"), stats=stats)
json.dump(meta, open(os.path.join(OUT, "where_ref_v2.json"), "w"), indent=1); print(json.dumps(meta)[:3000])
