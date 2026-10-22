"""sex_chromosomes.py - how every IAM instrument handles the X and Y chromosomes, from the chain's own files (2026-10-21; author: "even though
the researcher knows this, they may assume WE didnt account for it properly and we MUST show we did").

Every number here is computed from committed files at run time, never typed: each instrument's site list (its runtime matrix), the frozen
list of atlas sites on the X and Y (chain/Runtime Matrices/Atlas_V21/sex_chromosome_sites_v21.json, from the two Illumina manifests Stage 1
is pinned to), IAM-A's chromosome lists in its counter, Stage 0's sex rule, Group-A's sealed bar 2, and the atlas's uncertainty summary.
It writes one section, in Markdown and in LaTeX, between markers in each document that carries it:
  <!-- SEX-SECTION:BEGIN --> ... <!-- SEX-SECTION:END -->      (Markdown documents)
  % SEX-SECTION:BEGIN ... % SEX-SECTION:END                    (the book)
Usage: python3 CANON/sex_chromosomes.py [--check]   (--check: refuse when any document's section differs from what the files give)"""
import os, sys, json, re, importlib
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MP = os.path.join(R, "Biological_Physics", "MethylPhys"); CH = os.path.join(MP, "chain")
RM = os.path.join(CH, "Runtime Matrices")
DOCS_MD = [os.path.join(MP, "manual", "decoder_text.md"), os.path.join(MP, "QUESTIONS_AND_ANSWERS.md"), os.path.join(MP, "INSTRUMENTS.md"), os.path.join(MP, "sop", "MethylPhys_CPG_SOP_v4.md")]
DOC_TEX = os.path.join(R, "docs", "book", "part6", "p6_19_chain.tex")

def facts():
    X = json.load(open(os.path.join(RM, "Atlas_V21", "sex_chromosome_sites_v21.json"))); XS, YS = set(X["X"]) | set(X["X_array"]), set(X["Y"]) | set(X["Y_array"])
    def count(ids):
        ids = list(ids)   # every id is an atlas site or a probe of the pinned EPIC v1 and 450K manifests, whose X and Y probes are all in the frozen list
        return dict(n=len(ids), x=sum(1 for i in ids if i in XS), y=sum(1 for i in ids if i in YS))
    F = json.load(open(os.path.join(RM, "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]["sites"]
    C = json.load(open(os.path.join(RM, "Met_A_Floors", "neutrophil_reference_v1_2.json")))["sites_ordered"]
    import glob
    N = json.load(open(glob.glob(os.path.join(RM, "**", "noise_sites_EPIC_v1.json"), recursive=True)[0]))["sites"]
    B = json.load(open(glob.glob(os.path.join(RM, "**", "blood_composition_EPIC_v1.json"), recursive=True)[0]))["markers"]
    sys.path.insert(0, CH); D = importlib.import_module("stage_deconv_v21"); Q = importlib.import_module("stage_q2_selfref"); S0 = importlib.import_module("stage_0_intake")
    dm = list(D.markers(D.reference(["neutrophils", "monocytes", "b cells", "cd4 t cells", "cd8 t cells", "nk cells"])))
    G = json.load(open(os.path.join(MP, "doors", "data", "GROUPA_COMMISSIONING", "scored_groupa_01.json")))["B2"]
    U = json.load(open(os.path.join(MP, "atlas", "v2.1", "sky", "uncertainty_v21.json")))
    ratio = sorted(U["xy_median_spread"][c] / U["median_spread"][c] for c in U["cells"])
    bam = list(Q.AUTOSOMES); pat = [c for c, _ in Q.HG19_CPGS]
    src = open(os.path.join(CH, "stage_q2_selfref.py")).read(); i = src.index("def count_pat"); body = src[i:src.index("return _finish", i)]
    held_checks_chrom = "tiles.get(c)" in body.split("elif")[1] if "elif" in body else False
    return dict(atlas=dict(n=X["atlas_sites"], x=X["x_sites"], y=X["y_sites"]), meta=count(F), cscore=count(C), noise=count(N), mix=count(B), deconv=count(dm),
                cutoff=S0.SEX_GETSEX_CUTOFF, groupa=dict(diff=G["rel_difference"], t=G["t"], women=G["n1"], men=G["n2"]),
                unc=dict(lo=ratio[0], hi=ratio[-1]), iama=dict(bam_autosomes=bam == [f"chr{k}" for k in range(1, 23)], pat_tiles_autosomes=pat == [f"chr{k}" for k in range(1, 23)],
                pat_held_all_chromosomes=not held_checks_chrom))

def section(f):
    m, c, nz = f["meta"], f["cscore"], f["noise"]
    bound = (nz["x"] + nz["y"]) / nz["n"]
    rows = [
        ("Stage 0, intake", f"Predicts the array's sex from its X and Y probe brightness (step 0.8: female when log2 Y minus log2 X is below {f['cutoff']}). "
                            "When a sex is declared and the array disagrees, the array is stopped and the reason printed: a swapped sample is caught before any reading."),
        ("Met-A", f"{m['n']:,} neutrophil identity sites: {m['x']} on the X, {m['y']} on the Y. They were chosen where the six purified reference arrays, one of them a woman's, agree. "
                  f"Group-A's sealed test read {f['groupa']['women']} women and {f['groupa']['men']} men the same after the tare "
                  f"({100 * f['groupa']['diff']:+.2f} %, {abs(f['groupa']['t']):.1f} standard errors)."),
        ("C-Score", f"The same {c['n']:,} sites ({c['x']} on the X, {c['y']} on the Y), in genome order; it reads how departures cluster, read against the same healthy reference."),
        ("Noise index", f"{nz['n']:,} sites, {nz['x']} on the X and {nz['y']} on the Y. Even if every one of them read at maximum noise, the index could move by at most "
                        f"{bound:.4f} bits ({100 * bound:.2f} % of its 1-bit scale)."),
        ("Stage A, the blood mix", f"{f['mix']['n']:,} markers, {f['mix']['x']} on the X or Y."),
        ("IAM-Deconvolver", f"{f['deconv']['n']:,} markers, {f['deconv']['x']} on the X, {f['deconv']['y']} on the Y."),
        ("IAM-A", "Each sample is read against its own predicted error, so no other person, of either sex, enters its reading. "
                  + ("Aligned files (BAM) are counted on chromosomes 1 to 22 only. " if f["iama"]["bam_autosomes"] else "")
                  + ("From a .pat file the loose share is counted on chromosomes 1 to 22, and the held error on every chromosome the file holds (the route IAM-A was "
                     "commissioned on; a recount with the X and Y left out is on the to-do list, Decoder finding F37)." if f["iama"]["pat_held_all_chromosomes"] else
                     "From a .pat file both counts use chromosomes 1 to 22.")),
        ("Group-A", "Compares groups run in the same series; its sealed bar 2 is a women-against-men null (above)."),
        ("IAM-Atlas v2.1 and the globes", f"{f['atlas']['n']:,} atlas sites, {f['atlas']['x']:,} on the X and {f['atlas']['y']} on the Y. The atlas's donors are women and men, "
                  f"so it is {f['unc']['lo']:.1f} to {f['unc']['hi']:.1f} times less sure on the X and Y than elsewhere (the noise globe shows it). Every change count on a globe "
                  "is also given on the X and Y, the page prints the sex Stage 0 read, and a tumor against its own normal tissue (same person, same sex) is counted with and without "
                  "the X and Y: in a tumor an X change can be real biology."),
    ]
    why = ("**Why sex matters to a methylation reading.** A woman has two X chromosomes and silences one in every cell by methylating it (X inactivation). "
           "At the switch sites of hundreds of X genes a man's single X reads near 0 and a woman's two copies average about 0.5. The Y exists only in men. "
           "A comparison between people of different sex can therefore light up the X without any disease. Each instrument handles this as follows "
           "(numbers computed from the chain's own files by `CANON/sex_chromosomes.py`):")
    md = [why, "", "| Instrument | X and Y |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    esc = lambda s: s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("**", "")
    tx = [r"\paragraph{Sex and the X chromosome.} " + esc(why.split("**", 2)[2].strip()).replace("`CANON/sex\\_chromosomes.py`", r"\texttt{CANON/sex\_chromosomes.py}"),
          r"\begin{description}"] + [rf"\item[{esc(a)}] {esc(b)}" for a, b in rows] + [r"\end{description}"]
    return "\n".join(md), "\n".join(tx)

def put(text, block, md=True):
    b, e = ("<!-- SEX-SECTION:BEGIN -->", "<!-- SEX-SECTION:END -->") if md else ("% SEX-SECTION:BEGIN", "% SEX-SECTION:END")
    new = f"{b}\n{block}\n{e}"
    if b in text: return re.sub(re.escape(b) + r".*?" + re.escape(e), lambda _: new, text, flags=re.S)
    return None

if __name__ == "__main__":
    f = facts(); md, tx = section(f); bad = []
    for p, blk, ismd in [(d, md, True) for d in DOCS_MD] + [(DOC_TEX, tx, False)]:
        t = open(p, encoding="utf-8").read(); n = put(t, blk, ismd)
        if n is None: bad.append(f"{os.path.relpath(p, R)}: no SEX-SECTION markers"); continue
        if n != t:
            if "--check" in sys.argv: bad.append(f"{os.path.relpath(p, R)}: section differs from the chain's files")
            else: open(p, "w", encoding="utf-8").write(n)
    if bad: print("sex_chromosomes: REFUSED: " + "; ".join(bad) + " (run python3 CANON/sex_chromosomes.py)"); sys.exit(1)
    print(f"sex_chromosomes: ok ({len(DOCS_MD) + 1} documents)" if "--check" in sys.argv else "sex_chromosomes: written"); print(json.dumps(f) if "--facts" in sys.argv else "")
