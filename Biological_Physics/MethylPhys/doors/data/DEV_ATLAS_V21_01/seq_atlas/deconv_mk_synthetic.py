"""The IAM-Deconvolver on sequencing with the provisional megakaryocyte column (DEV-ONE-DRAW-01; development, diagnosis only): synthetic,
written and committed before it runs. The instrument's job: a mix of the reference's own cells reads its planted shares.
Known answer: 6 plasma-like mixes planted from the levels of REF (the reference built with run_sample.py --build seq-deconv-ref
--build-with-mk) at its marker sites, reads drawn k of n (20 reads per site), written as .lbeta files and read twice by the one command:
against REF (--dev-deconv-ref REF) and against the committed reference (no megakaryocytes). Mixes: granulocytes 0.5, monocytes 0.2 and,
in turn, megakaryocytes 0.2 with erythrocyte progenitors 0.1; megakaryocytes 0.3; erythrocyte progenitors 0.3; none of either (0.3 to
granulocytes); megakaryocytes 0.1 with erythrocyte progenitors 0.2; megakaryocytes 0.15 with erythrocyte progenitors 0.15.
Bars (against REF, operational, set before the run): the megakaryocyte and erythrocyte progenitor shares each within 0.05 of planted, in
every mix. Recorded, not scored: against the committed reference, where the planted megakaryocyte DNA lands.
Usage: python3 deconv_mk_synthetic.py REF WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd
REF, W, OUT = sys.argv[1:4]; os.makedirs(W, exist_ok=True); REF = os.path.abspath(REF)
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); sys.path.insert(0, C)
import stage_seq_atlas as Q, stage_seq_values as SV
R, M = Q.deconv_reference_of(REF); mu = R["mu"].loc[M]; assert "megakaryocytes" in R["cells"], "REF has no megakaryocyte column"
k = SV.key(); idx = pd.Series(k["idx"], index=k["probe"]); idx = idx[~idx.index.duplicated()]; rng = np.random.default_rng(20261104); N = 20
G, MO, MK, EP = "granulocytes", "monocytes", "megakaryocytes", "erythrocyte progenitors"
mixes = [{G: .5, MO: .2, MK: .2, EP: .1}, {G: .5, MO: .2, MK: .3}, {G: .5, MO: .2, EP: .3}, {G: .8, MO: .2}, {G: .5, MO: .2, MK: .1, EP: .2}, {G: .5, MO: .2, MK: .15, EP: .15}]
def read(f, j, tag, ref):
    b = os.path.join(W, f"mix{j}_{tag}.json"); cmd = [sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--specimen", "unknown",
                                                      "--id", f"mix{j}{tag}", "--out", os.path.join(W, f"mix{j}{tag}.html"), "--bundle", b] + (["--dev-deconv-ref", ref] if ref else [])
    subprocess.run(cmd, cwd=os.path.join(C, "MethylPhys_Interface"), capture_output=True, text=True)
    cs = json.load(open(b)).get("cell_shares") or {} if os.path.exists(b) else {}; return {x["cell"]: x["share"] for x in cs.get("cells") or []}
L = [f"Deconvolver with megakaryocytes, synthetic: {len(mixes)} mixes, {len(M)} marker sites, {N} reads per site, {len(R['cells'])} cells, seed 20261104",
     "mix | planted MK, EP | read against REF: MK, EP | bars | read against the committed reference: EP, and the three largest other than G and MO"]; ok = []
for j, mx in enumerate(mixes):
    p = sum(f * mu[c].fillna(0) for c, f in mx.items()).clip(0, 1); a = np.zeros((SV.HG19_CPGS, 2), np.uint16)
    ii = idx.reindex(p.index).dropna().astype(np.int64); kk = rng.binomial(N, p.loc[ii.index].to_numpy())
    a[ii.to_numpy() - 1, 0] = kk; a[ii.to_numpy() - 1, 1] = N; f = os.path.join(W, f"mix{j}.lbeta"); a.tofile(f)
    s1 = read(f, j, "mk", REF); s0 = read(f, j, "v1", None); os.remove(f)
    e = [abs(s1.get(c, 0) - mx.get(c, 0)) for c in (MK, EP)] if s1 else [None, None]; w = bool(s1) and max(e) <= 0.05; ok.append(w)
    top = sorted(((c, v) for c, v in s0.items() if c not in (G, MO)), key=lambda t: -t[1])[:3]
    L.append(f"{j} | {mx.get(MK, 0):.2f}, {mx.get(EP, 0):.2f} | {s1.get(MK, 0):.3f}, {s1.get(EP, 0):.3f} | {'met' if w else 'not met'} | {s0.get(EP, 0):.3f}; " + ", ".join(f"{c} {v:.3f}" for c, v in top))
L.append(f"synthetic: {'met' if all(ok) else 'not met'}"); open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
