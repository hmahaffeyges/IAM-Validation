"""Derivation register A15 (exploration on spent counts, found after the fact; 2026-11-13): is the unmarked side's stray-mark rate the
writer's own selection error? Where the physics is: one equilibrium selection step by the writer (DNMT1) picks the right site over a wrong
one with odds D, so a wrong write happens with probability 1/(1+D) (Boltzmann: D = e^{dE/kT}); one powered proofreading step
(Hopfield 1974, Ninio 1975) would square it. D is measured (CANON four_domains gate_error: two enzymes, 47.4 to 80). IAM's Law is not in
this number: it is measured selection physics, set beside the held side (A6), which M prices.
Counts: gain_err (chain Stage Q2: a lone methylated call between two unmethylated calls on molecules held unmarked), healthy samples of
the atlas laboratory (percall_out, neurons set aside) and of the referee laboratory (referee_counts). Limits: gain_err also carries the
meter's own false marks (incomplete conversion, sequencing error), which add to the writer's error; D is measured in vitro.
Writes unmarked_writer_summary.txt."""
import os, json, glob, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
def rates(pattern):
    v = []
    for f in glob.glob(os.path.join(H, pattern)):
        c = json.load(open(f))
        for x in (c if isinstance(c, list) else [c]):
            if isinstance(x, dict) and x.get("gain_err") is not None and "neuron" not in str(x.get("cell", "")).lower():
                v.append(float(x["gain_err"]))
    return np.array(v)
D = np.array([47.4, 80.0]); one, two = 1 / (1 + D), (1 / (1 + D)) ** 2
L = [f"single selection 1/(1+D), D 47.4-80: {one.min():.4f}-{one.max():.4f}; with one proofreading step: {two.min():.5f}-{two.max():.5f}"]
for name, pat in (("atlas laboratory (healthy, all types)", "percall_out/*.json"), ("referee laboratory (healthy blood)", "referee_counts/*_count.json")):
    g = rates(pat); inside = float(np.mean((g >= one.min()) & (g <= one.max())))
    L.append(f"{name}: n {len(g)}, median {np.median(g):.4f}, 5-95 % {np.percentile(g, 5):.4f}-{np.percentile(g, 95):.4f}; "
             f"share inside the single-selection band {100 * inside:.0f} %; median over the proofread level {np.median(g) / two.max():.0f}x")
L.append("Reading: both laboratories sit inside the single-selection band and far above the proofread level: the unmarked side reads like one"
         " selection step with no proofreading. After the fact; the meter's own false marks are not yet subtracted (next: each library's"
         " conversion control).")
open(os.path.join(H, "unmarked_writer_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
