"""DEV-IAMA-NEURON-01 cortical-neuron reference (rule in DEV_IAMA_NEURON_01.md, 'Reference rule', written before the donor table was read).
Reads neuron_eps_01.csv (Stage Q whole-file copy error per sample, with donor fields from Loyfer et al. 2023 Table S1).
Cortical neurons only (cerebellum is another cell type; the sample with no region is not placed in cortex by the donor table). One cell type:
a sample Loyfer estimates at < 0.80 neurons (Table S10 purity) is not neurons; only Cortex-Neuron-Z000000TF (0.478) falls there, and every other
sample is >= 0.898, so any cut between them gives the same set. Donor value = mean eps over that donor's samples (one donor gave three).
Leave-one-donor-out IAM-A_rel = H(eps_d) / H(mean eps of the other donors). Printed twice: all cortical donors, and without the donors whose
recorded cause of death is suicide (descriptive, author decision 2026-10-12). Writes neuron_reference_01_output.txt."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = pd.read_csv(os.path.join(HERE, "neuron_eps_01.csv"))
H = lambda e: -(e * np.log2(e) + (1 - e) * np.log2(1 - e))
C = R[(R.region == "Cortex") & (R.purity >= 0.80) & (R.q0 == "PROCEED")]
out = [f"samples {len(R)}; cortical, >= 0.80 neurons, Q0 PROCEED: {len(C)} from {C.donor.nunique()} donors",
       "excluded: " + "; ".join(f"{r.sample.split('_',1)[1]} ({'not cortex' if r.region != 'Cortex' else 'purity %.3f' % r.purity})" for r in R[~R.index.isin(C.index)].itertuples())]
def ref(Cx, label):
    d = Cx.groupby("donor").agg(eps=("eps", "mean"), age=("age", "first"), sex=("sex", "first"), death=("death", "first"), n=("eps", "size"))
    d["IAMA_rel_lodo"] = [H(d.eps[k]) / H(d.eps.drop(k).mean()) for k in d.index]
    out.append(f"{label}: donors {len(d)}, reference eps {d.eps.mean():.5f}, H_ref {H(d.eps.mean()):.4f} bits; leave-one-donor-out IAM-A_rel "
               f"{d.IAMA_rel_lodo.min():.3f}-{d.IAMA_rel_lodo.max():.3f}; inside Normal 0.95-1.05: {int(d.IAMA_rel_lodo.between(.95, 1.05).sum())} of {len(d)}")
    out.extend("   " + f"{k}: eps {r.eps:.5f} (n {r.n}), age {r.age}, {r.sex}, {r.death if isinstance(r.death, str) and r.death else '-'}, IAM-A_rel {r.IAMA_rel_lodo:.3f}" for k, r in d.iterrows())
ref(C, "all cortical donors")
ref(C[C.death.fillna("") != "Suicide"], "without suicide donors (descriptive)")
open(os.path.join(HERE, "neuron_reference_01_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
