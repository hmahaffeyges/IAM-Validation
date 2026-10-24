"""DEV-METAA-HEALTHY-01 step 3 version 2 (written and committed 2026-10-24 before its counts exist, AFTER version 1 was scored, not met, and its
diagnostic read; disclosed). One change: the held error is counted at Met-A's neutrophil identity sites themselves (run_sample.py
--dev-count-sites neutrophils; stage_q2_selfref.count_pat_sites, Stage Q2's own rules), not over the whole genome. A lone error at a site
reads r = eps_site (1 - eps_nb)^2, its neighbors being mostly backbone CpGs, so eps_site = r / (1 - eps_nb)^2 with eps_nb the same
samples' genome-wide per-call rate on the same side (version 1's). The derivation, the sites, the arrays and bar 1 are version 1's.
Was version 1:
DEV-METAA-HEALTHY-01 step 3 scorer (written and committed before any granulocyte molecule is counted). The derived neutrophil mark:
the three granulocyte samples' held errors (Stage Q2 counts kept by count_atlas_01.py; pooled by opportunities; per-call rate from the lone-
error rate r = eps (1 - eps)^2, as layer 1) through stage_m_met_a.derived_baseline at Met-A's 6,000 commissioned neutrophil identity sites
(metA_floors_v1_3.json), each site held or held absent by IAM-Atlas v2.1's neutrophil level (>= 0.5). Then the six commissioned arrays read
against it: Met-A(derived) = A_loo_frozen_sites x measured mark / derived mark (metA_floors_v1_3_loo.csv, the chain's readings on record).
Bar 1 (the instrument's job): all six read Normal, 0.95 to 1.05, against the derived mark. Usage: python3 step3_score.py FOLDER [OUT.txt]"""
import sys, os, glob, json, numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import brentq
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain"); sys.path.insert(0, CH); import stage_m_met_a as M
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "step3_v2_summary.txt")
S = pd.read_csv(os.path.join(H, "step3_samples.csv")).set_index("gsm"); C = {}; CS = {}
for f in glob.glob(os.path.join(src, "**", "GSM*_bundle.json"), recursive=True):
    g = os.path.basename(f).split("_")[0]; c = (json.load(open(f)).get("iam_a_count_only") or {}).get("count") or {}
    cs = (json.load(open(f)).get("iam_a_count_only") or {}).get("count_sites") or {}
    if g in S.index and c.get("held_opp") and c.get("gain_opp") and cs.get("held_opp") and cs.get("gain_opp"): C[g] = c; CS[g] = cs
per_call = lambda r: brentq(lambda e: e * (1 - e) ** 2 - r, 0, 1 / 3)
el = per_call(sum(c["held_err"] * c["held_opp"] for c in C.values()) / sum(c["held_opp"] for c in C.values()))
eg = per_call(sum(c["gain_err"] * c["gain_opp"] for c in C.values()) / sum(c["gain_opp"] for c in C.values()))
el_g, eg_g = el, eg   # version 1's genome-wide per-call rates, the neighbors' rate
el = (sum(c["held_iso"] for c in CS.values()) / sum(c["held_opp"] for c in CS.values())) / (1 - el_g) ** 2
eg = (sum(c["gain_iso"] for c in CS.values()) / sum(c["gain_opp"] for c in CS.values())) / (1 - eg_g) ** 2
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
AT = os.path.join(H, "..", "..", "..", "atlas", "v2.1"); ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
for n in (1, 2, 3):
    t = pq.ParquetFile(os.path.join(AT, f"mu_part{n}.parquet"))
    if "neutrophils" in t.schema.names: u = pq.read_table(os.path.join(AT, f"mu_part{n}.parquet"), columns=["neutrophils"]).column(0).to_numpy()
lv = pd.Series(np.where(u == 65535, np.nan, u / 1e4), index=ids)   # stored as uint16 x 10^4, 65535 = not measured
st = lv.reindex(F["sites"]); held = (st.dropna() >= 0.5).to_numpy()   # sites the atlas never measured in neutrophils take the proportion of the rest (stated before the run)
B = M.derived_baseline(held, el, eg); L = pd.read_csv(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3_loo.csv"))
L = L[L.cell == "neutrophils"]; L["A_derived"] = L.A_loo_frozen_sites * F["floor"] / B["H"]; ok = int(L.A_derived.between(0.95, 1.05).sum())
T = [f"DEV-METAA-HEALTHY-01 step 3 version 2 (held error at the identity sites): genome-wide per-call loss {el_g:.5f}, gain {eg_g:.5f} | at the sites: loss opp {sum(c['held_opp'] for c in CS.values())}, gain opp {sum(c['gain_opp'] for c in CS.values())}",
     f"{len(C)} of 3 granulocyte samples counted ({', '.join(sorted(C))}); sites with an atlas level {int(st.notna().sum())} of {len(st)}",
     f"eps_loss {el:.5f} | eps_gain {eg:.5f} | array-scale level held {B['level_held']:.4f}, held absent {B['level_absent']:.4f} | sites held {B['n_held']}, absent {B['n_absent']}",
     f"derived mark {B['H']:.5f} bits | measured mark {F['floor']:.5f} bits (6 arrays) | derived / measured {B['H'] / F['floor']:.4f}",
     "array | Met-A against the measured mark (frozen sites) | against the derived mark"]
T += [f"{r.ref} | {r.A_loo_frozen_sites:.4f} | {r.A_derived:.4f}" for r in L.itertuples()]
T.append(f"bar 1 (all six Normal, 0.95-1.05, against the derived mark): {ok} of {len(L)}: {'met' if ok == len(L) == 6 and len(C) == 3 else 'not met'}")
open(out, "w").write("\n".join(T) + "\n"); print("\n".join(T))
