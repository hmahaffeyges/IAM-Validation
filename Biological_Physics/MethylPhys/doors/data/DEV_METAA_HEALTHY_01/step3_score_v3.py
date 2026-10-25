"""DEV-METAA-HEALTHY-01 step 3 version 3 (written and committed 2026-10-24 before its counts exist, AFTER versions 1 and 2 were scored, not
met; disclosed). Built on what was already stated (lesson C41): Met-A's identity sites are chosen at a mean of 0.75 to 0.95 (or 0.05 to 0.25),
so their level is set by the share of copies that do not hold the mark as well as by errors on held copies: level = held share x held level
+ loose share x loose level + absent share x absent level. Each term is counted on the granulocytes' own molecules at the sites
(run_sample.py --dev-count-sites neutrophils; stage_q2_selfref.count_pat_site_classes), the site level is their sum (every call), put on the
array scale through the atlas's own inverse transfer, and the mark follows (stage_m_met_a.derived_baseline, with 1 - eps_loss and eps_gain
replaced by the counted levels). Nothing fitted; this is a baseline MEASURED from the cell's own molecules, not yet derived: the derivation is
the physics of the loose share at these sites (open). Known differences, stated before: granulocytes (neutrophils with some eosinophils and
basophils) stand in for purified neutrophils; this is the third version on the same three samples and six arrays, so any pass is development
until confirmed on samples that played no part. The sites, the six arrays and bar 1 are version 1's.
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
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "step3_v3_summary.txt")
S = pd.read_csv(os.path.join(H, "step3_samples.csv")).set_index("gsm"); C = {}; K = {}
for f in glob.glob(os.path.join(src, "**", "GSM*_bundle.json"), recursive=True):
    g = os.path.basename(f).split("_")[0]; c = (json.load(open(f)).get("iam_a_count_only") or {}).get("count") or {}
    k = (json.load(open(f)).get("iam_a_count_only") or {}).get("count_site_classes") or {}
    if g in S.index and k.get("held", {}).get("calls") and k.get("absent", {}).get("calls"): C[g] = c; K[g] = k
per_call = lambda r: brentq(lambda e: e * (1 - e) ** 2 - r, 0, 1 / 3)
el = per_call(sum(c["held_err"] * c["held_opp"] for c in C.values()) / sum(c["held_opp"] for c in C.values()))
eg = per_call(sum(c["gain_err"] * c["gain_opp"] for c in C.values()) / sum(c["gain_opp"] for c in C.values()))
pool = lambda side, key: sum(K[g][side][key] for g in K)
lh = sum(K[g]["held"]["level"] * K[g]["held"]["calls"] for g in K) / pool("held", "calls"); la = sum(K[g]["absent"]["level"] * K[g]["absent"]["calls"] for g in K) / pool("absent", "calls")
dec = {side: {c: round(sum(K[g][side][c + "_calls"] for g in K) / sum(K[g][side][x + "_calls"] for g in K for x in ("held", "absent", "loose")), 4) for c in ("held", "absent", "loose")} for side in ("held", "absent")}
el, eg = 1 - lh, la   # the counted site levels enter derived_baseline in place of 1 - eps_loss and eps_gain
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
AT = os.path.join(H, "..", "..", "..", "atlas", "v2.1"); ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
for n in (1, 2, 3):
    t = pq.ParquetFile(os.path.join(AT, f"mu_part{n}.parquet"))
    if "neutrophils" in t.schema.names: u = pq.read_table(os.path.join(AT, f"mu_part{n}.parquet"), columns=["neutrophils"]).column(0).to_numpy()
lv = pd.Series(np.where(u == 65535, np.nan, u / 1e4), index=ids)   # stored as uint16 x 10^4, 65535 = not measured
st = lv.reindex(F["sites"]); held = (st.dropna() >= 0.5).to_numpy()   # sites the atlas never measured in neutrophils take the proportion of the rest (stated before the run)
B = M.derived_baseline(held, el, eg); L = pd.read_csv(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3_loo.csv"))
L = L[L.cell == "neutrophils"]; L["A_derived"] = L.A_loo_frozen_sites * F["floor"] / B["H"]; ok = int(L.A_derived.between(0.95, 1.05).sum())
T = [f"DEV-METAA-HEALTHY-01 step 3 version 3 (molecule classes at the identity sites): sequencing-scale level held {lh:.4f} ({pool('held', 'calls')} calls), held absent {la:.4f} ({pool('absent', 'calls')} calls)",
     f"call shares by molecule class (held, held absent, loose): at held sites {dec['held']} | at held-absent sites {dec['absent']}",
     f"{len(C)} of 3 granulocyte samples counted ({', '.join(sorted(C))}); sites with an atlas level {int(st.notna().sum())} of {len(st)}",
     f"eps_loss {el:.5f} | eps_gain {eg:.5f} | array-scale level held {B['level_held']:.4f}, held absent {B['level_absent']:.4f} | sites held {B['n_held']}, absent {B['n_absent']}",
     f"derived mark {B['H']:.5f} bits | measured mark {F['floor']:.5f} bits (6 arrays) | derived / measured {B['H'] / F['floor']:.4f}",
     "array | Met-A against the measured mark (frozen sites) | against the derived mark"]
T += [f"{r.ref} | {r.A_loo_frozen_sites:.4f} | {r.A_derived:.4f}" for r in L.itertuples()]
T.append(f"bar 1 (all six Normal, 0.95-1.05, against the derived mark): {ok} of {len(L)}: {'met' if ok == len(L) == 6 and len(C) == 3 else 'not met'}")
open(out, "w").write("\n".join(T) + "\n"); print("\n".join(T))
