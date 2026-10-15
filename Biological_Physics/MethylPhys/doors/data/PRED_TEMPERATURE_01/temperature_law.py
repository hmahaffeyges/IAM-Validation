"""PRED-TEMPERATURE-01 (PHY-057), written 2026-10-15 before any data is searched. KISS: IAM's Boltzmann premise says a held mark's holding
energy E (in joules) is fixed and kT is the yardstick, so ln[(1 - eps)/eps] = E / kT scales as 1/T. Nothing fitted: the 37 C values are the
measured line (loose_end_line.json: firm 0.0212, loose 0.0497; island median 0.0124), and the prediction at temperature T is
eps(T) = 1 / (1 + exp(ln[(1 - eps37)/eps37] * 310.15 / T)). Writes temperature_law.json."""
import json, os, math
H = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(H, "..", "..", "..", "..", "..", "development", "loose_end_2026-10-15", "loose_end_line.json")))
base = {"firm": B["firm_end"], "loose": B["loose_end"], "island": B["island_median"]}
def eps(e37, T): return 1 / (1 + math.exp(math.log((1 - e37) / e37) * 310.15 / T))
temps = {"hibernation torpor 5 C": 5, "cold fish 10 C": 10, "18 C": 18, "28 C": 28, "human 37 C": 37, "dog 38.5 C": 38.5, "bird 41 C": 41}
O = {"basis_37C": base, "predicted": {k: {kind: round(eps(v, 273.15 + t), 4) for kind, v in base.items()} for k, t in temps.items()},
     "within_species_logodds_ratio": {"18 C vs 28 C": round(301.15 / 291.15, 4), "torpor 5 C vs active 37 C": round(310.15 / 278.15, 4)}}
json.dump(O, open(os.path.join(H, "temperature_law.json"), "w"), indent=1); print(json.dumps(O, indent=1))
