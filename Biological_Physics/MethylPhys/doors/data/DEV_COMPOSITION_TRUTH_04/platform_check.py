"""DEV-COMPOSITION-TRUTH-04 platform check (author decision 2026-10-11: GPL29753 counts as EPIC v1 for bar 7 only if every site the commissioned
chain reads is on it). Manifest only (GEO GPL29753_platform.txt.gz), no array read. Site lists from the chain's runtime matrices: Met-A neutrophil
floor (metA_floors_v1_3.json), composition markers and neutrophil sites (blood_composition_EPIC_v1.json), noise sites (noise_sites_EPIC_v1.json).
A site counts as present when its cg ID is on GPL29753 with Probe_Source EPIC. Writes platform_check_output.txt.
Usage: python3 platform_check.py GPL29753_platform.txt.gz"""
import os, sys, json, gzip
HERE = os.path.dirname(os.path.abspath(__file__)); RM = os.path.join(HERE, "../../../chain/Runtime Matrices/Met_A_Floors")
ids, src = set(), {}
with gzip.open(sys.argv[1], "rt", errors="replace") as f:
    head = None
    for l in f:
        if l.startswith(("#", "!", "^")): continue
        p = l.rstrip("\n").split("\t")
        if head is None:                       # the file opens with a column-description block ("ID<TAB>Unique probe identifier"); the table header has Probe_Source
            if p[0] == "ID" and "Probe_Source" in p: head = p; i_id = head.index("ID"); i_s = head.index("Probe_Source")
            continue
        ids.add(p[i_id]); src[p[i_id]] = p[i_s] if i_s is not None and i_s < len(p) else ""
def sites(o):
    out = set()
    if isinstance(o, dict): [out.update(sites(v)) for v in o.values()]
    elif isinstance(o, list):
        for x in o:
            if isinstance(x, str) and x[:2] in ("cg", "ch"): out.add(x)
            else: out.update(sites(x))
    return out
B = json.load(open(os.path.join(RM, "blood_composition_EPIC_v1.json")))
F = json.load(open(os.path.join(RM, "metA_floors_v1_3.json")))
lists = {"Met-A floor sites (metA_floors_v1_3)": sites(F), "composition markers": set(B["markers"]) if isinstance(B["markers"][0], str) else sites(B["markers"]),
         "composition neutrophil sites": set(B["neutrophil_sites"]), "noise sites": sites(json.load(open(os.path.join(RM, "noise_sites_EPIC_v1.json"))))}
out = [f"GPL29753 probes: {len(ids):,}; EPIC-source: {sum(1 for v in src.values() if v.upper().startswith('EPIC')):,}"]
allok = True
for k, s in lists.items():
    miss = sorted(x for x in s if x not in ids); nonepic = sorted(x for x in s if x in ids and not src.get(x, "").upper().startswith("EPIC"))
    allok &= not miss and not nonepic
    out.append(f"{k}: {len(s):,} sites, missing {len(miss)}, present but not EPIC-source {len(nonepic)}" + (f" (first missing: {', '.join(miss[:5])})" if miss else ""))
out.append("PLATFORM CHECK " + ("PASSED: every site is on GPL29753" if allok else "FAILED"))
open(os.path.join(HERE, "platform_check_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
