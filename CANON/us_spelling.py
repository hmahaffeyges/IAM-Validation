"""American spelling in donor-facing text (author rule 2026-10-12: British spelling makes readers think the work is not his own).
check: lists British spellings in the donor-facing files (prose only; code identifiers such as tumour_gsm are not prose and are not touched).
fix: rewrites them to American spelling in place. Donor-facing: README.md, FOLLOW.md, Biological_Physics/README.md, fundraiser/*.txt,
and the 'plain', 'title' and 'headline' fields of CANON/results_register.json. Usage: python3 CANON/us_spelling.py check | fix"""
import os, re, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
MAP = {"neighbourhood": "neighborhood", "neighbourhoods": "neighborhoods", "neighbouring": "neighboring", "neighbour": "neighbor", "neighbours": "neighbors", "centred": "centered", "tumour": "tumor", "tumours": "tumors", "colour": "color", "colours": "colors", "behaviour": "behavior", "behaviours": "behaviors",
       "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing", "centre": "center", "centres": "centers", "modelling": "modeling",
       "modelled": "modeled", "labelled": "labeled", "labelling": "labeling", "ageing": "aging", "programme": "program", "programmes": "programs",
       "favour": "favor", "organised": "organized", "organise": "organize", "recognised": "recognized", "recognise": "recognize", "realise": "realize",
       "realised": "realized", "minimise": "minimize", "optimise": "optimize", "optimised": "optimized", "haematological": "hematological",
       "oesophageal": "esophageal", "paediatric": "pediatric", "travelled": "traveled", "fibre": "fiber", "fibres": "fibers", "grey": "gray",
       "characterised": "characterized", "normalised": "normalized", "summarise": "summarize", "summarised": "summarized", "artefact": "artifact",
       "artefacts": "artifacts", "cancelled": "canceled", "judgement": "judgment", "licence": "license", "defence": "defense", "favourable": "favorable"}
RX = re.compile(r"(?<![_\w])(" + "|".join(sorted(MAP, key=len, reverse=True)) + r")(?![_\w])", re.I)
def us(m):
    w = m.group(0); r = MAP[w.lower()]; return r.capitalize() if w[0].isupper() else r
FILES = ["README.md", "FOLLOW.md", "Biological_Physics/README.md"] + sorted(os.path.join("fundraiser", f) for f in os.listdir("fundraiser") if f.endswith(".txt"))
def prose(t):   # skip inline code and link targets
    return re.sub(r"`[^`]*`|\]\([^)]*\)|https?://\S+", "", t)
mode = sys.argv[1] if len(sys.argv) > 1 else "check"; bad = []
for f in FILES:
    if not os.path.exists(f): continue
    t = open(f, encoding="utf-8").read()
    if mode == "fix":
        parts = re.split(r"(`[^`]*`|\]\([^)]*\)|https?://\S+)", t); t2 = "".join(p if i % 2 else RX.sub(us, p) for i, p in enumerate(parts))
        if t2 != t: open(f, "w", encoding="utf-8").write(t2)
    else: bad += [f"{f}: {w}" for w in sorted(set(RX.findall(prose(t))))]
R = json.load(open("CANON/results_register.json", encoding="utf-8")); ch = False
for e in R["entries"]:
    for k in ("plain", "title", "headline"):
        if not isinstance(e.get(k), str): continue
        if mode == "fix":
            n = RX.sub(us, e[k]); ch |= n != e[k]; e[k] = n
        else: bad += [f"results_register {e['id']}.{k}: {w}" for w in sorted(set(RX.findall(prose(e[k]))))]
if mode == "fix" and ch: json.dump(R, open("CANON/results_register.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False); open("CANON/results_register.json", "a").write("\n")
if mode == "check":
    print("\n".join(bad) if bad else "us_spelling: ok"); sys.exit(1 if bad else 0)
