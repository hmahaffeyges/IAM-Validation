"""Atlas v2.1 build sample list (readiness L2), one row per array, with the 15 adopted decisions (DEV_ATLAS_V21_01.md) applied by rule.

Every roster row with role 'build' is expanded to its arrays. Each array gets a state:
  IN           enters the build;
  CONDITIONAL  enters only if its stated condition is met at a data step (twin test, purity cut, paper check); the condition is written
               here before any array is downloaded, and conditions_v21.json (L8) records each outcome;
  OUT          excluded, with the decision that excludes it.
Decisions applied: D1/D6 granulocytes only if marker-selected (CD15+ or CD16+); D2 at most 12 donors per source per cell, the first 12
donors in ascending GSM order (fixed here, before any data); D4 EPIC v2 deferred; D5 sperm, thymus, cord-blood HSC, placenta left out;
D7 GSE306226 astrocytes flagged; D9 E-MTAB-4957 flagged pediatric; D10 enriched-not-sorted epithelia (GSE146552, GSE137716) conditional
on the purity cut; D11 adipocyte WGBS deferred. IDAT names come from the committed GEO snapshots (geo_snapshots/) and, for E-MTAB-4957,
its committed ArrayExpress SDRF. Writes atlas_v21_build_samples.csv. Usage: python3 build_samples_v21.py
"""
import os, re, gzip, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "atlas_v21_roster.csv")); B = R[R.role == "build"]
def snap(g):
    p = os.path.join(HERE, "geo_snapshots", f"{g}.txt.gz"); out = {}
    if not os.path.exists(p): return out
    for b in gzip.open(p, "rt", encoding="utf-8").read().replace("\r", "").split("^SAMPLE = ")[1:]:
        gsm = b.split("\n")[0].strip(); get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
        files = [f.split("/")[-1] for f in get("!Sample_supplementary_file") if ".idat" in f.lower()]
        ch = {c.split(":", 1)[0].strip().lower(): c.split(":", 1)[1].strip() for c in get("!Sample_characteristics_ch1") if ":" in c}
        donor = next((ch[k] for k in ch if re.search(r"donor|subject|individual|patient|^id$", k)), "")
        sex = next((ch[k] for k in ch if k in ("sex", "gender")), "")
        out[gsm] = dict(idat=";".join(sorted(files)), donor=donor, sex=sex, platform_id=";".join(get("!Sample_platform_id")))
    return out
def emtab(acc):
    T = pd.read_csv(os.path.join(HERE, f"{acc}.sdrf.txt"), sep="\t"); out = {}
    fcol = [c for c in T.columns if c.startswith("Array Data File")]
    for _, r in T.iterrows():
        k = str(r.iloc[0]); fs = sorted({str(r[c]) for c in fcol if str(r[c]).endswith(".idat")})
        out.setdefault(k, dict(idat="", donor=str(r.get("Characteristics[individual]", "")), sex=str(r.get("Characteristics[sex]", "")), platform_id=acc))
        out[k]["idat"] = ";".join(sorted(set(out[k]["idat"].split(";") + fs) - {""}))
    return out
def state(row):
    s, c, t = str(row.series), str(row.cell_type).lower(), str(row.platform)
    if "EPIC v2" in t: return "OUT", "D4: EPIC v2 deferred to v2.2 (Stage 1 EPIC v2 read check first)"
    if s in ("GSE114753", "GSE273074") or re.search(r"sperm|thymocyte|placent|trophoblast|hofbauer", c): return "OUT", "D5: no v2.1 use needs this cell"
    if s == "GSE111445": return "OUT", "D11: adipocyte WGBS deferred"
    if s in ("GSE68456", "GSE82084"): return "OUT", "D8: cord blood kept out to protect the GSE127824 truth set"
    if "neutrophil" in c and ("granulocyte" in c or "D6" in str(row.status)):
        p = str(row.purification)
        if re.search(r"CD15|CD16", p, re.I): return "CONDITIONAL", "D1: marker-selected granulocytes; enters if each array passes the sample-level twin test"
        if s == "GSE103541": return "CONDITIONAL", "D1: FACS granulocytes; enters only if the paper (PMC8011804) names CD15 or CD16 as the gate, and each array passes the twin test"
        return "OUT", "D1: granulocytes without a stated CD15/CD16 selection"
    # D7 is about the astrocyte fraction only (the roster lists the series' neurons, microglia and oligodendrocytes as firm). Until
    # 2026-10-14 this line made the whole series conditional (found 2026-10-14 while resolving L8); fixed in seal 3.
    if s == "GSE306226" and "astrocyte" in c: return "CONDITIONAL", "D7: kept only if closer to Tian astrocytes than to any other cell, cross-laboratory r at least 0.986"
    if s in ("GSE146552", "GSE137716"):
        if "ovary" in c and "WAIT" in str(row.cell_type): return "CONDITIONAL", "D10: OSE; enters only if non-epithelial content is under the plan's cut (and the cell leaves WAIT)"
        return "CONDITIONAL", "D10: enriched, not sorted; enters only if non-epithelial content is under the plan's cut"
    if s == "E-MTAB-4957": return "CONDITIONAL", "D9: pediatric, flagged; enters once the 'n'-suffix arrays are identified before the seal"
    return "IN", "firm, entry rule met on metadata (twin test at Stage 1 applies to every array)"
rows = []
for _, r in B.iterrows():
    ids = [x for x in str(r.samples).replace(" ", "").split(";") if x and x != "nan"]
    meta = emtab(r.series) if str(r.series).startswith("E-MTAB") else snap(r.series)
    st, why = state(r)
    for x in ids:
        m = meta.get(x, {})
        rows.append(dict(cell=re.sub(r"^NEW: ", "", str(r.cell_type)).split(" [")[0].strip().lower(), cell_type_roster=r.cell_type, series=r.series,
                         sample=x, donor=m.get("donor", ""), sex=m.get("sex", ""), platform=r.platform, platform_id=m.get("platform_id", ""),
                         idat_files=m.get("idat", ""), state=st, reason=why, found_in_snapshot=bool(m)))
T = pd.DataFrame(rows)
# D2: at most 12 donors per source per cell, first 12 donors in ascending sample order
capped = []
for (cell, ser), g in T[T.state != "OUT"].groupby(["cell", "series"]):
    don = g.donor.where(g.donor != "", g["sample"])
    keep = sorted(set(don), key=lambda d: g["sample"][don == d].min())[:12]
    drop = g.index[~don.isin(keep)]
    T.loc[drop, ["state", "reason"]] = ["OUT", "D2: beyond 12 donors for this source and cell (first 12 donors in ascending sample order kept)"]; capped += list(drop)
T = T.sort_values(["cell", "series", "sample"]).reset_index(drop=True)
T.to_csv(os.path.join(HERE, "atlas_v21_build_samples.csv"), index=False)
print("arrays", len(T), "|", T.state.value_counts().to_dict(), "| capped by D2", len(capped), "| not found in snapshot", int((~T.found_in_snapshot).sum()),
      "| IN/COND without IDAT names", int(((T.state != "OUT") & (T.idat_files == "") & ~T.platform.astype(str).str.contains("seq", case=False)).sum()))
print(T[T.state != "OUT"].groupby("cell").size().sort_values(ascending=False).head(40).to_dict())
