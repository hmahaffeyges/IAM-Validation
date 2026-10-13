"""Cell names exactly as Loyfer (GSE186458) and Moss (GSE122126) recorded them (author, 2026-10-13: 'name these cells exactly the
same as Loyfer and Moss did'). Nothing is typed by hand.

  fetch: one GEO request per series (sample records, brief text), saved verbatim as geo_GSE186458_samples.txt and
         geo_GSE122126_samples.txt with their checksums in cell_names_v21_snapshot.json.
  build: for every atlas cell, every Loyfer or Moss sample the atlas uses (atlas/v2/inputs/roster_samples.csv), and that sample's
         GEO fields verbatim: title, source name, and each characteristic (Loyfer: cell type, tissue; Moss: sample type).
         Writes cell_names_v21.csv (one row per sample) and cell_names_v21_by_cell.csv (one row per atlas cell, with every distinct
         source name). An atlas cell with no Loyfer or Moss sample is listed with its own name and the note that neither atlas has it.
  check: refuses if a snapshot's checksum changed, a GSM the atlas uses is missing from the snapshot, or the outputs differ from a
         rebuild.
Usage: python3 cell_names_v21.py fetch | build | check
"""
import os, sys, json, hashlib, re, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
RS = os.path.join(ROOT, "Biological_Physics", "MethylPhys", "atlas", "v2", "inputs", "roster_samples.csv")
SERIES = {"Loyfer2023": "GSE186458", "Moss2018": "GSE122126"}
snap = lambda g: os.path.join(HERE, f"geo_{g}_samples.txt")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def fetch():
    J = {}
    for src, g in SERIES.items():
        t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": g, "targ": "gsm", "form": "text", "view": "brief"}, timeout=600).text
        open(snap(g), "w", encoding="utf-8").write(t); J[g] = {"sha256": sha(snap(g)), "samples": t.count("^SAMPLE = "), "fetched": "2026-10-13"}
    json.dump(J, open(os.path.join(HERE, "cell_names_v21_snapshot.json"), "w"), indent=1); print(J)
def parse(g):
    out = {}
    for b in open(snap(g), encoding="utf-8").read().replace("\r", "").split("^SAMPLE = ")[1:]:
        gsm = b.split("\n")[0].strip(); get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
        ch = {c.split(":", 1)[0].strip(): c.split(":", 1)[1].strip() for c in get("!Sample_characteristics_ch1") if ":" in c}
        out[gsm] = dict(geo_title=(get("!Sample_title") or [""])[0], geo_source_name=(get("!Sample_source_name_ch1") or [""])[0],
                        geo_characteristics=json.dumps(ch, ensure_ascii=False), geo_cell_type=ch.get("cell type", ch.get("sample type", "")),
                        geo_tissue=ch.get("tissue", ""))
    return out
def build():
    R = pd.read_csv(RS); rows = []
    P = {src: parse(g) for src, g in SERIES.items()}
    for _, r in R.iterrows():
        if r.source in P:
            m = P[r.source].get(str(r["sample"]).split("_")[0])   # Moss ids carry the chip position (GSM_chip_RxxCxx)
            rows.append(dict(atlas_cell=r.cell, source=r.source, series=SERIES[r.source], gsm=r["sample"], **(m or dict(geo_title="", geo_source_name="", geo_characteristics="", geo_cell_type="", geo_tissue="")), found=m is not None))
    T = pd.DataFrame(rows).sort_values(["atlas_cell", "source", "gsm"])
    T.to_csv(os.path.join(HERE, "cell_names_v21.csv"), index=False)
    cells = sorted(set(R.cell))
    B = []
    for c in cells:
        g = T[T.atlas_cell == c]
        f = lambda s, k: " | ".join(sorted(set(g[g.source == s][k]) - {""}))
        B.append(dict(atlas_cell=c, loyfer_cell_type=f("Loyfer2023", "geo_cell_type"), loyfer_tissue=f("Loyfer2023", "geo_tissue"),
                      moss_sample_type=f("Moss2018", "geo_cell_type"), moss_source_name=f("Moss2018", "geo_source_name"),
                      note="" if len(g) else "in neither source atlas; our own name"))
    pd.DataFrame(B).to_csv(os.path.join(HERE, "cell_names_v21_by_cell.csv"), index=False)
    return T, pd.DataFrame(B)
def check():
    P = []; J = json.load(open(os.path.join(HERE, "cell_names_v21_snapshot.json")))
    for g, x in J.items():
        if not os.path.exists(snap(g)) or sha(snap(g)) != x["sha256"]: P.append(f"GEO snapshot {g} changed or missing")
    if not P:
        old = [open(os.path.join(HERE, f)).read() for f in ("cell_names_v21.csv", "cell_names_v21_by_cell.csv")]
        T, _ = build(); new = [open(os.path.join(HERE, f)).read() for f in ("cell_names_v21.csv", "cell_names_v21_by_cell.csv")]
        if old != new: P.append("cell_names outputs differ from a rebuild")
        if not T.found.all(): P.append(f"{int((~T.found).sum())} atlas samples missing from the GEO snapshot")
    print("cell_names_v21: " + ("ok" if not P else "REFUSED: " + "; ".join(P))); return 1 if P else 0
if __name__ == "__main__":
    m = sys.argv[1]
    if m == "fetch": fetch()
    elif m == "build":
        T, B = build(); print(len(T), "samples |", int(T.found.sum()), "found in GEO |", len(B), "atlas cells |", int((B.note != "").sum()), "in neither atlas")
    else: sys.exit(check())
