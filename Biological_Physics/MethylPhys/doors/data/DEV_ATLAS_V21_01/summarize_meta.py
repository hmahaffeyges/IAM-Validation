"""Summarize cached GEO metadata (meta_cache/*.json) per series: contributors, institute, platforms, per-sample raw-file names,
RAW.tar member count/size from filelist.txt, ENA run bytes, characteristic value counts. Metadata only."""
import json, glob, re, collections, io
import pandas as pd

ATLAS_PEOPLE = {  # v2 source laboratories (atlas v2 roster sources) and reference groups
    "Hebrew U (Loyfer/Dor/Glaser/Shemer/Kaplan)": ["loyfer", "dor", "glaser", "shemer", "kaplan", "zick", "magenheim"],
    "Dartmouth/UCSF/Brown (Salas/Koestler/Christensen/Kelsey/Wiencke/Butler)": ["salas", "koestler", "christensen", "kelsey", "wiencke", "butler", "hansen"],
    "Moss": ["moss"],
    "Stanford Majeti (GSE63409)": ["majeti", "jung", "dethlefsen"],
    "Salk/UCSD Tian/Ecker": ["tian", "ecker"],
    "ENCODE": ["encode"],
}
ATLAS_INST = ["hebrew university", "hadassah", "dartmouth", "ucsf", "university of california, san francisco", "brown university",
              "salk", "stanford"]

def people_flags(contribs, inst):
    hits = []
    low = [c.lower() for c in contribs]
    for grp, names in ATLAS_PEOPLE.items():
        for c in low:
            last = c.split(",")[-1].strip() if "," in c else c.split()[-1]
            # contributor strings in SOFT are 'First,M,Last'
            parts = [p.strip() for p in c.split(",")]
            last = parts[-1] if parts else c
            if last in names:
                hits.append(f"{grp}:{c}")
    ih = [i for i in ATLAS_INST if i in inst.lower()]
    return hits, ih

def summarize(fn):
    d = json.load(open(fn))
    s = d["series"]; S = d["samples"]
    contribs = s.get("Series_contributor", [])
    inst = "; ".join(sorted(set(s.get("Series_contact_institute", []))))
    hits, ih = people_flags(contribs, inst)
    plats = collections.Counter(x.get("Sample_platform_id", ["?"])[0] for x in S)
    rows = []
    for x in S:
        sup = [u.split("/")[-1] for u in x.get("Sample_supplementary_file", []) if u and u != "NONE"]
        rel = x.get("Sample_relation", [])
        sra = [r.split("=")[-1] for r in rel if "SRA:" in r]
        ch = x.get("Sample_characteristics_ch1", [])
        rows.append({"gsm": x["_id"], "title": x.get("Sample_title", [""])[0], "source": x.get("Sample_source_name_ch1", [""])[0],
                     "platform": x.get("Sample_platform_id", [""])[0], "char": " | ".join(ch), "supp": ";".join(sup),
                     "n_idat": sum(f.lower().endswith((".idat", ".idat.gz")) for f in sup), "sra": ";".join(sra),
                     "molecule": x.get("Sample_molecule_ch1", [""])[0]})
    T = pd.DataFrame(rows)
    fl = d.get("filelist")
    tar_bytes, tar_members, tar_idat = None, None, None
    if fl:
        F = pd.read_csv(io.StringIO(fl), sep="\t")
        a = F[F["#Archive/File"] == "Archive"]
        tar_bytes = int(a.Size.sum()) if len(a) else 0
        mem = F[F["#Archive/File"] == "File"]
        tar_members = len(mem)
        tar_idat = int(mem.Name.str.lower().str.contains(r"\.idat").sum())
    ena_bytes, ena_runs = None, None
    if d.get("ena_runs"):
        E = pd.read_csv(io.StringIO(d["ena_runs"]), sep="\t")
        ena_runs = len(E)
        b = E.fastq_bytes.astype(str).str.split(";").apply(lambda v: sum(int(z) for z in v if z.isdigit()))
        ena_bytes = int(b.sum())
    return {"acc": d["acc"], "title": s.get("Series_title", [""])[0], "pubmed": ";".join(s.get("Series_pubmed_id", [])),
            "public": s.get("Series_status", [""])[0], "contributors": "; ".join(contribs), "institute": inst,
            "contact": s.get("Series_contact_name", [""])[0], "atlas_people": "; ".join(hits), "atlas_inst": "; ".join(ih),
            "platforms": dict(plats), "n_samples": len(S), "n_samples_with_2idat": int((T.n_idat >= 2).sum()),
            "n_samples_with_sra": int((T.sra != "").sum()), "tar_bytes": tar_bytes, "tar_members": tar_members, "tar_idat": tar_idat,
            "ena_runs": ena_runs, "ena_fastq_bytes": ena_bytes, "design": " ".join(s.get("Series_overall_design", []))[:1500],
            "summary": " ".join(s.get("Series_summary", []))[:1500], "relation": "; ".join(s.get("Series_relation", []))}, T

if __name__ == "__main__":
    out = []
    for fn in sorted(glob.glob("meta_cache/GSE*.json")):
        r, T = summarize(fn)
        T.to_csv(f"meta_cache/{r['acc']}_samples.csv", index=False)
        out.append(r)
    pd.DataFrame(out).to_csv("series_summary.csv", index=False)
    print(len(out))
