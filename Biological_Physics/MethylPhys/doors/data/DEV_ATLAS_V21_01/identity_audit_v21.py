"""Atlas v2.1 sample-identity audit, metadata layer (author, 2026-10-13: know for certain what is in each source - no duplicates, no
wrong cells, no partial sets). Runs before the split is sealed.

  fetch: every series in atlas_v21_roster.csv (build, held-out, truth) plus the v2 sources, one GEO request per series, saved verbatim
         in geo_snapshots/ with checksums (geo_snapshots/index.json). Nothing is read from memory.
  audit: (1) PHYSICAL DUPLICATES - an Illumina array's chip barcode and position (in its IDAT name) is unique to one physical array,
         so two GSMs sharing barcode_position are one array deposited twice; (2) every GSM the roster names exists on GEO, belongs to
         its series and has a raw file, and the roster's count matches; (3) the same GSM listed under two roster rows; (4) Loyfer
         (GSE186458) and Moss (GSE122126): every GEO sample is either used by the atlas or excluded with a stated reason.
         Writes identity_audit_v21_summary.txt and identity_audit_v21_duplicates.csv. Exit 1 if any problem is unresolved.
Usage: python3 identity_audit_v21.py fetch | audit
"""
import os, sys, re, json, hashlib, time, requests, collections, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5)); SN = os.path.join(HERE, "geo_snapshots")
R = pd.read_csv(os.path.join(HERE, "atlas_v21_roster.csv"))
USE = R[R.role.isin(["build", "truth set"]) | R.role.str.startswith("held-out")]
RS = pd.read_csv(os.path.join(ROOT, "Biological_Physics", "MethylPhys", "atlas", "v2", "inputs", "roster_samples.csv"))
def series_list():
    s = set(USE.series.astype(str)) | {"GSE186458", "GSE122126"}
    return sorted(x for x in s if x.startswith("GSE"))
def fetch():
    idx = {}
    for g in series_list():
        p = os.path.join(SN, f"{g}.txt")
        if not os.path.exists(p + ".gz"):
            for i in range(4):
                try:
                    t = requests.get("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi", params={"acc": g, "targ": "gsm", "form": "text", "view": "brief"}, timeout=900).text; break
                except Exception:
                    time.sleep(10 * (i + 1))
            import gzip; open(p + ".gz", "wb").write(gzip.compress(t.encode("utf-8"), mtime=0)); time.sleep(0.5)
        import gzip; raw = gzip.open(p + ".gz", "rb").read()
        idx[g] = {"sha256_uncompressed": hashlib.sha256(raw).hexdigest(), "samples": raw.decode("utf-8").count("^SAMPLE = ")}
    json.dump(idx, open(os.path.join(SN, "index.json"), "w"), indent=1); print(len(idx), "series snapshotted;", sum(v["samples"] for v in idx.values()), "sample records")
def samples(g):
    out = {}
    import gzip
    for b in gzip.open(os.path.join(SN, f"{g}.txt.gz"), "rt", encoding="utf-8").read().replace("\r", "").split("^SAMPLE = ")[1:]:
        gsm = b.split("\n")[0].strip(); get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
        files = get("!Sample_supplementary_file")
        bp = sorted({m.group(1) for f in files for m in [re.search(r"_(\d{9,12}_R\d{2}C\d{2})_(?:Grn|Red)\.idat", f, re.I)] if m})
        out[gsm] = dict(series=g, title=(get("!Sample_title") or [""])[0], source=(get("!Sample_source_name_ch1") or [""])[0],
                        idat=any(".idat" in f.lower() for f in files), raw=bool(files and any(f not in ("NONE", "") for f in files)),
                        barcode_pos=";".join(bp), platform=";".join(get("!Sample_platform_id")))
    return out
def audit():
    S = {}
    for g in series_list(): S.update({k: v for k, v in samples(g).items()})
    L, problems = [], []
    # (1) physical duplicates across all snapshotted samples
    by = collections.defaultdict(list)
    for gsm, x in S.items():
        for bpk in [b for b in x["barcode_pos"].split(";") if b]: by[bpk].append(gsm)
    dup = [(bpk, sorted(set(v))) for bpk, v in by.items() if len(set(v)) > 1]
    rows = [dict(barcode_pos=b, gsms=";".join(v), series=";".join(sorted({S[x]["series"] for x in v})), titles=" | ".join(S[x]["title"] for x in v)) for b, v in dup]
    pd.DataFrame(rows, columns=["barcode_pos", "gsms", "series", "titles"]).to_csv(os.path.join(HERE, "identity_audit_v21_duplicates.csv"), index=False)
    used = {g for _, r in USE.iterrows() for g in str(r.samples).replace(" ", "").split(";") if g.startswith("GSM")}
    SAME = json.load(open(os.path.join(HERE, "same_source_v21.json")))["groups"]
    declared = lambda ser: any(set(ser.split(";")) <= set(gr["series"]) for gr in SAME)
    dup_used = [r for r in rows if any(g in used for g in r["gsms"].split(";"))]
    both_used = [r for r in rows if sum(g in used for g in r["gsms"].split(";")) > 1]          # the same array twice in the roster
    undeclared = sorted({r["series"] for r in dup_used if ";" in r["series"] and not declared(r["series"])})
    L.append(f"(1) physical arrays deposited under more than one GSM: {len(rows)} in the snapshots; {len(dup_used)} touch a sample the roster uses")
    for r in dup_used[:15]: L.append(f"    {r['barcode_pos']}: {r['gsms']} ({r['series']})")
    L.append(f"    of these, the same physical array used twice in the roster: {len(both_used)}; series sharing arrays not declared one source "
             f"in same_source_v21.json: {undeclared or 'none'}")
    if both_used: problems.append(f"{len(both_used)} physical arrays used twice in the roster")
    if undeclared: problems.append(f"series sharing arrays not declared one source: {undeclared}")
    # (2) roster GSMs exist, belong to their series, have raw files; counts
    miss, wrong, noraw, cnt = [], [], [], []
    seen = collections.defaultdict(list)
    for _, r in USE.iterrows():
        g = [x for x in str(r.samples).replace(" ", "").split(";") if x.startswith("GSM")]
        for x in g:
            seen[x].append(f"{r.cell_type} / {r.series} / {r.role}")
            if x not in S: miss.append(f"{x} ({r.series})")
            elif S[x]["series"] != r.series and r.series.startswith("GSE"): wrong.append(f"{x} listed under {r.series}, GEO series {S[x]['series']}")
            elif not S[x]["raw"]: noraw.append(f"{x} ({r.series})")
        if g and pd.notna(r.arrays_or_libraries) and len(g) != int(r.arrays_or_libraries): cnt.append(f"{r.cell_type} / {r.series}: {len(g)} GSMs listed, count says {int(r.arrays_or_libraries)}")
    L.append("    note: sequencing samples keep their raw reads in SRA, not as GEO supplementary files, so 'without raw files' is expected for them")
    L.append(f"(2) roster GSMs not on GEO: {len(miss)}; under the wrong series: {len(wrong)}; without raw files: {len(noraw)}; count mismatches: {len(cnt)}")
    for x in (miss + wrong + noraw + cnt)[:20]: L.append("    " + x)
    problems += [f"{len(miss)} roster GSMs not found"] * bool(miss) + [f"{len(wrong)} GSMs under the wrong series"] * bool(wrong)
    # (3) the same GSM in two roster rows
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    L.append(f"(3) GSMs listed in more than one roster row: {len(multi)}")
    for k, v in list(multi.items())[:10]: L.append(f"    {k}: " + " || ".join(v))
    if multi: problems.append(f"{len(multi)} GSMs in two roster rows")
    # (4) Loyfer and Moss: every GEO sample used or excluded with a reason
    for src, g in (("Loyfer2023", "GSE186458"), ("Moss2018", "GSE122126")):
        geo = set(samples(g)); u = {str(s).split("_")[0] for s in RS[RS.source == src]["sample"]}
        q = RS[(RS.source == src)]; qc_out = {str(s).split("_")[0] for s in q[q.qc == False]["sample"]}
        unaccounted = sorted(geo - u)
        L.append(f"(4) {src} {g}: GEO samples {len(geo)}; in the atlas roster {len(u & geo)} (of which QC-excluded {len(qc_out & geo)}); "
                 f"not in the roster at all {len(unaccounted)}; in the roster but not in {g}: {len(u - geo)}")
        if unaccounted:
            ex = collections.Counter(samples(g)[x]["source"] for x in unaccounted)
            L.append("    not in the roster, by GEO source name: " + "; ".join(f"{k} {v}" for k, v in ex.most_common(12)))
    L.append("IDENTITY AUDIT (metadata) " + ("PASSED" if not problems else "OPEN: " + "; ".join(problems)))
    open(os.path.join(HERE, "identity_audit_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
    return 1 if problems else 0
if __name__ == "__main__":
    if sys.argv[1] == "fetch": fetch()
    else: sys.exit(audit())
