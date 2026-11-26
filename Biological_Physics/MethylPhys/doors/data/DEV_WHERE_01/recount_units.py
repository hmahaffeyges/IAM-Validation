"""Recount healthy libraries on EVERY Law Sphere unit (592,067 units over the 28,217,448 hg19 CpGs), for the Locator's healthy reference over every
place (owed OW-LOCATOR-ALL-UNITS, edit 273). Counting only, with the chain's own counter (stage_law_sphere.count_pat_units: IAM-A's molecule rule
per unit) and its own saved format (stage_law_sphere.save_counts), so the reference builder reads these files exactly as the chain's own.

Usage: recount_units.py LIST.csv START END OUTDIR [PAT_DIR]
  LIST.csv  columns gsm and cell (the atlas library list); rows START..END-1 are counted
  PAT_DIR   a folder holding <gsm>*.pat.gz; without it each file is fetched from GEO (its hg19 .pat.gz, never the hg38 one)
Writes OUTDIR/rc_<gsm>_counts.npz and OUTDIR/rc_<gsm>.json (lines, molecules, seconds), and skips a library whose counts file exists."""
import os, sys, csv, json, glob, gzip, time, re, subprocess, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
def chain_dir():
    for c in (os.environ.get("IAM_CHAIN"), os.path.join(HERE, "..", "..", "..", "chain"), os.path.join(os.getcwd(), "Biological_Physics", "MethylPhys", "chain")):
        if c and os.path.exists(os.path.join(c, "stage_law_sphere.py")): return os.path.abspath(c)
    raise SystemExit("chain folder not found (set IAM_CHAIN)")

def geo_pat(gsm, dest):
    u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:7]}nnn/{gsm}/suppl/"
    for t in range(6):
        try:
            names = [n for n in re.findall(r'href="([^"]+\.pat\.gz)"', urllib.request.urlopen(u, timeout=120).read().decode()) if ".hg38." not in n]
            if names: break
        except Exception: pass
        time.sleep(30 * (t + 1))
    else: raise RuntimeError(f"no hg19 .pat.gz for {gsm} on GEO")
    f = os.path.join(dest, f"{gsm}.pat.gz")
    for t in range(6):
        if subprocess.run(["curl", "-sSL", "--retry", "3", "-C", "-", "-o", f, u + names[0]]).returncode == 0 and subprocess.run(["gzip", "-t", f]).returncode == 0: return f
        if os.path.exists(f): os.remove(f)
        time.sleep(60 * (t + 1))
    raise RuntimeError(f"download of {gsm} incomplete")

def main():
    lst, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; pdir = sys.argv[5] if len(sys.argv) > 5 else None
    CH = chain_dir(); sys.path.insert(0, CH); import numpy as np, stage_law_sphere as LS
    Z = np.load(os.path.join(LS.REF_DIR, "units.npz")); umap = LS.unit_map(Z["isl_first"], Z["isl_last"], int(Z["block"][0]), int(Z["n_cpg"][0]))
    ref = dict(table=__import__("pandas").DataFrame(dict(unit=np.arange(int(umap.max()) + 1))))
    os.makedirs(out, exist_ok=True); rows = list(csv.DictReader(open(lst)))[a:b]; tmp = os.path.join(out, "_pat"); os.makedirs(tmp, exist_ok=True)
    for r in rows:
        g = r["gsm"]; cf = os.path.join(out, f"rc_{g}_counts.npz")
        if os.path.exists(cf): print(g, "already counted", flush=True); continue
        t0 = time.time()
        try:
            f = (glob.glob(os.path.join(pdir, f"{g}*.pat.gz")) or [None])[0] if pdir else geo_pat(g, tmp)
            if not f: raise RuntimeError("no molecule file in " + str(pdir))
            n = 0
            def lines():
                nonlocal n
                with gzip.open(f, "rt") as fh:
                    for l in fh: n += 1; yield l
            C = LS.count_pat_units(lines(), umap)
            assert len(C["err"]) == int(umap.max()) + 1, "every unit counted"
            LS.save_counts(cf, C, ref, {r["cell"]: 1.0}, None, sample=g)
            info = dict(gsm=g, cell=r["cell"], lines=n, units=int(len(C["err"])), held_opp=float(C["opp"].sum()), gain_opp=float(C["gopp"].sum()), seconds=round(time.time() - t0, 1))
        except Exception as e:
            info = dict(gsm=g, cell=r.get("cell"), error=f"{type(e).__name__}: {str(e)[:300]}", seconds=round(time.time() - t0, 1))
        json.dump(info, open(os.path.join(out, f"rc_{g}.json"), "w")); print(json.dumps(info), flush=True)
        if not pdir:
            for x in glob.glob(os.path.join(tmp, "*")): os.remove(x)

if __name__ == "__main__":
    main()
