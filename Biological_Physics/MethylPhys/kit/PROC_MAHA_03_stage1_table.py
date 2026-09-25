#!/usr/bin/env python3
"""PROC-MAHA-03: the Sentrix-chip term on the commissioned scale, and the k-sweep of the two panel protocols.

Deviation from the pre-registration, recorded: the run uses the Stage 1 noob betas calibrated from IDATs
(results/percell/stage1_betas_*.pkl.xz) rather than the GEO series matrices. The series matrices for the four
cohorts are header-only (their betas ship as separate supplementary files), and the IDAT filenames carry the
Sentrix barcode - so GSE42861 Karolinska, which the pre-registration excluded for want of a barcode, is IN.
"""
import json, glob, lzma, pickle, re, os, math, random, collections
import numpy as np

CH = "Biological_Physics/MethylPhys/chain"
ident = json.load(open(glob.glob(CH + "/**/iamatlas_gauge_identity_loci_v1_0.json", recursive=True)[0]))["immune"]
LOCI = set(ident["loci"]); HMIN = float(ident["H_min"])
band = json.load(open(glob.glob(CH + "/**/identity_band_v3.json", recursive=True)[0]))
POOLED = band["pooled"]; SIGMA_BAND = (POOLED["p90"] - POOLED["p10"]) / (2 * 1.2816)
coh = band["_meta"]["cohorts"]
if isinstance(coh, str):
    import ast; coh = ast.literal_eval(coh)
ZLAB = {k.split("_")[0]: v["z_lab_full_cohort"] for k, v in coh.items()}
TAIL_PUB = {k.split("_")[0]: v["tail_p95"] for k, v in coh.items()}
CURVE = {int(k): v for k, v in json.load(open(glob.glob(CH + "/**/reference_age_curve_v1.json", recursive=True)[0]))["curve"].items()}
THR95 = 1.959964
random.seed(5011); np.random.seed(5011)


def H(b):
    b = min(max(b, 1e-12), 1 - 1e-12)
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


# GSM -> chip from the IDAT filenames
chip_of = {}
for p in glob.glob("**/*_Grn.idat*", recursive=True):
    m = re.search(r"(GSM\d+)_(\d{9,12})_(R0\dC0\d)", os.path.basename(p))
    if m: chip_of[m.group(1)] = m.group(2)

# GSM -> age from each cohort's series-matrix header
age_of = {}
for gse in ZLAB:
    p = f"geo/{gse}_series_matrix.txt.gz"
    if not os.path.exists(p): continue
    import gzip
    gsms = None; ages = {}
    with gzip.open(p, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"): gsms = re.findall(r'"([^"]+)"', line)
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = re.findall(r'"([^"]*)"', line)
                if vals and sum(1 for v in vals[:5] if re.search(r"\bage\b", v, re.I)) >= 2:
                    for i, v in enumerate(vals):
                        mm = re.search(r"(\d{1,3})(?:\.\d+)?\s*$", v.strip())
                        if mm: ages[i] = int(mm.group(1))
            elif line.startswith("!series_matrix_table_begin"): break
    if gsms:
        for i, g in enumerate(gsms):
            if i in ages: age_of[g] = ages[i]


def c_of(age):
    if age is None: return None
    d = max(min(int(age) // 10 * 10, max(CURVE)), min(CURVE))
    return CURVE[d]


rows = []
for gse in ("GSE87571", "GSE42861", "GSE111629", "GSE125105"):
    p = f"results/percell/stage1_betas_{gse}.pkl.xz"
    if not os.path.exists(p): print(f"{gse}: no stage 1 betas", flush=True); continue
    with lzma.open(p, "rb") as f: df = pickle.load(f)
    keep = [i for i in df.index if i in LOCI]
    sub = df.loc[keep]
    bb = sub.mean(axis=0, skipna=True)
    n_ok = 0
    for gsm, bbar in bb.items():
        if not np.isfinite(bbar): continue
        c = c_of(age_of.get(gsm)); z = ZLAB[gse]; ch = chip_of.get(gsm)
        if c is None or ch is None: continue
        A = H(float(bbar)) / HMIN
        rows.append({"gsm": gsm, "cohort": gse, "chip": ch, "age": age_of.get(gsm),
                     "A_mapped": A, "A_abs": A - c - z}); n_ok += 1
    ch_c = collections.Counter(r["chip"] for r in rows if r["cohort"] == gse)
    print(f"{gse}: {sub.shape[0]} identity loci x {df.shape[1]} arrays -> {n_ok} with age and chip | "
          f"chips {len(ch_c)} (per chip min {min(ch_c.values()) if ch_c else 0} "
          f"median {sorted(ch_c.values())[len(ch_c)//2] if ch_c else 0} max {max(ch_c.values()) if ch_c else 0})", flush=True)

json.dump(rows, open("handoff/maha03_rows.json", "w"))
print(f"\ntotal arrays usable: {len(rows)}")


def icc_and_null(vals_by_chip, n_perm=1000):
    """between-chip variance component and its permutation p, one cohort."""
    groups = [v for v in vals_by_chip.values() if len(v) >= 2]
    if len(groups) < 3: return None
    allv = np.concatenate(groups); grand = allv.mean()
    k = len(groups); n = len(allv)
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ssw = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in groups)
    msb = ssb / (k - 1); msw = ssw / (n - k) if n > k else float("nan")
    n0 = n / k
    var_b = max((msb - msw) / n0, 0.0)
    icc = var_b / (var_b + msw) if (var_b + msw) > 0 else 0.0
    F = msb / msw if msw > 0 else float("inf")
    sizes = [len(g) for g in groups]
    worse = 0
    for _ in range(n_perm):
        sh = np.random.permutation(allv); i = 0; gs = []
        for s in sizes: gs.append(sh[i:i + s]); i += s
        gb = sum(len(g) * (g.mean() - grand) ** 2 for g in gs) / (k - 1)
        gw = sum(((g - g.mean()) ** 2).sum() for g in gs) / (n - k)
        if gw > 0 and gb / gw >= F: worse += 1
    return {"n_chips_used": k, "n_arrays": int(n), "icc": round(icc, 4),
            "sd_between": round(math.sqrt(var_b), 5), "sd_within": round(math.sqrt(max(msw, 0)), 5),
            "F": round(F, 3), "p_perm": (worse + 1) / (n_perm + 1)}


by_cohort = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows: by_cohort[r["cohort"]][r["chip"]].append(r["A_abs"])

print("\n=== B1  the chip term, per cohort (1,000 chip-label shuffles) ===")
b1 = {}
for gse, d in by_cohort.items():
    res = icc_and_null({c: np.array(v) for c, v in d.items()})
    b1[gse] = res
    if res: print(f"  {gse}: ICC {res['icc']:.4f}  sd_between {res['sd_between']}  sd_within {res['sd_within']}  "
                  f"F {res['F']}  p_perm {res['p_perm']:.4f}  ({res['n_chips_used']} chips, {res['n_arrays']} arrays)")
    else: print(f"  {gse}: too few chips with 2+ arrays")


def tails(vals, sigma):
    v = np.array(vals); z = (v - 1.0) / sigma
    return float((np.abs(z) > THR95).mean())


def sigma_of(vals):
    v = np.array(vals)
    return float((np.percentile(v, 90) - np.percentile(v, 10)) / (2 * 1.2816))


print("\n=== B2/B3  held-out chip correction, k reference arrays per chip (B5: never the array being read) ===")
print(f"{'cohort':<12}{'k':>3}{'arrays corrected':>18}{'tail p95 raw':>14}{'tail band sigma':>17}{'tail re-derived':>17}{'sigma re-derived':>18}")
sweep = {}
for gse, d in sorted(by_cohort.items()):
    raw = [x for v in d.values() for x in v]
    raw_tail_band = tails(raw, SIGMA_BAND)
    for k in (1, 2, 3, 5):
        corrected = []
        for ch, v in d.items():
            if len(v) < k + 1: continue
            arr = list(v)
            for i in range(len(arr)):
                others = arr[:i] + arr[i + 1:]
                refs = random.sample(others, k) if len(others) >= k else None
                if refs is None: continue
                corrected.append(arr[i] - (float(np.mean(refs)) - 1.0))
        if len(corrected) < 30:
            sweep[(gse, k)] = None
            print(f"{gse:<12}{k:>3}{len(corrected):>18}{'':>14}{'not assessable at this chip depth':>52}")
            continue
        s_re = sigma_of(corrected)
        t_band = tails(corrected, SIGMA_BAND); t_re = tails(corrected, s_re)
        sweep[(gse, k)] = {"n": len(corrected), "tail_band_sigma": round(t_band, 4),
                           "tail_rederived": round(t_re, 4), "sigma_rederived": round(s_re, 5)}
        print(f"{gse:<12}{k:>3}{len(corrected):>18}{raw_tail_band:>14.4f}{t_band:>17.4f}{t_re:>17.4f}{s_re:>18.5f}")

print("\n=== B4  does the correction erase a real departure? (+2 sigma injected into one array) ===")
inj = 2 * SIGMA_BAND
b4 = {}
for gse, d in sorted(by_cohort.items()):
    for k in (1, 2, 3):
        rec = []
        for ch, v in d.items():
            if len(v) < k + 1: continue
            arr = list(v)
            for i in range(len(arr)):
                others = arr[:i] + arr[i + 1:]
                if len(others) < k: continue
                refs = random.sample(others, k)
                base = arr[i] - (float(np.mean(refs)) - 1.0)
                spiked = (arr[i] + inj) - (float(np.mean(refs)) - 1.0)
                rec.append((spiked - base) / inj)
        if rec:
            b4[(gse, k)] = round(float(np.mean(rec)), 4)
            print(f"  {gse} k={k}: {float(np.mean(rec))*100:.1f}% of the injected shift survives "
                  f"({len(rec)} placements)")

json.dump({"_meta": {"procedure": "PROC-MAHA-03", "run": "2026-09-22", "sigma_band": SIGMA_BAND,
                     "threshold_p95_1axis": THR95, "published_tail_p95": TAIL_PUB,
                     "deviation": "Stage 1 noob betas from IDATs, not the series matrices (header-only); "
                                  "GSE42861 therefore included, reversing the pre-registered exclusion"},
           "B1": b1,
           "B2_B3": {f"{g}|k={k}": v for (g, k), v in sweep.items()},
           "B4_recovery": {f"{g}|k={k}": v for (g, k), v in b4.items()}},
          open("handoff/maha03_results.json", "w"), indent=1)
print("\nwrote handoff/maha03_results.json")
