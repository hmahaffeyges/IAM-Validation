"""cpg_kit.py — shared helpers for the Issue 003 procedure scripts (PROC-*.py).

Everything a PROC needs to load, in one place, with no hidden state. Paths are resolved
relative to the kit root (this file's directory) unless overridden by environment variables:

  CPG_KIT_ENGINE   dir holding the engine files (default kit/engine)
  CPG_KIT_RUNTIME  dir holding the runtime JSONs  (default kit/runtime)
  CPG_KIT_DATA     dir holding large inputs        (default kit/data) — atlas CSV, GEO matrices,
                   betas_cache.pkl, IDATs. These are NOT shipped in the kit; RUNBOOK.md says
                   where each comes from and gives its sha256.

Formulas (RULING A3, Issue 003 §1.5):
  gauge_A(beta_vector, class)      = H(mean beta over the class IDENTITY loci) / H_min(class)
  separation_A(beta_vector, cell)  = mean over the cell's DISCRIMINATIVE markers of H(beta_i) / H_min(class of cell)
"""
import os, json, hashlib, gzip
import numpy as np

ROOT    = os.path.dirname(os.path.abspath(__file__))
ENGINE  = os.environ.get("CPG_KIT_ENGINE",  os.path.join(ROOT, "engine"))
RUNTIME = os.environ.get("CPG_KIT_RUNTIME", os.path.join(ROOT, "runtime"))
DATA    = os.environ.get("CPG_KIT_DATA",    os.path.join(ROOT, "data"))
CLASSES = ["terminal","secretory","immune","progenitor","cycling","stromal","stem_adult","stem_pluri"]

def sha256(path, block=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""): h.update(chunk)
    return h.hexdigest()

def H(b):
    """Binary Shannon entropy in bits, elementwise, clipped away from 0/1."""
    b = np.clip(np.asarray(b, float), 1e-12, 1 - 1e-12)
    return -b*np.log2(b) - (1-b)*np.log2(1-b)

def load_identity():
    """iamatlas_gauge_identity_loci_v1_0.json -> {class: {'loci': set, 'H_min': float, 'H_min_beta': float, 'band': float}}"""
    j = json.load(open(os.path.join(RUNTIME, "iamatlas_gauge_identity_loci_v1_0.json")))
    return {c: {"loci": set(v["loci"]), "H_min": float(v["H_min"]), "H_min_beta": float(v.get("H_min_beta", float("nan"))), "band": float(v.get("band", float("nan")))}
            for c, v in j.items() if isinstance(v, dict) and "loci" in v}

def load_markers(which="chrX_removed"):
    """iamatlas_celltype_markers_v0_2 — 'chrX_removed' (canonical per RULING M1b) or 'repo_head' (pre-fix, what the v1 seal used)."""
    fn = {"chrX_removed": "iamatlas_celltype_markers_v0_2.json", "repo_head": "iamatlas_celltype_markers_v0_2_REPO_HEAD_prechrX.json"}[which]
    j = json.load(open(os.path.join(RUNTIME, fn)))
    return j["markers_by_celltype"], j["celltype_to_class"]

def hmin_table():
    """The 40-cell H_MIN_TABLE parsed out of cpg_gauge_engine.py (class -> [methyl, nucl, fuzz, wps, frag])."""
    import re, ast
    src = open(os.path.join(ENGINE, "cpg_gauge_engine.py"), encoding="utf-8").read()
    m = re.search(r"^H_MIN_TABLE\s*=\s*(\{.*?^\})", src, re.M | re.S)
    return ast.literal_eval(m.group(1))

def gauge_A(beta, cls, ident=None, presence=None, detect_floor=0.01, jensen_max=0.05):
    """Class gauge, RULING A3: A = H(mean beta over identity loci) / H_min(class). Returns (A, info).
    Refuses (A=None) when:
      - the class is absent from the substrate (presence < detect_floor)                       -> ABSENT
      - the panel received is one where averaging beta first is not valid: the Jensen gap
        H(beta_mean) - mean_i H(beta_i), measured on THIS panel, exceeds jensen_max.            -> INVALID_PANEL
    The second test is the quantity SOP v1.4.0 s105 warns about, measured directly rather than inferred
    from shape. Calibration on the 11 test samples (2026-09-19): identity loci in whole blood 0.029;
    identity loci in bulk tissue 0.17-0.24 (mixture - refused by design, cf. glioma-LL-002);
    HSC marker panel 0.29 (s105's own false-BREACH example); Hepatocytes markers 0.78."""
    ident = ident or load_identity()
    loci = ident[cls]["loci"]; hm = ident[cls]["H_min"]
    get = beta.get if isinstance(beta, dict) else (lambda k, d=None: beta[k] if k in beta.index else d)
    v = np.array([x for x in (get(c) for c in loci) if x is not None and x == x], float)
    info = {"class": cls, "n_loci": int(len(v)), "coverage": len(v)/max(len(loci), 1), "H_min": hm}
    if presence is not None and presence < detect_floor:
        info["refused"] = f"ABSENT: presence {presence:.4f} < {detect_floor}"; return None, info
    if len(v) == 0: info["refused"] = "NO_LOCI"; return None, info
    bm = float(v.mean()); A_hm = float(H(bm)/hm); A_mh = float(H(v).mean()/hm); gap = A_hm - A_mh
    info.update(beta_mean=bm, A_H_of_mean=A_hm, A_mean_of_H=A_mh, jensen_gap=gap, frac_lo=float((v < 0.2).mean()), frac_hi=float((v > 0.8).mean()))
    if gap > jensen_max:
        info["refused"] = f"INVALID_PANEL: Jensen gap {gap:.3f} > {jensen_max} - averaging beta first is not valid on this panel (mixture or marker-type)"; return None, info
    info.update(A=A_hm, ceiling=1/hm, saturated=A_hm >= 1/hm - 0.005)
    return A_hm, info

def separation_A(beta, cell, markers, c2c, ident=None):
    """Per-cell separation statistic, RULING A3 / SOP v1.4.0 §105: mean_i H(beta_i) / H_min(class of cell)."""
    ident = ident or load_identity()
    get = beta.get if isinstance(beta, dict) else (lambda k, d=None: beta[k] if k in beta.index else d)
    v = np.array([x for x in (get(c) for c in markers.get(cell, [])) if x is not None and x == x], float)
    if len(v) == 0: return None, {"cell": cell, "n": 0}
    hm = ident[c2c[cell]]["H_min"]
    return float(H(v).mean()/hm), {"cell": cell, "n": int(len(v)), "H_min": hm, "beta_mean": float(v.mean())}

def stream_geo_matrix(path, keep_cpgs=None, keep_gsms=None):
    """Stream a GEO series_matrix.txt.gz once. Returns (gsms, {cpg: np.array over gsms}). keep_* restrict rows/cols."""
    f = gzip.open(path, "rt", errors="replace")
    for l in f:
        if l.startswith("!series_matrix_table_begin"): break
    hdr = next(f); cols = [v.strip('"') for v in hdr.rstrip("\n").split("\t")[1:]]
    idx = [i for i, c in enumerate(cols) if (keep_gsms is None or c in keep_gsms)]
    out = {}
    for l in f:
        if l.startswith("!series_matrix_table_end"): break
        tab = l.find("\t"); cg = l[:tab].strip('"')
        if keep_cpgs is not None and cg not in keep_cpgs: continue
        p = l.rstrip("\n").split("\t")
        out[cg] = np.array([np.nan if p[i+1].strip('"') in ("", "NA", "null") else float(p[i+1].strip('"')) for i in idx])
    f.close()
    return [cols[i] for i in idx], out

def report(name, rows, verdict, path=None):
    """Print a PROC block (input/operation/expected/observed/verdict) and optionally write it as JSON."""
    print(f"\n=== {name} ===")
    for k, v in rows: print(f"  {k:<12} {v}")
    print(f"  {'verdict':<12} {verdict}")
    if path:
        json.dump({"proc": name, "rows": rows, "verdict": verdict}, open(path, "w"), indent=1)
