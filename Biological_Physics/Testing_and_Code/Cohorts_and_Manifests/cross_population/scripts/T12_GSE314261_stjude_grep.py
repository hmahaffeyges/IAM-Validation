#!/usr/bin/env python3
"""
T12 — GSE314261 (St Jude Lifetime Cohort) n=5013 streaming analysis
====================================================================
Strategy: curl | zcat | grep -Ff panel.txt → small filtered file → parse.
This pushes the hot loop into grep's C implementation, which runs at ~75 MB/s
— essentially matching zcat's decompression throughput.

Expected runtime:
  - Download 28 GB compressed (at 60-90 MB/s): 320-470s
  - grep filter to 538 CpG rows (runs concurrently): ~same
  - Python parse of 538 rows: <1s

IMPORTANT CAVEAT — DIFFERENT BIOLOGICAL QUESTION
-------------------------------------------------
This is NOT a breast cancer pre-diagnostic test. These are childhood cancer
survivors (predominantly ALL, Hodgkin, CNS tumors, NHL) with potential
treatment-related late effects, compared to community controls. Running the
Xu-538 adult BC pre-dx panel on this cohort tests SPECIFICITY:
  (a) positive result  -> panel not BC-specific; captures general treatment /
                          immune-stress biology
  (b) null result      -> panel is BC-specific

Either outcome is informative, but NOT a replication of the EPIC-Italy
BC pre-diagnostic d=+1.85 signal.
"""

import argparse, csv, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np

H_MIN_IMMUNE = 0.838889
RANDOM_SEED  = 20260420
np.random.seed(RANDOM_SEED)

# ============================================================================
# CORE
# ============================================================================
def H_binary(beta):
    if beta is None or not (0.0 < beta < 1.0) or math.isnan(beta):
        return 0.0
    return -beta * math.log2(beta) - (1.0 - beta) * math.log2(1.0 - beta)

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    s1 = float(np.std(a, ddof=1)); s2 = float(np.std(b, ddof=1))
    pooled = math.sqrt(((len(a)-1)*s1*s1 + (len(b)-1)*s2*s2) / (len(a)+len(b)-2))
    return 0.0 if pooled == 0 else float((np.mean(a) - np.mean(b)) / pooled)

def permutation_p(cases, controls, n_perm=5000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(cases, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    d_obs = cohens_d(a, b)
    if math.isnan(d_obs): return float("nan"), d_obs
    combined = np.concatenate([a, b]); n_a = len(a); count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        if abs(cohens_d(combined[:n_a], combined[n_a:])) >= abs(d_obs): count_ge += 1
    return float((count_ge+1)/(n_perm+1)), float(d_obs)

def bootstrap_ci(cases, controls, n_boot=1000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed+1)
    a = np.asarray(cases, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    ds = []
    for _ in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        dd = cohens_d(ra, rb)
        if not math.isnan(dd): ds.append(dd)
    if not ds: return float("nan"), float("nan")
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

def load_panel(path):
    with open(path) as f: data = json.load(f)
    if isinstance(data, list): return set(data)
    if isinstance(data, dict):
        for key in ("cpgs","panel","cpg_ids","probes"):
            if key in data and isinstance(data[key], list): return set(data[key])
        cpgs = set()
        for v in data.values():
            if isinstance(v, list):
                cpgs.update(x for x in v if isinstance(x, str) and x.startswith("cg"))
        if cpgs: return cpgs
    sys.exit("[FATAL] Cannot parse panel")

# ============================================================================
# STREAMING FILTER — curl | zcat | grep -Ff
# ============================================================================
def stream_filter_rows(url, panel_cpgs, out_path, header_path,
                       checkpoint_path=None):
    """
    Pipeline: curl URL | zcat | awk(split header) | grep -Ff panel_patterns > out_path

    awk writes line 1 to header_path and passes all remaining lines through to grep.
    This avoids bash process substitution (which requires bash, not /bin/sh).
    """
    patterns_file = out_path + ".patterns"
    with open(patterns_file, "w") as f:
        for c in panel_cpgs:
            f.write(f"{c},\n")

    # awk dispatches header to a file; remaining lines stream through to grep
    cmd = (
        f"curl -sS -L --max-time 1800 '{url}' | "
        f"zcat 2>/dev/null | "
        f"awk 'NR==1 {{print > \"{header_path}\"; next}} {{print}}' | "
        f"grep -Ff '{patterns_file}' > '{out_path}'"
    )
    print(f"  Running pipeline:", flush=True)
    print(f"    curl | zcat | awk(header→{os.path.basename(header_path)}) | grep -Ff panel", flush=True)
    t0 = time.time()
    # Use bash explicitly (awk inline uses special chars); capture stderr
    ret = subprocess.run(['bash','-c', cmd], capture_output=True, text=True, timeout=1700)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        # returncode 1 from grep just means no matches; both 0 and 1 are OK
        # Anything else is a real error (pipeline failure, curl failure, etc.)
        if ret.returncode != 1:
            print(f"  [WARN] pipeline returncode={ret.returncode}", flush=True)
            if ret.stderr: print(f"  stderr: {ret.stderr[:500]}", flush=True)
    print(f"  Pipeline complete in {elapsed:.1f}s", flush=True)
    try: os.remove(patterns_file)
    except OSError: pass
    hsz = os.path.getsize(header_path) if os.path.exists(header_path) else 0
    osz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    print(f"  Header file:  {hsz} bytes", flush=True)
    print(f"  Matched rows: {osz} bytes", flush=True)
    return elapsed

def parse_matched_rows(header_path, out_path, panel_cpgs):
    """Parse the header and matched rows into per-sample A-scores."""
    with open(header_path) as f:
        header_line = f.readline().rstrip('\n')
    header = header_line.split(',')
    sample_beadchips = header[1:]
    n_samples = len(sample_beadchips)
    print(f"  Header columns: {len(header)} (expected 5014)", flush=True)

    sum_A = np.zeros(n_samples, dtype=np.float64)
    count = np.zeros(n_samples, dtype=np.int64)
    cpgs_matched = set()
    n_rows = 0

    with open(out_path, 'rb') as f:
        for line_b in f:
            n_rows += 1
            line = line_b.decode('utf-8', errors='replace').rstrip('\n')
            fc = line.find(',')
            if fc < 0: continue
            cpg_id = line[:fc]
            if cpg_id.startswith('"'): cpg_id = cpg_id.strip('"')
            if cpg_id not in panel_cpgs:
                continue
            cpgs_matched.add(cpg_id)
            fields = line[fc+1:].split(',')
            for i in range(min(n_samples, len(fields))):
                v = fields[i]
                if v == '' or v == 'NA' or v == 'NaN': continue
                try: b = float(v)
                except ValueError: continue
                if math.isnan(b): continue
                sum_A[i] += H_binary(b) / H_MIN_IMMUNE
                count[i] += 1
    print(f"  Parsed {n_rows} matched rows", flush=True)
    A = np.where(count > 0, sum_A / count, np.nan)
    return sample_beadchips, A, count, cpgs_matched

# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--panel",        required=True)
    ap.add_argument("--output_dir",   required=True)
    ap.add_argument("--data_url",     default=
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE314nnn/GSE314261/suppl/"
        "GSE314261_SJLIFE_IlluminaEPIC_proccessed.txt.gz")
    ap.add_argument("--skip_stream", action="store_true",
                    help="Skip download; reuse existing filtered files")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("T12 GSE314261 — St Jude Lifetime Cohort n=5013 (curl|zcat|grep pipeline)", flush=True)
    print("=" * 78, flush=True)
    pan_sha  = sha256_of_file(args.panel)
    meta_sha = sha256_of_file(args.metadata_csv)
    print(f"  panel sha256:        {pan_sha}", flush=True)
    print(f"  metadata sha256:     {meta_sha}", flush=True)
    print(f"  data URL:            {args.data_url}", flush=True)
    print(f"  H_min(immune):       {H_MIN_IMMUNE}", flush=True)
    print(f"  Random seed:         {RANDOM_SEED}", flush=True)
    print()

    panel = load_panel(args.panel)
    print(f"Panel CpGs: {len(panel)}", flush=True)
    print()

    # Load metadata
    meta = {}
    with open(args.metadata_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row['beadchip_id']:
                meta[row['beadchip_id']] = row
    print(f"Metadata rows with beadchip_id: {len(meta)}", flush=True)
    from collections import Counter
    groups = Counter(r.get('group','') for r in meta.values())
    sources = Counter(r.get('source','') for r in meta.values())
    print(f"  groups:  {dict(groups)}", flush=True)
    print(f"  sources: {dict(sources)}", flush=True)
    print()

    header_path   = str(out_dir / "GSE314261_header.csv")
    matched_path  = str(out_dir / "GSE314261_panel_rows.csv")
    checkpoint_path = str(out_dir / "STREAM_PROGRESS.json")

    if not args.skip_stream:
        print(f"Streaming 28 GB compressed / ~250 GB decompressed via curl|zcat|grep...", flush=True)
        stream_elapsed = stream_filter_rows(
            args.data_url, panel, matched_path, header_path,
            checkpoint_path=checkpoint_path)
    else:
        print(f"--skip_stream: using existing {matched_path}", flush=True)
        stream_elapsed = 0.0

    # Parse the filtered rows
    print()
    print(f"Parsing filtered panel-matched rows...", flush=True)
    sample_beadchips, A, count, matched = parse_matched_rows(
        header_path, matched_path, panel)
    print(f"  panel CpGs matched: {len(matched)} / {len(panel)}", flush=True)
    print()

    # Merge with metadata
    rows = []
    for i, bid in enumerate(sample_beadchips):
        m = meta.get(bid, {})
        rows.append({
            'beadchip_id': bid,
            'gsm':         m.get('gsm', ''),
            'group':       m.get('group', ''),
            'source':      m.get('source', ''),
            'tissue':      m.get('tissue', ''),
            'Sex':         m.get('Sex', ''),
            'age':         m.get('age', ''),
            'diagnostic_group':  m.get('diagnostic group', ''),
            'race_group':  m.get('race group', ''),
            'age_at_dx':   m.get('age at cancer diagnosis', ''),
            'A_score':     A[i],
            'n_cpgs_used': int(count[i]),
        })
    csv_path = out_dir / "GSE314261_per_sample_A.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV: {csv_path}  sha256: {csv_sha}", flush=True)
    print(f"  rows: {len(rows)}", flush=True)
    print()

    valid_rows = [r for r in rows if not math.isnan(r['A_score'])]
    print(f"Valid per-sample A-scores: {len(valid_rows)} / {len(rows)}", flush=True)
    A_all = np.array([r['A_score'] for r in valid_rows])
    cnt   = np.array([r['n_cpgs_used'] for r in valid_rows])
    if len(A_all) > 0:
        print(f"  A summary: mean={A_all.mean():.4f}  sd={A_all.std():.4f}", flush=True)
        print(f"  n_cpgs_used: min={cnt.min()}  max={cnt.max()}  median={int(np.median(cnt))}", flush=True)
    print()

    # Primary specificity test: blood-only survivors vs controls
    blood = [r for r in rows if r['source'] == 'blood' and not math.isnan(r['A_score'])]
    survivors = [r['A_score'] for r in blood if r['group'] == 'cancer survivor']
    controls  = [r['A_score'] for r in blood if r['group'] == 'control']
    print("=" * 78, flush=True)
    print("PRIMARY TEST — Xu-538 specificity on childhood cancer survivors vs controls", flush=True)
    print("=" * 78, flush=True)
    print(f"  n survivors: {len(survivors)}", flush=True)
    print(f"  n controls:  {len(controls)}", flush=True)

    primary = {}
    if len(survivors) >= 2 and len(controls) >= 2:
        s_arr = np.asarray(survivors); c_arr = np.asarray(controls)
        d_obs = cohens_d(s_arr, c_arr)
        p_perm, _ = permutation_p(s_arr, c_arr, n_perm=5000, seed=RANDOM_SEED)
        ci_lo, ci_hi = bootstrap_ci(s_arr, c_arr, n_boot=1000, seed=RANDOM_SEED)
        delta = float(s_arr.mean() - c_arr.mean())
        print(f"  survivors A mean: {s_arr.mean():.4f}  sd {s_arr.std(ddof=1):.4f}", flush=True)
        print(f"  controls  A mean: {c_arr.mean():.4f}  sd {c_arr.std(ddof=1):.4f}", flush=True)
        print(f"  Δ: {delta:+.4f}   Cohen's d: {d_obs:+.3f}", flush=True)
        print(f"  p (5k perm): {p_perm:.4f}   95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]", flush=True)
        primary = {
            'n_survivors':      int(len(survivors)),
            'n_controls':       int(len(controls)),
            'survivors_A_mean': float(s_arr.mean()),
            'survivors_A_sd':   float(s_arr.std(ddof=1)),
            'controls_A_mean':  float(c_arr.mean()),
            'controls_A_sd':    float(c_arr.std(ddof=1)),
            'delta_A':          float(delta),
            'cohens_d':         float(d_obs),
            'p_perm':           float(p_perm),
            'CI95':             [float(ci_lo), float(ci_hi)],
        }
    print()

    # Diagnostic-group stratification
    print("=" * 78, flush=True)
    print("SECONDARY — stratified by diagnostic group (survivors only vs all controls)", flush=True)
    print("=" * 78, flush=True)
    dx_groups = sorted(set(r['diagnostic_group'] for r in blood
                           if r['group']=='cancer survivor' and r['diagnostic_group']))
    c_arr = np.asarray(controls)
    stratified = []
    for dx in dx_groups:
        dx_scores = [r['A_score'] for r in blood
                     if r['group']=='cancer survivor' and r['diagnostic_group']==dx]
        if len(dx_scores) < 10: continue
        dx_arr = np.asarray(dx_scores)
        d_strat = cohens_d(dx_arr, c_arr)
        print(f"  {dx[:50]:<50}  n={len(dx_scores):<4}  "
              f"mean={dx_arr.mean():.4f}  d_vs_controls={d_strat:+.3f}", flush=True)
        stratified.append({
            'diagnostic_group':         dx,
            'n':                        int(len(dx_scores)),
            'A_mean':                   float(dx_arr.mean()),
            'A_sd':                     float(dx_arr.std(ddof=1)),
            'cohens_d_vs_controls':     float(d_strat),
        })
    print()

    out = {
        'cohort':          'GSE314261_StJude_Lifetime_Cohort_2026',
        'test_type':       'specificity (NOT BC pre-dx replication)',
        'design':          ('4669 childhood cancer survivors + 342 community controls '
                            '(5005 blood + 8 saliva), Illumina EPIC. Specificity test '
                            'of Xu-538 adult BC panel against treatment-related '
                            'late-effect biology.'),
        'caveats':         ('NOT a breast cancer pre-diagnostic replication. Childhood '
                            'cancer survivors have undergone chemotherapy/radiation which '
                            'can leave a treatment-related methylation signature. A positive '
                            'result here indicates the panel is non-specific; a null result '
                            'indicates BC-specificity. Either outcome does not replicate '
                            'the EPIC-Italy d=+1.85 pre-diagnostic finding.'),
        'data_url':        args.data_url,
        'streaming_pipeline': 'curl | zcat | grep -Ff panel_patterns',
        'n_samples_total': int(len(rows)),
        'n_samples_with_A':int(len(valid_rows)),
        'panel':           'xu538_breast',
        'panel_n_cpgs':    len(panel),
        'panel_matched':   int(len(matched)),
        'H_min_immune':    H_MIN_IMMUNE,
        'random_seed':     RANDOM_SEED,
        'streaming_elapsed_s': float(stream_elapsed),
        'input_sha256':    {'panel': pan_sha, 'metadata_csv': meta_sha},
        'output_sha256':   {'per_sample_csv': csv_sha},
        'primary_specificity_test': primary,
        'by_diagnostic_group': stratified,
    }
    json_path = out_dir / "GSE314261_analysis.json"
    with open(json_path, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"Output JSON: {json_path}", flush=True)
    print(f"  sha256: {sha256_of_file(json_path)}", flush=True)
    print()
    print("T12 complete.", flush=True)

if __name__ == "__main__":
    main()
