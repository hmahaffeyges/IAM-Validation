#!/usr/bin/env python3
"""
T12 — GSE314261 (St Jude Lifetime Cohort) n=5013 streaming analysis
====================================================================

Cohort
------
St. Jude Lifetime Cohort: 4669 childhood cancer survivors + 342 community
controls (5005 blood + 8 saliva). Illumina MethylationEPIC BeadChip.
Published 2026 (PMID 41577698).

IMPORTANT CAVEAT — DIFFERENT BIOLOGICAL QUESTION
-------------------------------------------------
This is NOT a breast cancer pre-diagnostic test. These are childhood cancer
survivors (predominantly ALL, Hodgkin, CNS tumors, NHL) with potential
treatment-related late effects (cardiotoxicity, epigenetic age acceleration),
compared to community controls. Running the Xu-538 adult BC pre-dx panel
on this cohort tests SPECIFICITY:
  (a) positive result  -> panel not BC-specific; captures general treatment /
                          immune-stress biology
  (b) null result      -> panel is BC-specific; does not bleed over into
                          treatment-related aging biology

Either outcome is informative, but NOT a replication of the EPIC-Italy
BC pre-diagnostic d=+1.85 signal.

Streaming strategy
------------------
Compressed file is 28.3 GB; decompressed is ~250 GB. We stream from NCBI
via curl | zcat and filter to panel CpGs only, never landing the full
file on disk. Per-sample accumulator: sum(H_binary(β)/H_min) and count
of non-NaN panel CpGs per sample.

Sample mapping
--------------
Data table header columns are Illumina beadchip IDs (e.g. "201114440018_R01C01").
We join them to GSMs and group labels via GSE314261_sample_metadata.csv,
which was built from the series matrix supplementary file URLs.

Canonical formula
-----------------
Per-sample A-score = mean over Xu-538 CpGs of [H_binary(β) / H_min(immune)]
H_min(immune) = 0.838889; canonical (β=0/1 retained in denominator)
"""

import argparse, csv, hashlib, json, math, re, subprocess, sys
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
# STREAMING DATA TABLE (from URL)
# ============================================================================
def stream_analyze(url, panel_cpgs, checkpoint_path=None):
    """Stream-download via curl | zcat and parse row-by-row."""
    # Use subprocess for: curl URL | zcat
    curl = subprocess.Popen(['curl','-sS','-L','--max-time','1800', url],
                            stdout=subprocess.PIPE)
    zcat = subprocess.Popen(['zcat'], stdin=curl.stdout, stdout=subprocess.PIPE,
                            bufsize=1024*1024)
    curl.stdout.close()  # allow curl to receive SIGPIPE
    stream = zcat.stdout

    # Read header (line 1) — up to newline
    header_bytes = b""
    while True:
        chunk = stream.read(65536)
        if not chunk:
            sys.exit("[FATAL] stream ended before header complete")
        header_bytes += chunk
        if b"\n" in chunk:
            # Split off the header row
            nl = header_bytes.find(b"\n")
            header_line = header_bytes[:nl].decode('utf-8')
            leftover    = header_bytes[nl+1:]
            break
    # Parse header
    header = header_line.split(',')
    n_cols = len(header)
    # first column is "CpG_ID"
    sample_beadchips = header[1:]
    n_samples = len(sample_beadchips)
    print(f"  Header: {n_cols} columns ({n_samples} samples)", flush=True)

    # Accumulators
    sum_A  = np.zeros(n_samples, dtype=np.float64)
    count  = np.zeros(n_samples, dtype=np.int64)
    cpgs_matched = set()
    n_lines = 0

    # Read rest of file line-by-line using a manual byte buffer + generator
    # STAY IN BYTES for non-matching rows (99.94%) — only decode on match.
    def line_iter_bytes():
        buffer = leftover
        while True:
            # Yield all complete lines from buffer (as bytes)
            while True:
                nl = buffer.find(b"\n")
                if nl < 0: break
                line_bytes = buffer[:nl]
                buffer = buffer[nl+1:]
                yield line_bytes
            # Read more
            chunk = stream.read(1024*1024)
            if not chunk:
                if buffer:
                    yield buffer
                return
            buffer += chunk

    # Pre-compute byte-encoded panel for fast membership testing
    panel_bytes = {c.encode('ascii') for c in panel_cpgs}

    # Status reporting
    import time
    t0 = time.time()
    last_report = t0
    for line_b in line_iter_bytes():
        n_lines += 1
        # Fast: find first comma and slice CpG ID in bytes
        first_comma = line_b.find(b',')
        if first_comma < 0:
            continue
        cpg_id_b = line_b[:first_comma]
        # Strip optional quote bytes
        if cpg_id_b.startswith(b'"'):
            cpg_id_b = cpg_id_b.strip(b'"')
        if cpg_id_b not in panel_bytes:
            # Print status periodically
            now = time.time()
            if now - last_report > 30:
                print(f"  ... {n_lines:>7} rows processed ({n_lines/(now-t0):.0f}/s), "
                      f"{len(cpgs_matched)}/{len(panel_cpgs)} panel CpGs found so far", flush=True)
                last_report = now
                # Write checkpoint so we can monitor from another process
                if checkpoint_path:
                    import json as _json
                    with open(checkpoint_path, 'w') as ck:
                        _json.dump({
                            'n_rows_processed': int(n_lines),
                            'elapsed_s':        float(now - t0),
                            'rows_per_s':       float(n_lines/(now-t0)),
                            'cpgs_matched':     len(cpgs_matched),
                            'cpgs_total':       len(panel_cpgs),
                            'complete':         False,
                        }, ck)
            continue
        # Matched — now decode and parse the values
        cpgs_matched.add(cpg_id_b.decode('ascii'))
        line = line_b.decode('utf-8', errors='replace')
        fields = line[first_comma+1:].rstrip('\n').split(',')
        for i in range(min(n_samples, len(fields))):
            v = fields[i]
            if v == '' or v == 'NA' or v == 'NaN':
                continue
            try:
                b = float(v)
            except ValueError:
                continue
            if math.isnan(b):
                continue
            sum_A[i] += H_binary(b) / H_MIN_IMMUNE
            count[i] += 1

    # Drain any remaining stdout from zcat
    stream.close()
    try:
        zcat.wait(timeout=5)
    except subprocess.TimeoutExpired:
        zcat.kill()
    try:
        curl.wait(timeout=5)
    except subprocess.TimeoutExpired:
        curl.kill()

    t1 = time.time()
    print(f"  STREAMING COMPLETE: {n_lines} rows in {t1-t0:.1f}s "
          f"({n_lines/(t1-t0):.0f}/s); matched {len(cpgs_matched)}/{len(panel_cpgs)} panel CpGs",
          flush=True)
    # Final checkpoint
    if checkpoint_path:
        import json as _json
        with open(checkpoint_path, 'w') as ck:
            _json.dump({
                'n_rows_processed': int(n_lines),
                'elapsed_s':        float(t1 - t0),
                'rows_per_s':       float(n_lines/(t1-t0)),
                'cpgs_matched':     len(cpgs_matched),
                'cpgs_total':       len(panel_cpgs),
                'complete':         True,
            }, ck)
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
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("T12 GSE314261 — St Jude Lifetime Cohort n=5013 streaming analysis", flush=True)
    print("=" * 78, flush=True)
    pan_sha = sha256_of_file(args.panel)
    meta_sha = sha256_of_file(args.metadata_csv)
    print(f"  panel sha256:        {pan_sha}", flush=True)
    print(f"  metadata sha256:     {meta_sha}", flush=True)
    print(f"  data URL:            {args.data_url}", flush=True)
    print(f"  H_min(immune):       {H_MIN_IMMUNE}", flush=True)
    print(f"  Random seed:         {RANDOM_SEED}", flush=True)
    print(f"  A formula:           CANONICAL", flush=True)
    print(flush=True)

    panel = load_panel(args.panel)
    print(f"Panel CpGs: {len(panel)}", flush=True)

    # Load metadata
    meta = {}
    with open(args.metadata_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            if row['beadchip_id']:
                meta[row['beadchip_id']] = row
    print(f"Metadata rows with beadchip_id: {len(meta)}", flush=True)
    # Count groups in metadata
    from collections import Counter
    groups = Counter(r.get('group','') for r in meta.values())
    sources = Counter(r.get('source','') for r in meta.values())
    print(f"  groups:  {dict(groups)}", flush=True)
    print(f"  sources: {dict(sources)}", flush=True)
    print(flush=True)

    print("Starting stream analysis (28 GB download, ~5-10 min expected)...", flush=True)
    import time
    t_start = time.time()
    checkpoint_path = out_dir / "STREAM_PROGRESS.json"
    sample_beadchips, A, count, matched = stream_analyze(
        args.data_url, panel, checkpoint_path=str(checkpoint_path))
    t_elapsed = time.time() - t_start
    print(f"Total elapsed: {t_elapsed:.1f}s", flush=True)
    print(flush=True)

    # Merge A-scores with metadata
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
        for r in rows:
            w.writerow(r)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV: {csv_path}  sha256: {csv_sha}", flush=True)
    print(f"  rows: {len(rows)}", flush=True)
    print(flush=True)

    # Per-sample summary
    valid_rows = [r for r in rows if not math.isnan(r['A_score'])]
    print(f"Valid per-sample A-scores: {len(valid_rows)} / {len(rows)}", flush=True)
    A_all = np.array([r['A_score'] for r in valid_rows])
    cnt   = np.array([r['n_cpgs_used'] for r in valid_rows])
    print(f"  A summary: mean={A_all.mean():.4f}  sd={A_all.std():.4f}", flush=True)
    print(f"  n_cpgs_used: min={cnt.min()}  max={cnt.max()}  median={int(np.median(cnt))}", flush=True)
    print(flush=True)

    # ========================================================================
    # ANALYSIS — SPECIFICITY TEST: cancer survivors vs community controls
    # ========================================================================
    # Restrict to blood samples
    blood = [r for r in rows if r['source'] == 'blood' and not math.isnan(r['A_score'])]
    survivors = [r['A_score'] for r in blood if r['group'] == 'cancer survivor']
    controls  = [r['A_score'] for r in blood if r['group'] == 'control']
    print("=" * 78, flush=True)
    print("PRIMARY TEST — Xu-538 specificity on non-BC pathology", flush=True)
    print("  (childhood cancer survivors vs community controls)", flush=True)
    print("=" * 78, flush=True)
    print(f"  n survivors (cases):  {len(survivors)}", flush=True)
    print(f"  n controls:           {len(controls)}", flush=True)
    print(flush=True)

    s_arr = np.asarray(survivors); c_arr = np.asarray(controls)
    d_obs = cohens_d(s_arr, c_arr)
    p_perm, _ = permutation_p(s_arr, c_arr, n_perm=5000, seed=RANDOM_SEED)
    ci_lo, ci_hi = bootstrap_ci(s_arr, c_arr, n_boot=1000, seed=RANDOM_SEED)
    delta = float(s_arr.mean() - c_arr.mean())
    print(f"  survivors A mean:  {s_arr.mean():.4f}  sd {s_arr.std(ddof=1):.4f}", flush=True)
    print(f"  controls  A mean:  {c_arr.mean():.4f}  sd {c_arr.std(ddof=1):.4f}", flush=True)
    print(f"  Δ (survivors - controls):  {delta:+.4f}", flush=True)
    print(f"  Cohen's d:                 {d_obs:+.3f}", flush=True)
    print(f"  p (5k perm):               {p_perm:.4f}", flush=True)
    print(f"  95% CI (1k bootstrap):     [{ci_lo:+.3f}, {ci_hi:+.3f}]", flush=True)
    print(flush=True)

    # Secondary — by diagnostic group (show effect direction across pathologies)
    print("=" * 78, flush=True)
    print("SECONDARY — stratified by diagnostic group (survivors only vs all controls)", flush=True)
    print("=" * 78, flush=True)
    dx_groups = sorted(set(r['diagnostic_group'] for r in blood
                           if r['group']=='cancer survivor' and r['diagnostic_group']))
    stratified = []
    for dx in dx_groups:
        dx_scores = [r['A_score'] for r in blood
                     if r['group']=='cancer survivor' and r['diagnostic_group']==dx]
        if len(dx_scores) < 10:
            continue
        dx_arr = np.asarray(dx_scores)
        d_strat = cohens_d(dx_arr, c_arr)
        print(f"  {dx[:50]:<50}  n={len(dx_scores):<4}  "
              f"mean={dx_arr.mean():.4f}  d_vs_controls={d_strat:+.3f}", flush=True)
        stratified.append({
            'diagnostic_group': dx,
            'n':                int(len(dx_scores)),
            'A_mean':           float(dx_arr.mean()),
            'A_sd':             float(dx_arr.std(ddof=1)),
            'cohens_d_vs_controls': float(d_strat),
        })
    print(flush=True)

    # Output JSON
    out = {
        'cohort':          'GSE314261_StJude_Lifetime_Cohort_2026',
        'test_type':       'specificity (NOT BC pre-dx replication)',
        'design':          ('4669 childhood cancer survivors + 342 community controls '
                            '(5005 blood + 8 saliva), Illumina EPIC. Specificity test '
                            'of Xu-538 adult BC panel against treatment-related late-effect biology.'),
        'caveats':         ('This is NOT a breast cancer pre-diagnostic replication. '
                            'Childhood cancer survivors have undergone chemotherapy/radiation '
                            'which can leave a treatment-related methylation signature. '
                            'A positive signal here would indicate the Xu-538 panel is '
                            'non-specific; a null result would indicate BC-specificity. '
                            'Either outcome does not replicate the EPIC-Italy d=+1.85 '
                            'pre-diagnostic finding.'),
        'data_url':        args.data_url,
        'n_samples_total': int(len(rows)),
        'n_samples_with_A':int(len(valid_rows)),
        'panel':           'xu538_breast',
        'panel_n_cpgs':    len(panel),
        'panel_matched':   int(len(matched)),
        'H_min_immune':    H_MIN_IMMUNE,
        'random_seed':     RANDOM_SEED,
        'streaming_elapsed_s': float(t_elapsed),
        'input_sha256':    {'panel': pan_sha, 'metadata_csv': meta_sha},
        'output_sha256':   {'per_sample_csv': csv_sha},
        'primary_specificity_test': {
            'n_survivors':    int(len(survivors)),
            'n_controls':     int(len(controls)),
            'survivors_A_mean': float(s_arr.mean()),
            'survivors_A_sd':   float(s_arr.std(ddof=1)),
            'controls_A_mean':  float(c_arr.mean()),
            'controls_A_sd':    float(c_arr.std(ddof=1)),
            'delta_A':          float(delta),
            'cohens_d':         float(d_obs),
            'p_perm':           float(p_perm),
            'CI95':             [float(ci_lo), float(ci_hi)],
        },
        'by_diagnostic_group': stratified,
    }
    json_path = out_dir / "GSE314261_analysis.json"
    with open(json_path, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"Output JSON: {json_path}", flush=True)
    print(f"  sha256: {sha256_of_file(json_path)}", flush=True)
    print(flush=True)
    print("T12 complete.", flush=True)

if __name__ == "__main__":
    main()
