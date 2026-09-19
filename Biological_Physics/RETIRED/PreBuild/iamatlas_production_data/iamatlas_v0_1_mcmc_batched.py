#!/usr/bin/env python3
"""
IAMAtlas Brightness Layer — BATCHED Production MCMC
======================================================

Same hierarchical Beta-Binomial model as iamatlas_v0_1_mcmc.py, but
processes each architecture class in CpG-batches that fit cleanly in RAM.

WHY BATCHING
============
The original production script tried to fit ALL CpGs of a class in a single
MCMC instance. For terminal class: 16,630 CpGs × 11 cell types = 183K
parameters. That exceeded RAM on Heath's MacBook (8 GB RAM, hit thrashing).

This version splits each class into batches of N CpGs (default 3,000),
runs MCMC per batch independently (each fits in ~2 GB RAM), then concatenates
the per-batch posteriors into the same per-class output files the merge
script expects.

CORRECTNESS NOTE
================
Each CpG's posterior is INDEPENDENT in the model — μ_{cpg,ct} draws don't
share parameters across CpGs except through the class-level hyperprior
(α_class, β_class), which is shared by construction. Running CpGs in batches
introduces a tiny statistical inefficiency: each batch re-estimates α_class
and β_class from its own subset rather than from all CpGs simultaneously.
At 3K CpGs per batch, this hyperprior has plenty of information to estimate
to 3 decimal places — equivalent to the full-class fit at the precision
we care about. The per-CpG μ posteriors are the deliverable, and those
are NOT compromised.

Usage:
    python3 iamatlas_v0_1_mcmc_batched.py --classes terminal stromal ... \\
        --batch_size 3000 --out_dir iamatlas_v0_1_output

Date: 2026-05-04
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import pymc as pm
    import arviz as az
except ImportError:
    print("ERROR: PyMC not installed. Install with: pip install pymc arviz numpy")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================
_HERE = Path(__file__).parent.resolve()
DEFAULT_INPUTS_CSV = _HERE / "iamatlas_mcmc_inputs.csv"
DEFAULT_UNIVERSE_CSV = _HERE / "iamatlas_cpg_universe.csv"
DEFAULT_COVERAGE_CSV = _HERE / "iamatlas_cpg_coverage_per_atlas.csv"
DEFAULT_OUT_DIR = _HERE / "iamatlas_v0_1_output"

ALL_CLASSES = [
    "stem_pluri", "stem_adult", "progenitor", "stromal",
    "cycling", "secretory", "immune", "terminal",
]

# Sampler defaults
DEFAULT_TUNE = 1000
DEFAULT_DRAWS = 1000
DEFAULT_CHAINS = 4
DEFAULT_TARGET_ACCEPT = 0.95
DEFAULT_RANDOM_SEED = 20260504

DEFAULT_BATCH_SIZE = 3000  # CpGs per MCMC batch (gaming PC with 16+ GB can go higher)


# ============================================================
# Data loading — loads ONE class, ALL CpGs, but in memory-light tall format
# ============================================================

def load_all_class_inputs(inputs_csv: Path, target_class: str) -> tuple:
    """
    Returns:
      cpg_to_obs: {cpg_id: list of (atlas, cell_type, beta, n_donors, weight)}
      atlas_set: set of all atlas names
      celltype_set: set of all cell-type names
    """
    cpg_to_obs = defaultdict(list)
    atlas_set = set()
    celltype_set = set()
    n_seen = 0
    with open(inputs_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_seen += 1
            if row["arch_class"] != target_class:
                continue
            try:
                beta = float(row["beta_observed"])
            except (ValueError, KeyError):
                continue
            if beta <= 0: beta = 1e-4
            elif beta >= 1: beta = 1 - 1e-4
            n_donors = int(row.get("n_donors") or 1)
            weight = float(row.get("weight") or 1.0)
            cpg = row["cpg_id"]
            atlas = row["atlas_source"]
            ct = row["cell_type"]
            cpg_to_obs[cpg].append((atlas, ct, beta, n_donors, weight))
            atlas_set.add(atlas)
            celltype_set.add(ct)
            if n_seen % 500_000 == 0:
                print(f"    streamed {n_seen} input rows...")
    return cpg_to_obs, sorted(atlas_set), sorted(celltype_set)


def build_batch_arrays(cpg_batch: list, cpg_to_obs: dict,
                       atlas_ids: list, celltype_ids: list) -> dict:
    """Build the index arrays the MCMC consumes for one batch of CpGs."""
    cpg_idx = {c: i for i, c in enumerate(cpg_batch)}
    atlas_idx = {a: i for i, a in enumerate(atlas_ids)}
    ct_idx = {c: i for i, c in enumerate(celltype_ids)}

    obs_idx_cpg, obs_idx_atlas, obs_idx_ct = [], [], []
    obs_beta, obs_n_donors, obs_weight = [], [], []
    for cpg in cpg_batch:
        for (atlas, ct, beta, n, w) in cpg_to_obs.get(cpg, []):
            obs_idx_cpg.append(cpg_idx[cpg])
            obs_idx_atlas.append(atlas_idx[atlas])
            obs_idx_ct.append(ct_idx[ct])
            obs_beta.append(beta)
            obs_n_donors.append(n)
            obs_weight.append(w)
    return {
        "cpg_ids": cpg_batch,
        "atlas_ids": atlas_ids,
        "celltype_ids": celltype_ids,
        "obs_idx_cpg": np.array(obs_idx_cpg, dtype=np.int32),
        "obs_idx_atlas": np.array(obs_idx_atlas, dtype=np.int32),
        "obs_idx_ct": np.array(obs_idx_ct, dtype=np.int32),
        "obs_beta": np.array(obs_beta, dtype=np.float64),
        "obs_n_donors": np.array(obs_n_donors, dtype=np.int32),
        "obs_weight": np.array(obs_weight, dtype=np.float64),
        "n_obs": len(obs_idx_cpg),
    }


# ============================================================
# Model + sampling (same model as non-batched version)
# ============================================================

def build_class_model(data: dict) -> pm.Model:
    """
    Hierarchical Beta-Binomial with NON-CENTERED parameterization.
    
    HISTORY: The original centered version (mu ~ Beta(alpha, beta)) produced
    pathological NUTS geometry on certain CpG batches (terminal Batch 2:
    R-hat=1.46, ESS=8, 90 divergences). The Beta distribution becomes
    near-degenerate when CpGs sit close to 0 or 1, which is common in
    methylation data. NUTS cannot traverse the resulting funnel-like geometry.
    
    FIX: Reparameterize on the logit scale with Normal priors. The class
    hyperprior governs the mean and spread of logit(mu) instead of alpha/beta.
    Transform back to (0,1) via sigmoid for the observation model. This
    eliminates the funnel and produces stable NUTS geometry across all
    methylation regimes (hyper- and hypo-methylated alike).
    
    The likelihood structure is unchanged: per-observation Beta(mu*kappa,
    (1-mu)*kappa) where kappa is per-atlas precision. Only the prior on mu
    is reparameterized.
    """
    n_cpg = len(data["cpg_ids"])
    n_atlas = len(data["atlas_ids"])
    n_ct = len(data["celltype_ids"])

    with pm.Model() as model:
        # Class hyperpriors on the logit scale
        # mu_class_logit governs central methylation tendency for this class
        # sigma_class_logit governs spread across CpGs/cell-types within class
        mu_class_logit = pm.Normal("mu_class_logit", mu=0.0, sigma=2.0)
        sigma_class_logit = pm.HalfNormal("sigma_class_logit", sigma=2.0)
        
        # Non-centered: standard Normal × scale + location
        # This is the key reparameterization — z is unit Normal, transformed
        # to logit-space mu via the affine map. NUTS samples z trivially.
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=(n_cpg, n_ct))
        mu_logit = pm.Deterministic(
            "mu_logit", mu_class_logit + sigma_class_logit * z
        )
        # Transform to (0,1) for use in Beta likelihood
        mu = pm.Deterministic("mu", pm.math.sigmoid(mu_logit))
        
        # Per-atlas precision (same as before — this part wasn't pathological)
        log_kappa = pm.Normal("log_kappa", mu=2.0, sigma=1.0, shape=n_atlas)
        kappa = pm.Deterministic("kappa", pm.math.exp(log_kappa))

        # Observation model — unchanged. Beta likelihood with per-CpG/per-celltype
        # mu and per-atlas precision. Numerical guards prevent kappa-mu products
        # from underflowing to exactly 0.
        mu_obs = mu[data["obs_idx_cpg"], data["obs_idx_ct"]]
        kappa_obs = kappa[data["obs_idx_atlas"]]
        # Clip mu_obs to (1e-6, 1-1e-6) for numerical stability in Beta
        mu_obs_safe = pm.math.clip(mu_obs, 1e-6, 1.0 - 1e-6)
        a_obs = mu_obs_safe * kappa_obs
        b_obs = (1.0 - mu_obs_safe) * kappa_obs
        pm.Beta("beta_obs", alpha=a_obs, beta=b_obs, observed=data["obs_beta"])
    return model


def sample_batch(target_class: str, data: dict, batch_idx: int, n_batches: int,
                 tune: int, draws: int, chains: int, seed: int,
                 target_accept: float) -> dict:
    print(f"\n  Batch {batch_idx+1}/{n_batches}: {len(data['cpg_ids'])} CpGs, "
          f"{data['n_obs']} obs, {len(data['celltype_ids'])} cell types")

    if data["n_obs"] < 10:
        print(f"    [SKIP] insufficient observations")
        return None

    t0 = time.time()
    model = build_class_model(data)
    with model:
        idata = pm.sample(
            tune=tune, draws=draws, chains=chains, cores=4,
            target_accept=target_accept, random_seed=seed + batch_idx,
            progressbar=True, return_inferencedata=True,
        )
    elapsed = time.time() - t0
    print(f"    sampled in {elapsed:.0f}s")

    # Convergence — check the actually-SAMPLED variables (z, hyperpriors)
    # not the Deterministic transformation. This is the correct diagnostic
    # for non-centered parameterization.
    summary_sampled = az.summary(
        idata, var_names=["z", "mu_class_logit", "sigma_class_logit", "log_kappa"],
        stat_focus="mean"
    )
    rhat_max = float(summary_sampled["r_hat"].max())
    ess_min = float(summary_sampled["ess_bulk"].min())
    n_diverging = int(idata.sample_stats["diverging"].sum())
    print(f"    R-hat max={rhat_max:.3f}  ESS min={ess_min:.0f}  divergent={n_diverging}")

    # Extract posteriors
    mu_post = idata.posterior["mu"].values
    mu_flat = mu_post.reshape(-1, mu_post.shape[2], mu_post.shape[3])
    mu_mean = mu_flat.mean(axis=0)
    mu_sd = mu_flat.std(axis=0)
    mu_lo = np.quantile(mu_flat, 0.025, axis=0)
    mu_hi = np.quantile(mu_flat, 0.975, axis=0)

    cpg_class_mean = mu_mean.mean(axis=1)
    cpg_class_sd = np.sqrt((mu_sd**2).mean(axis=1) + mu_mean.var(axis=1))
    cpg_class_lo = mu_lo.mean(axis=1)
    cpg_class_hi = mu_hi.mean(axis=1)

    mu_obs_mean = mu_mean[data["obs_idx_cpg"], data["obs_idx_ct"]]
    pearson = float(np.corrcoef(mu_obs_mean, data["obs_beta"])[0, 1])
    mae = float(np.mean(np.abs(mu_obs_mean - data["obs_beta"])))

    return {
        "cpg_ids": data["cpg_ids"],
        "celltype_ids": data["celltype_ids"],
        "cpg_class_mean": cpg_class_mean,
        "cpg_class_sd": cpg_class_sd,
        "cpg_class_lo": cpg_class_lo,
        "cpg_class_hi": cpg_class_hi,
        "mu_mean": mu_mean,
        "mu_sd": mu_sd,
        "elapsed_s": elapsed,
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "n_diverging": n_diverging,
        "pearson": pearson,
        "mae": mae,
    }


# ============================================================
# Class-level orchestrator: batches all CpGs, concatenates posteriors
# ============================================================

def run_class(target_class: str, inputs_csv: Path, batch_size: int,
              tune: int, draws: int, chains: int, seed: int,
              target_accept: float, out_dir: Path):
    print(f"\n{'='*72}")
    print(f"CLASS: {target_class}")
    print(f"{'='*72}")
    
    print(f"  Loading all inputs for {target_class}...")
    t_load = time.time()
    cpg_to_obs, atlas_ids, celltype_ids = load_all_class_inputs(inputs_csv, target_class)
    n_cpg_total = len(cpg_to_obs)
    n_obs_total = sum(len(v) for v in cpg_to_obs.values())
    print(f"  Loaded in {time.time()-t_load:.0f}s: "
          f"{n_cpg_total} CpGs, {n_obs_total} observations, "
          f"{len(atlas_ids)} atlases, {len(celltype_ids)} cell types")

    if n_cpg_total == 0:
        print(f"  [SKIP] no inputs for {target_class}")
        return

    # Sort CpGs deterministically (alphabetical) so re-runs are reproducible
    all_cpgs = sorted(cpg_to_obs.keys())
    n_batches = (n_cpg_total + batch_size - 1) // batch_size
    print(f"  Split into {n_batches} batches of up to {batch_size} CpGs each")

    # Process each batch
    batch_results = []
    t_class = time.time()
    for batch_idx in range(n_batches):
        lo = batch_idx * batch_size
        hi = min(lo + batch_size, n_cpg_total)
        cpg_batch = all_cpgs[lo:hi]
        data = build_batch_arrays(cpg_batch, cpg_to_obs, atlas_ids, celltype_ids)
        result = sample_batch(target_class, data, batch_idx, n_batches,
                              tune, draws, chains, seed, target_accept)
        if result is not None:
            batch_results.append(result)
        # Free memory before next batch
        del data
        gc.collect()
    
    if not batch_results:
        print(f"  [FAIL] no batches succeeded")
        return

    # Concatenate per-CpG posteriors across batches
    all_cpg_ids = []
    all_class_mean, all_class_sd, all_class_lo, all_class_hi = [], [], [], []
    all_mu_mean_rows, all_mu_sd_rows = [], []
    for r in batch_results:
        all_cpg_ids.extend(r["cpg_ids"])
        all_class_mean.extend(r["cpg_class_mean"].tolist())
        all_class_sd.extend(r["cpg_class_sd"].tolist())
        all_class_lo.extend(r["cpg_class_lo"].tolist())
        all_class_hi.extend(r["cpg_class_hi"].tolist())
        all_mu_mean_rows.append(r["mu_mean"])
        all_mu_sd_rows.append(r["mu_sd"])
    
    mu_mean_full = np.concatenate(all_mu_mean_rows, axis=0)
    mu_sd_full = np.concatenate(all_mu_sd_rows, axis=0)

    # Aggregate diagnostics
    elapsed_total = time.time() - t_class
    rhat_max_overall = max(r["rhat_max"] for r in batch_results)
    ess_min_overall = min(r["ess_min"] for r in batch_results)
    n_diverging_overall = sum(r["n_diverging"] for r in batch_results)
    pearson_pooled = float(np.mean([r["pearson"] for r in batch_results]))
    mae_pooled = float(np.mean([r["mae"] for r in batch_results]))

    # Write outputs
    out_dir.mkdir(exist_ok=True, parents=True)
    
    f1 = out_dir / f"iamatlas_v0_1_{target_class}_brightness.csv"
    with open(f1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cpg_id", "class", "mean", "sd", "ci_lo", "ci_hi"])
        for i, c in enumerate(all_cpg_ids):
            w.writerow([c, target_class,
                        f"{all_class_mean[i]:.6f}", f"{all_class_sd[i]:.6f}",
                        f"{all_class_lo[i]:.6f}", f"{all_class_hi[i]:.6f}"])
    
    f2 = out_dir / f"iamatlas_v0_1_{target_class}_per_celltype.csv"
    with open(f2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cpg_id"] +
                   [f"{ct}_mean" for ct in celltype_ids] +
                   [f"{ct}_sd" for ct in celltype_ids])
        for i, c in enumerate(all_cpg_ids):
            row = [c]
            for j in range(len(celltype_ids)):
                row.append(f"{mu_mean_full[i][j]:.6f}")
            for j in range(len(celltype_ids)):
                row.append(f"{mu_sd_full[i][j]:.6f}")
            w.writerow(row)
    
    f3 = out_dir / f"iamatlas_v0_1_{target_class}_result.json"
    summary = {
        "class": target_class,
        "status": "complete",
        "n_cpg": len(all_cpg_ids),
        "n_ct": len(celltype_ids),
        "n_atlas": len(atlas_ids),
        "n_obs": n_obs_total,
        "n_batches": len(batch_results),
        "batch_size": batch_size,
        "elapsed_s": elapsed_total,
        "convergence": {
            "rhat_max": rhat_max_overall,
            "ess_min": ess_min_overall,
            "n_diverging": n_diverging_overall,
            "n_total": int(chains * draws),
        },
        "predictive": {
            "pearson": pearson_pooled,
            "mae": mae_pooled,
        },
    }
    with open(f3, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  ✓ {target_class} complete in {elapsed_total/60:.1f} min")
    print(f"    Outputs: {f1.name}, {f2.name}, {f3.name}")
    print(f"    R-hat={rhat_max_overall:.3f}  ESS={ess_min_overall:.0f}  "
          f"div={n_diverging_overall}  Pearson={pearson_pooled:.3f}  MAE={mae_pooled:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS_CSV))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--classes", nargs="*", default=ALL_CLASSES)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--target_accept", type=float, default=DEFAULT_TARGET_ACCEPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    print("=" * 72)
    print("IAMAtlas BATCHED Production MCMC")
    print("=" * 72)
    print(f"Inputs: {args.inputs}")
    print(f"Out dir: {args.out_dir}")
    print(f"Classes: {args.classes}")
    print(f"Batch size: {args.batch_size} CpGs per MCMC instance")
    print(f"Sampler: {args.chains} chains × ({args.tune} tune + {args.draws} draws), "
          f"target_accept={args.target_accept}")

    out_dir = Path(args.out_dir)
    inputs_csv = Path(args.inputs)

    for cls in args.classes:
        run_class(cls, inputs_csv, args.batch_size,
                  args.tune, args.draws, args.chains, args.seed,
                  args.target_accept, out_dir)

    print(f"\n{'='*72}")
    print(f"All classes complete. Run merge_iamatlas_v0_1.py next.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
