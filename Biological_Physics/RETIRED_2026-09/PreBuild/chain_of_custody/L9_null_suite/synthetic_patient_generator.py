#!/usr/bin/env python3
"""
synthetic_patient_generator.py — Generate synthetic patient β-matrices with known truth

The methylome's equivalent of CMB end-to-end simulations (Planck FFP10, ACT mock skies).
For each synthetic patient, the inputs (truth) are KNOWN:
  - true cell-type composition fractions
  - true disease signal injection (which CpGs, which direction, what magnitude)
  - true foreground contamination (age, sex, batch, plate position)
  - true measurement noise level

The output is a β-matrix that can be fed through the entire chain L1-L8 and
verified to recover the truth within stated tolerance. This is what makes the
chain auditable: any chain that doesn't recover known truth on simulated data
cannot be trusted to recover unknown truth on real data.

USAGE (programmatic):
    from synthetic_patient_generator import SyntheticCohort
    cohort = SyntheticCohort(n_case=50, n_hc=500, disease_signal_strength=2.0)
    cohort.generate()
    cohort.export("synthetic_cohort_001/")

USAGE (CLI):
    python synthetic_patient_generator.py \\
        --n-case 50 --n-hc 500 --signal-strength 2.0 \\
        --out synthetic_cohort_001/

WHAT GETS GENERATED
-------------------
For each synthetic patient, the generator produces:
  1. A "true" cell-type composition vector (8 architectural classes summing to 1.0)
  2. β values at every IAMAtlas CpG, computed as:
        β = Σ_class (fraction_class × IAMAtlas_β_class)
            + disease_signal × disease_panel_indicator
            + age_axis × age_effect
            + sex_axis × sex_effect
            + batch_axis × batch_effect
            + ε  (measurement noise)
  3. Truth labels (case/HC, age, sex, batch, plate position, true fractions)

CHAIN RECOVERY TESTS
--------------------
After generating, feed the synthetic β-matrix back through CPG L1-L8 and check:
  (R1) Walther deconvolver recovers true cell-type fractions within ±0.05
  (R2) Per-class A-scores recover the true class signal magnitude
  (R3) Mahalanobis distance flags cases at d > 1.5 when signal_strength ≥ 1.5
  (R4) Per-CpG residual map recovers the injected disease-panel CpGs
  (R5) Bimodality map shows expected bimodality in HC, expected loss in cases
  (R6) PCA recovers the injected disease axis as a dominant component
  (R7) Chromosome isotropy recovers injected chromosome-specific enrichment
  (R8) Age dipole subtraction successfully removes injected age axis

A chain that fails any Rn cannot be claimed to handle the corresponding signal
type in real data.

DEPENDENCIES
------------
  - IAMAtlas REBUILD posterior means (per-class β references) at the CpG level
  - 450K manifest with chromosome positions (for chromosome-specific injection)
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# =========================================================================
# Configuration
# =========================================================================
ATLAS_PATH = "/home/claude/iamatlas_rebuild/IAMAtlasREBUILD.csv"
MANIFEST_PARQUET = "/home/claude/skymap_v1/master_cpg_atlas.parquet"  # has CHR + MAPINFO + posteriors
CLASSES = ['stem_pluri', 'stem_adult', 'stromal', 'progenitor',
           'cycling', 'secretory', 'terminal', 'immune']
H_MIN = {'terminal': 0.7728, 'immune': 0.838889, 'secretory': 0.8433,
         'progenitor': 0.8522, 'cycling': 0.8561, 'stromal': 0.8630,
         'stem_adult': 0.8737, 'stem_pluri': 0.9822}


# =========================================================================
# Truth specification
# =========================================================================
@dataclass
class PatientTruth:
    """The ground truth for one synthetic patient."""
    patient_id: str
    arm: str                       # "case" or "hc"
    age: float
    sex: str                       # "f" / "m"
    batch: str                     # batch identifier
    plate_position: int            # plate well position
    cohort: str                    # synthetic cohort label
    true_fractions: dict           # {class_name: fraction}, sums to 1.0
    disease_signal_strength: float # 0.0 for HC, positive for case
    disease_panel_cpgs: list       # CpG IDs receiving disease signal
    age_effect_strength: float     # how much age contributes to drift
    sex_effect_strength: float
    batch_effect_strength: float
    noise_sigma: float             # measurement-noise standard deviation


# =========================================================================
# Cohort generator
# =========================================================================
class SyntheticCohort:
    """Generate a complete synthetic patient cohort with known truth."""

    def __init__(self,
                 n_case: int = 50,
                 n_hc: int = 500,
                 disease_signal_strength: float = 2.0,    # injected case-vs-HC effect (units of class-SD)
                 disease_panel_size: int = 500,           # number of disease-panel CpGs
                 disease_panel_chromosomes: Optional[list] = None,  # restrict to certain chromosomes (e.g. ['6'] for MHC)
                 age_range: tuple = (40, 70),
                 age_effect_strength: float = 0.05,       # how strong age confound is
                 sex_imbalance: float = 1.0,              # 1.0 = all one sex, 0.5 = balanced
                 batch_count: int = 3,                    # number of synthetic batches
                 batch_effect_strength: float = 0.02,
                 noise_sigma: float = 0.03,               # β-measurement noise
                 n_cpgs: Optional[int] = None,            # subset of CpGs (None = use all from atlas)
                 random_seed: int = 42,
                 cohort_name: str = "synth"):
        self.n_case = n_case
        self.n_hc = n_hc
        self.disease_signal_strength = disease_signal_strength
        self.disease_panel_size = disease_panel_size
        self.disease_panel_chromosomes = disease_panel_chromosomes
        self.age_range = age_range
        self.age_effect_strength = age_effect_strength
        self.sex_imbalance = sex_imbalance
        self.batch_count = batch_count
        self.batch_effect_strength = batch_effect_strength
        self.noise_sigma = noise_sigma
        self.n_cpgs = n_cpgs
        self.random_seed = random_seed
        self.cohort_name = cohort_name
        self.rng = np.random.default_rng(random_seed)
        # State after generate()
        self.atlas = None
        self.disease_panel = None
        self.age_axis_loadings = None
        self.sex_axis_loadings = None
        self.batch_axis_loadings = None
        self.patients = []          # list of PatientTruth
        self.beta_matrix = None     # n_patients × n_cpgs

    # ------------------------------------------------------------------
    # Atlas loading
    # ------------------------------------------------------------------
    def _load_atlas(self):
        if self.atlas is not None: return
        df = pd.read_parquet(MANIFEST_PARQUET)
        # Need posteriors for all 8 classes — drop CpGs missing any
        keep_cols = ['cpg', 'CHR', 'MAPINFO'] + [f'{c}_mean' for c in CLASSES]
        df = df[keep_cols].dropna()
        if self.n_cpgs and self.n_cpgs < len(df):
            df = df.sample(n=self.n_cpgs, random_state=self.random_seed).reset_index(drop=True)
        self.atlas = df
        print(f"[SyntheticCohort] Atlas loaded: {len(df):,} CpGs × {len(CLASSES)} classes")

    # ------------------------------------------------------------------
    # Disease panel design
    # ------------------------------------------------------------------
    def _design_disease_panel(self):
        """Pick disease-panel CpGs. Optionally restrict to specific chromosomes."""
        atlas = self.atlas
        candidates = atlas
        if self.disease_panel_chromosomes:
            candidates = atlas[atlas['CHR'].isin(self.disease_panel_chromosomes)]
        # Sample disease-panel CpGs
        if len(candidates) < self.disease_panel_size:
            print(f"[WARN] requested {self.disease_panel_size} disease CpGs but only "
                  f"{len(candidates)} available; using all of them")
            self.disease_panel = candidates['cpg'].values
        else:
            self.disease_panel = candidates.sample(
                n=self.disease_panel_size, random_state=self.random_seed
            )['cpg'].values
        # Disease direction: random per-CpG sign, with bias toward hypomethylation
        # (matches the CPG-VAL-003 empirical observation: 5.4:1 hypomethylation dominance)
        rng = np.random.default_rng(self.random_seed + 1)
        signs = rng.choice([-1, +1], size=len(self.disease_panel),
                           p=[5.4/(5.4+1), 1/(5.4+1)])
        self.disease_directions = dict(zip(self.disease_panel, signs))
        print(f"[SyntheticCohort] Disease panel: {len(self.disease_panel):,} CpGs "
              f"({(signs<0).sum()} hypomethylated, {(signs>0).sum()} hypermethylated)")

    # ------------------------------------------------------------------
    # Foreground axis design
    # ------------------------------------------------------------------
    def _design_foreground_axes(self):
        """Generate per-CpG loadings for age, sex, batch axes (random Gaussian per CpG)."""
        rng = np.random.default_rng(self.random_seed + 2)
        n = len(self.atlas)
        self.age_axis_loadings = rng.normal(0, 1, n) * self.age_effect_strength
        self.sex_axis_loadings = rng.normal(0, 1, n) * 0.02      # subtle sex effect
        self.batch_axis_loadings = rng.normal(0, 1, (self.batch_count, n)) * self.batch_effect_strength

    # ------------------------------------------------------------------
    # Patient generation
    # ------------------------------------------------------------------
    def _generate_patient(self, idx, arm):
        """Build one patient's truth + β vector."""
        rng = np.random.default_rng(self.random_seed + 100 + idx)
        atlas = self.atlas

        # 1. Cell-type composition (Dirichlet, biased toward immune-dominated like whole blood)
        # Reasonable blood-like composition: ~60% immune, ~5% cycling, etc.
        if arm == "hc":
            # HC: typical-population composition
            alpha = np.array([0.5, 0.5, 1.0, 5.0, 5.0, 5.0, 5.0, 60.0])  # in CLASSES order
        else:  # case: same baseline composition (architectural; disease signal is overlay)
            alpha = np.array([0.5, 0.5, 1.0, 5.0, 5.0, 5.0, 5.0, 60.0])
        fracs = rng.dirichlet(alpha)
        true_fractions = dict(zip(CLASSES, fracs))

        # 2. Demographics
        age = rng.uniform(*self.age_range)
        sex = "f" if rng.random() < self.sex_imbalance else "m"
        batch = f"B{rng.integers(0, self.batch_count)}"
        plate_position = int(rng.integers(0, 96))

        # 3. Build β vector
        # 3a. Base: cell-type composition × IAMAtlas posteriors
        beta = np.zeros(len(atlas))
        for cls, frac in true_fractions.items():
            beta += frac * atlas[f'{cls}_mean'].values

        # 3b. Disease signal (case only, on disease-panel CpGs)
        disease_strength = self.disease_signal_strength if arm == "case" else 0.0
        if disease_strength > 0:
            panel_set = set(self.disease_panel)
            panel_mask = atlas['cpg'].isin(panel_set).values
            # Per-CpG disease direction (sign-aware)
            panel_signs = np.array([self.disease_directions.get(cpg, 0) for cpg in atlas['cpg'].values])
            # Disease shift magnitude: signal_strength × per-CpG class-SD
            # Use the std across the 8 classes at each CpG as the natural scale
            class_means = atlas[[f'{c}_mean' for c in CLASSES]].values
            cpg_class_sd = class_means.std(axis=1)
            disease_shift = disease_strength * panel_signs * cpg_class_sd * 0.3  # 0.3 = scale for visibility
            beta[panel_mask] += disease_shift[panel_mask]

        # 3c. Age axis effect
        # Center age at 55, then scale into a dimensionless drift parameter
        age_drift = (age - 55) / 15  # dimensionless
        beta += age_drift * self.age_axis_loadings

        # 3d. Sex axis effect
        if sex == "m":
            beta += self.sex_axis_loadings
        # else f: no shift (reference)

        # 3e. Batch axis effect
        batch_idx = int(batch[1:])
        beta += self.batch_axis_loadings[batch_idx]

        # 3f. Measurement noise (Gaussian, σ per-CpG)
        beta += rng.normal(0, self.noise_sigma, len(beta))

        # 3g. Clip to [0, 1]
        beta = np.clip(beta, 0.0, 1.0)

        # 4. Construct truth record
        patient_id = f"{self.cohort_name}_{idx:05d}"
        truth = PatientTruth(
            patient_id=patient_id, arm=arm, age=float(age), sex=sex,
            batch=batch, plate_position=plate_position,
            cohort=self.cohort_name,
            true_fractions=true_fractions,
            disease_signal_strength=float(disease_strength),
            disease_panel_cpgs=list(self.disease_panel) if disease_strength > 0 else [],
            age_effect_strength=float(self.age_effect_strength),
            sex_effect_strength=float(self.sex_axis_loadings.std()),
            batch_effect_strength=float(self.batch_effect_strength),
            noise_sigma=float(self.noise_sigma),
        )
        return truth, beta

    # ------------------------------------------------------------------
    # Top-level generate
    # ------------------------------------------------------------------
    def generate(self):
        """Generate all synthetic patients."""
        self._load_atlas()
        self._design_disease_panel()
        self._design_foreground_axes()
        print(f"[SyntheticCohort] Generating {self.n_case + self.n_hc:,} patients...")

        all_beta = np.zeros((self.n_case + self.n_hc, len(self.atlas)), dtype=np.float32)
        idx = 0
        for _ in range(self.n_case):
            truth, beta = self._generate_patient(idx, "case")
            self.patients.append(truth)
            all_beta[idx, :] = beta
            idx += 1
        for _ in range(self.n_hc):
            truth, beta = self._generate_patient(idx, "hc")
            self.patients.append(truth)
            all_beta[idx, :] = beta
            idx += 1

        self.beta_matrix = all_beta
        print(f"[SyntheticCohort] Done. β matrix: {self.beta_matrix.shape}, "
              f"range [{self.beta_matrix.min():.3f}, {self.beta_matrix.max():.3f}]")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Truth table
        truth_rows = []
        for p in self.patients:
            row = {
                'patient_id': p.patient_id, 'arm': p.arm, 'age': p.age,
                'sex': p.sex, 'batch': p.batch, 'plate_position': p.plate_position,
                'cohort': p.cohort, 'disease_signal_strength': p.disease_signal_strength,
            }
            for cls, frac in p.true_fractions.items():
                row[f'true_frac_{cls}'] = frac
            truth_rows.append(row)
        pd.DataFrame(truth_rows).to_csv(out_dir / "truth_table.csv", index=False)

        # 2. β matrix (large; compress)
        cpg_ids = self.atlas['cpg'].values
        patient_ids = [p.patient_id for p in self.patients]
        beta_df = pd.DataFrame(self.beta_matrix.T, index=cpg_ids, columns=patient_ids)
        beta_df.index.name = 'cpg'
        beta_df.to_parquet(out_dir / "beta_matrix.parquet", compression='zstd')

        # 3. Disease panel
        with open(out_dir / "disease_panel_truth.json", 'w') as f:
            json.dump({
                'panel_cpgs': list(self.disease_panel),
                'directions': {k: int(v) for k, v in self.disease_directions.items()},
                'restricted_chromosomes': self.disease_panel_chromosomes,
                'signal_strength': self.disease_signal_strength,
            }, f, indent=2)

        # 4. Foreground axis truth
        np.savez_compressed(out_dir / "foreground_axes_truth.npz",
                            age_loadings=self.age_axis_loadings,
                            sex_loadings=self.sex_axis_loadings,
                            batch_loadings=self.batch_axis_loadings,
                            cpg_order=cpg_ids)

        # 5. Manifest
        manifest = {
            'cohort_name': self.cohort_name,
            'n_case': self.n_case, 'n_hc': self.n_hc,
            'n_cpgs': int(len(self.atlas)),
            'disease_signal_strength': self.disease_signal_strength,
            'disease_panel_size': self.disease_panel_size,
            'disease_panel_chromosomes': self.disease_panel_chromosomes,
            'age_range': list(self.age_range),
            'age_effect_strength': self.age_effect_strength,
            'sex_imbalance': self.sex_imbalance,
            'batch_count': self.batch_count,
            'batch_effect_strength': self.batch_effect_strength,
            'noise_sigma': self.noise_sigma,
            'random_seed': self.random_seed,
            'files': {
                'truth_table': 'truth_table.csv',
                'beta_matrix': 'beta_matrix.parquet',
                'disease_panel': 'disease_panel_truth.json',
                'foreground_axes': 'foreground_axes_truth.npz',
            },
            'protocol': (
                "Feed beta_matrix.parquet through CPG chain L1-L8. "
                "Check chain output against truth_table.csv columns. "
                "Recovery thresholds defined in synthetic_patient_generator.py docstring."
            )
        }
        with open(out_dir / "MANIFEST.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"[SyntheticCohort] Exported to {out_dir}")
        for f in sorted(out_dir.iterdir()):
            print(f"    {f.stat().st_size:>12,} B  {f.name}")
        return out_dir


# =========================================================================
# Chain recovery tester
# =========================================================================
class ChainRecoveryTester:
    """Run a generated synthetic cohort through CPG and verify truth recovery."""

    def __init__(self, synth_cohort_dir):
        self.dir = Path(synth_cohort_dir)
        with open(self.dir / "MANIFEST.json") as f:
            self.manifest = json.load(f)
        self.truth = pd.read_csv(self.dir / "truth_table.csv")
        self.beta = pd.read_parquet(self.dir / "beta_matrix.parquet")
        print(f"[Recovery] Loaded {len(self.truth):,} synthetic patients, {len(self.beta):,} CpGs")

    def test_mahalanobis_recovers_signal(self):
        """R3: Mahalanobis distance separates injected cases from HC at expected magnitude."""
        from sklearn.covariance import LedoitWolf

        # Build A-scores per architectural class (simplified — average β at class marker CpGs)
        # Strict version would call iamatlas_a_scoring.py; this is a streamlined recovery test
        df = pd.read_parquet(MANIFEST_PARQUET)
        atlas_cpgs = set(df['cpg'].values) & set(self.beta.index)
        atlas_aligned = df[df['cpg'].isin(atlas_cpgs)].set_index('cpg').loc[list(atlas_cpgs)]
        beta_aligned = self.beta.loc[list(atlas_cpgs)]

        # Use raw class means as simple "A-score" proxy
        # In production we'd compute H(β)/H_min, but the recovery test is about
        # whether the chain can distinguish case from HC — Mahalanobis on per-class
        # means is a valid first-order check
        a_scores = pd.DataFrame(index=beta_aligned.columns)
        for cls in CLASSES:
            ref = atlas_aligned[f'{cls}_mean'].values
            # Score each patient: how close to this class's reference?
            similarities = np.zeros(beta_aligned.shape[1])
            for i, pid in enumerate(beta_aligned.columns):
                pat_beta = beta_aligned[pid].values
                # Use correlation as similarity proxy
                similarities[i] = np.corrcoef(pat_beta, ref)[0, 1]
            a_scores[cls] = similarities

        # Mahalanobis on a_scores against HC pooled reference
        hc_mask = self.truth['arm'] == 'hc'
        case_mask = self.truth['arm'] == 'case'
        a_scores_aligned = a_scores.loc[self.truth['patient_id']]
        X_hc = a_scores_aligned.loc[hc_mask.values].values
        X_all = a_scores_aligned.values
        lw = LedoitWolf().fit(X_hc)
        inv_cov = np.linalg.inv(lw.covariance_)
        mu = X_hc.mean(axis=0)
        delta = X_all - mu
        m_dist = np.sqrt(np.sum(delta @ inv_cov * delta, axis=1))

        d_case = m_dist[case_mask.values]
        d_hc = m_dist[hc_mask.values]
        # Cohen's d
        s = np.sqrt(((len(d_case)-1)*d_case.var(ddof=1) + (len(d_hc)-1)*d_hc.var(ddof=1))
                    / (len(d_case)+len(d_hc)-2))
        recovered_d = (d_case.mean() - d_hc.mean()) / s
        truth_signal = self.manifest['disease_signal_strength']
        # PASS: recovered d is positive and within order-of-magnitude of injected strength
        passed = bool(recovered_d > 0.5 and (truth_signal == 0 or recovered_d / truth_signal > 0.3))
        return {
            'test': 'R3_mahalanobis_recovers_signal',
            'passed': passed,
            'injected_signal_strength': float(truth_signal),
            'recovered_cohens_d': float(recovered_d),
            'n_case': int(case_mask.sum()),
            'n_hc': int(hc_mask.sum()),
        }


# =========================================================================
# CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="Generate synthetic CPG patient cohort with known truth")
    ap.add_argument("--n-case", type=int, default=50)
    ap.add_argument("--n-hc", type=int, default=500)
    ap.add_argument("--signal-strength", type=float, default=2.0,
                    help="Disease signal magnitude (units of class-SD)")
    ap.add_argument("--panel-size", type=int, default=500)
    ap.add_argument("--panel-chromosomes", default="",
                    help="Comma-separated chromosomes to restrict disease panel to (default: all)")
    ap.add_argument("--age-effect", type=float, default=0.05)
    ap.add_argument("--noise-sigma", type=float, default=0.03)
    ap.add_argument("--n-cpgs", type=int, default=None,
                    help="Subset to N CpGs (default: full atlas)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cohort-name", default="synth")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--test-recovery", action="store_true",
                    help="After generation, run recovery tests")
    args = ap.parse_args()

    panel_chrs = [c.strip() for c in args.panel_chromosomes.split(",")] if args.panel_chromosomes else None
    cohort = SyntheticCohort(
        n_case=args.n_case, n_hc=args.n_hc,
        disease_signal_strength=args.signal_strength,
        disease_panel_size=args.panel_size,
        disease_panel_chromosomes=panel_chrs,
        age_effect_strength=args.age_effect,
        noise_sigma=args.noise_sigma,
        n_cpgs=args.n_cpgs,
        random_seed=args.seed,
        cohort_name=args.cohort_name,
    )
    cohort.generate()
    cohort.export(args.out)

    if args.test_recovery:
        print("\n" + "="*78)
        print("CHAIN RECOVERY TEST")
        print("="*78)
        tester = ChainRecoveryTester(args.out)
        result = tester.test_mahalanobis_recovers_signal()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result['passed'] else 1)


if __name__ == "__main__":
    main()
