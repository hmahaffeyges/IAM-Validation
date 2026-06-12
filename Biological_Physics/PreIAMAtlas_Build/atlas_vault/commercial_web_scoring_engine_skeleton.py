"""
EDEAR Commercial Scoring Engine — Reference Skeleton
=====================================================

This is the **loading and scoring pattern** for the production EDEAR server
(commercial.web.py). It is NOT the full production engine — it shows how
the server loads the atlas vault and computes A-scores against H_min anchors.

What this skeleton demonstrates:
  1. How to load every atlas / reference matrix from the vault at startup
  2. How to compute Stage 1, Stage 2, Stage 3 A-scores against H_min anchors
  3. How to produce a customer-facing report (cellular age, per-class A-score,
     immune class signature, tier classification)
  4. How to handle 450K vs EPIC platform differences
  5. How to enforce CHK-3.2 cross-cohort baseline check before reporting

What this skeleton is NOT:
  - The H_min derivation chain (The Recipe — patent-protected, held separately)
  - The pre-registration / VAL pipeline (lives in validation_runs/)
  - The PDF report builder (FullVersion_build_gape_issue002.py handles that)
  - The Flask web frontend (separate concern)
  - The customer auth / billing / data-handling (separate concern)

When the production server is built, this skeleton becomes the seed for the
scoring core. It deliberately leaves H_min values, scoring formulas, and any
patented methodology as opaque function calls (`recipe.score(...)`) — the
real production engine imports those from a private module.

Heath W. Mahaffey  |  IAMPerformance Research Initiative
2026-04-26  |  Patents pending 64/012,720, 64/014,568
"""

from pathlib import Path
import json
import hashlib
import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

VAULT_ROOT = Path(__file__).parent
# In production: VAULT_ROOT = Path("/var/edear/atlas_vault")
# This skeleton lives inside the vault itself; in production move the vault
# elsewhere on disk and point VAULT_ROOT at it.

# H_min anchors per architecture class (8 classes from GAPE_WEB_v13)
# These values are calibrated from G-002 + G-003b MCMC posteriors.
# In the production engine these come from the Recipe module.
H_MIN_ANCHORS = {
    "stem_pluri":  0.9822,
    "stem_adult":  0.8737,
    "progenitor":  0.8522,
    "stromal":     0.8630,
    "cycling":     0.8561,
    "secretory":   0.8433,
    "immune":      0.8389,
    "terminal":    0.7728,
}

# Cell-class tier thresholds (consistent with breast-epic / ad-immune / glioma-epic)
# A < lower bound = NORMAL or BELOW_NORMAL (homogenization)
# A in interval = MARGINAL / DETECTABLE / URGENT
# A > upper bound = FLOOR_BREACH
TIER_THRESHOLDS = {
    "BELOW_NORMAL":   (None, 0.99),
    "NORMAL":         (0.99, 1.01),
    "MARGINAL":       (1.01, 1.05),
    "DETECTABLE":     (1.05, 1.10),
    "URGENT":         (1.10, 1.15),
    "FLOOR_BREACH":   (1.15, None),
}


# ═══════════════════════════════════════════════════════════════════════════
# CORE PHYSICS — Shannon binary entropy
# ═══════════════════════════════════════════════════════════════════════════

def H(beta):
    """Shannon binary entropy of Bernoulli(beta), in bits.
    H(0) = H(1) = 0; H(0.5) = 1.
    """
    if beta <= 0.0 or beta >= 1.0:
        return 0.0
    return -beta * math.log2(beta) - (1.0 - beta) * math.log2(1.0 - beta)


def H_vec(betas):
    """Vectorized binary entropy."""
    arr = np.asarray(betas, dtype=float)
    out = np.zeros_like(arr)
    mask = (arr > 0.0) & (arr < 1.0)
    out[mask] = -arr[mask] * np.log2(arr[mask]) - (1.0 - arr[mask]) * np.log2(1.0 - arr[mask])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# VAULT LOADER
# ═══════════════════════════════════════════════════════════════════════════

class AtlasVault:
    """
    Loads every atlas / reference matrix from the vault at server startup.

    The vault layout:
        atlas_vault/
            INVENTORY.json
            README.md
            stage2_cell_of_origin/
                loyfer_moss_2018/reference_atlas.csv
                episcore_zhu_teschendorff_2022/{14 tissues × 2 matrices}.csv
                caggiano_celfie_2021/tim_matrix.txt
                sabedot_gelb_2021/GeLB.R
                marlin_capper_training/{R scripts + filter probes}
            stage3_immune_fraction/
                salas_blood_epic_idol/IDOLOptimizedCpGs_compTable.csv (450 EPIC)
                salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv (350 450K)
                salas_idol_ext/{Pheno + metadata + R wrapper}
                unilife_guo_2025/centUniLIFE_reference_matrix.csv (1906 × 19)
                epidish_companion_panels/{cent12CT, centBloodSub, ...}.csv
    """

    def __init__(self, vault_root: Path = VAULT_ROOT):
        self.vault_root = Path(vault_root)
        if not self.vault_root.exists():
            raise FileNotFoundError(f"Atlas vault not found at {self.vault_root}")
        self.inventory = self._load_inventory()
        self.atlases = {}
        self._load_all()

    def _load_inventory(self) -> list:
        """Load INVENTORY.json — the SHA-256 catalog of every file."""
        inv_path = self.vault_root / "INVENTORY.json"
        if not inv_path.exists():
            raise FileNotFoundError(f"Vault INVENTORY.json missing at {inv_path}")
        with open(inv_path) as f:
            inventory = json.load(f)
        logger.info(f"Vault inventory: {len(inventory)} files catalogued")
        return inventory

    def verify_integrity(self) -> bool:
        """Verify every file in the vault matches its INVENTORY SHA-256."""
        bad = []
        for entry in self.inventory:
            f = self.vault_root / entry["path"]
            if not f.exists():
                bad.append((entry["path"], "missing"))
                continue
            actual = hashlib.sha256(f.read_bytes()).hexdigest()
            if actual != entry["sha256"]:
                bad.append((entry["path"], f"SHA mismatch (expected {entry['sha256'][:16]}, got {actual[:16]})"))
        if bad:
            for path, reason in bad:
                logger.error(f"Vault integrity FAIL: {path} — {reason}")
            return False
        logger.info(f"Vault integrity OK: all {len(self.inventory)} files verified")
        return True

    def _load_all(self):
        """Load every atlas / reference matrix into self.atlases dict."""
        # Stage 2 — Loyfer/Moss
        self.atlases["loyfer_moss"] = pd.read_csv(
            self.vault_root / "stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv",
            index_col=0,
        )
        logger.info(f"Loaded Loyfer/Moss: {self.atlases['loyfer_moss'].shape[0]} CpGs × {self.atlases['loyfer_moss'].shape[1]} cell types")

        # Stage 2 — EpiSCORE (28 matrices)
        episcore_dir = self.vault_root / "stage2_cell_of_origin/episcore_zhu_teschendorff_2022"
        self.atlases["episcore"] = {}
        for csv_file in sorted(episcore_dir.glob("*.csv")):
            tissue_key = csv_file.stem  # e.g. "BreastRef__mrefBreast_m"
            self.atlases["episcore"][tissue_key] = pd.read_csv(csv_file, index_col=0)
        logger.info(f"Loaded EpiSCORE: {len(self.atlases['episcore'])} tissue reference matrices")

        # Stage 2 — Caggiano CelFiE TIM (WGBS-region)
        celfie_path = self.vault_root / "stage2_cell_of_origin/caggiano_celfie_2021/tim_matrix.txt"
        self.atlases["caggiano_tim"] = pd.read_csv(celfie_path, sep="\t", low_memory=False)
        logger.info(f"Loaded Caggiano CelFiE TIM: {self.atlases['caggiano_tim'].shape[0]} markers × 19 tissues (WGBS-region)")

        # Stage 3 — Salas Blood.EPIC IDOL
        salas_dir = self.vault_root / "stage3_immune_fraction/salas_blood_epic_idol"
        self.atlases["salas_blood_epic"] = pd.read_csv(salas_dir / "IDOLOptimizedCpGs_compTable.csv", index_col=0)
        self.atlases["salas_blood_450k"] = pd.read_csv(salas_dir / "IDOLOptimizedCpGs450k_compTable.csv", index_col=0)
        logger.info(f"Loaded Salas Blood.EPIC IDOL: {self.atlases['salas_blood_epic'].shape[0]} EPIC CpGs × 6 cell types; "
                    f"{self.atlases['salas_blood_450k'].shape[0]} 450K legacy CpGs × 6 cell types")

        # Stage 3 — UniLIFE
        self.atlases["unilife"] = pd.read_csv(
            self.vault_root / "stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv",
            index_col=0,
        )
        logger.info(f"Loaded UniLIFE: {self.atlases['unilife'].shape[0]} CpGs × {self.atlases['unilife'].shape[1]} immune cell types")

        # Stage 3 — EpiDISH companion panels
        epidish_dir = self.vault_root / "stage3_immune_fraction/epidish_companion_panels"
        self.atlases["epidish_companion"] = {}
        for csv_file in sorted(epidish_dir.glob("*_reference_matrix.csv")):
            panel_key = csv_file.stem.replace("_reference_matrix", "")
            self.atlases["epidish_companion"][panel_key] = pd.read_csv(csv_file, index_col=0)
        logger.info(f"Loaded EpiDISH companion: {len(self.atlases['epidish_companion'])} panels "
                    f"({list(self.atlases['epidish_companion'].keys())})")

        logger.info(f"Vault load complete: 8 atlases / "
                    f"{1 + len(self.atlases['episcore']) + 1 + 2 + 1 + len(self.atlases['epidish_companion'])} "
                    f"reference matrices in memory")


# ═══════════════════════════════════════════════════════════════════════════
# IDAT INPUT — placeholder for the customer's β-matrix
# ═══════════════════════════════════════════════════════════════════════════

def load_customer_idat(idat_path: Path) -> pd.DataFrame:
    """
    Convert customer IDAT files to β-value matrix.

    Production implementation: use sesame (R) or methylprep (Python) to extract
    β values from IDAT pairs. Returns a DataFrame indexed by CpG_ID with one
    column per sample.

    This skeleton just defines the interface. Real implementation would call
    the sesame / methylprep pipeline.
    """
    raise NotImplementedError(
        "IDAT extraction is a separate concern. Use sesame or methylprep in production."
    )


# ═══════════════════════════════════════════════════════════════════════════
# A-SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

class ScoringEngine:
    """
    Computes Stage 1 / Stage 2 / Stage 3 A-scores for a customer's β-matrix
    against the loaded atlas vault.

    Run-everything architecture: every IDAT runs every panel, every atlas,
    every cell class. No conditional gating. The customer report layer
    decides what to display; the scoring layer computes everything.
    """

    def __init__(self, vault: AtlasVault):
        self.vault = vault

    def compute_a_score_per_class(self, beta_vector: np.ndarray, cls: str) -> float:
        """
        Compute A-score for a vector of β values against the H_min anchor
        for the given architecture class.

        A_class = mean(H(β) / H_min(class))

        β values outside [0,1] are filtered out before averaging.
        """
        if cls not in H_MIN_ANCHORS:
            raise KeyError(f"Unknown architecture class: {cls}")
        h_min = H_MIN_ANCHORS[cls]
        h_vals = H_vec(beta_vector)
        valid = (h_vals > 0) & np.isfinite(h_vals)
        if not valid.any():
            return float("nan")
        return float(np.mean(h_vals[valid] / h_min))

    def compute_stage_2_per_tile(self, betas: pd.Series, atlas_name: str = "loyfer_moss") -> dict:
        """
        Stage 2 — for each cell-type tile in the atlas, score the customer's
        β values on that tile's top-N discriminating CpGs against the cell
        type's architecture-class H_min.

        Returns: dict {tile_name: {a_score, n_cpgs, cell_class}}
        """
        atlas = self.vault.atlases[atlas_name]
        if atlas_name == "loyfer_moss":
            return self._stage_2_loyfer(betas, atlas)
        elif atlas_name == "episcore":
            return self._stage_2_episcore(betas)
        else:
            raise NotImplementedError(f"Stage 2 not implemented for atlas {atlas_name}")

    def _stage_2_loyfer(self, betas: pd.Series, atlas: pd.DataFrame) -> dict:
        """Stage 2 scoring against the Loyfer/Moss 25-cell-type atlas."""
        results = {}
        # Map each cell type to its architecture class — see GAPE_WEB_v13 _ARCH
        cell_to_class = {
            "B-cells_EPIC": "immune", "CD4T-cells_EPIC": "immune", "CD8T-cells_EPIC": "immune",
            "NK-cells_EPIC": "immune", "Monocytes_EPIC": "immune", "Neutrophils_EPIC": "immune",
            "Cortical_neurons": "terminal", "Left_atrium": "terminal",
            "Hepatocytes": "secretory", "Pancreatic_beta_cells": "secretory",
            "Pancreatic_acinar_cells": "secretory", "Pancreatic_duct_cells": "secretory",
            "Breast": "secretory", "Prostate": "secretory", "Thyroid": "secretory",
            "Lung_cells": "cycling", "Colon_epithelial_cells": "cycling",
            "Bladder": "cycling", "Kidney": "cycling", "Upper_GI": "cycling",
            "Uterus_cervix": "cycling", "Head_and_neck_larynx": "cycling",
            "Vascular_endothelial_cells": "stromal", "Adipocytes": "stromal",
            "Erythrocyte_progenitors": "progenitor",
        }

        common_cpgs = atlas.index.intersection(betas.index)
        if len(common_cpgs) == 0:
            logger.warning("No CpGs intersect between customer β and Loyfer/Moss atlas")
            return results

        atlas_sub = atlas.loc[common_cpgs]
        betas_sub = betas.loc[common_cpgs]

        for tile_name in atlas.columns:
            cls = cell_to_class.get(tile_name)
            if cls is None:
                continue
            # Discriminating CpGs: top-100 by |β(this cell) − mean(β(other cells))|
            other_cols = [c for c in atlas_sub.columns if c != tile_name]
            mean_others = atlas_sub[other_cols].mean(axis=1)
            specificity = (atlas_sub[tile_name] - mean_others).abs()
            top_100 = specificity.nlargest(100).index
            beta_vals = betas_sub.loc[top_100].values
            a = self.compute_a_score_per_class(beta_vals, cls)
            results[tile_name] = {
                "a_score": a,
                "n_cpgs": len(top_100),
                "cell_class": cls,
                "atlas": "loyfer_moss",
            }
        return results

    def _stage_2_episcore(self, betas: pd.Series) -> dict:
        """Stage 2 scoring against EpiSCORE 14-tissue panel."""
        # NOTE: EpiSCORE markers are gene-symbol or Entrez-ID indexed,
        # not CpG-ID indexed. Production engine must map gene → CpG via
        # the probeInfo450k.rda / probeInfo850k.rda bridges. Skeleton only.
        results = {}
        for tissue_key, ref in self.vault.atlases["episcore"].items():
            results[tissue_key] = {
                "n_markers": ref.shape[0],
                "cell_types": list(ref.columns),
                "_status": "Gene→CpG mapping not implemented in skeleton",
            }
        return results

    def compute_stage_3_immune_fractions(self, betas: pd.Series, panel: str = "salas") -> dict:
        """
        Stage 3 — immune cell fractions via reference-based deconvolution.

        Production: use EpiDISH RPC, CIBERSORT, or constrained quadratic
        programming (CP/QP) against the chosen reference matrix.

        This skeleton returns the reference matrix shape and intersection
        size; real numerical deconvolution is a separate concern.
        """
        if panel == "salas":
            ref = self.vault.atlases["salas_blood_epic"]
        elif panel == "unilife":
            ref = self.vault.atlases["unilife"]
        elif panel.startswith("epidish_"):
            sub = panel.replace("epidish_", "")
            ref = self.vault.atlases["epidish_companion"][sub]
        else:
            raise KeyError(f"Unknown Stage 3 panel: {panel}")

        common = ref.index.intersection(betas.index)
        return {
            "panel": panel,
            "ref_shape": list(ref.shape),
            "cell_types": list(ref.columns),
            "n_common_cpgs": len(common),
            "_status": "Deconvolution algorithm (RPC/CBS/CP-QP) not implemented in skeleton",
        }


# ═══════════════════════════════════════════════════════════════════════════
# CHK-3.2 CROSS-COHORT BASELINE CHECK (mandatory under run-everything)
# ═══════════════════════════════════════════════════════════════════════════

def chk_3_2_baseline_check(customer_per_tile: dict, healthy_baseline: dict, threshold: float = 1.0) -> dict:
    """
    Verify that every tile's customer A-score is within `threshold` anchor-SDs
    of the healthy baseline reference. Tiles outside threshold are flagged.

    Production: healthy baseline comes from a frozen reference cohort (e.g.
    GSE51057 healthy controls, age- and platform-matched) per-tile mean and SD.
    """
    flagged = {}
    for tile, customer_data in customer_per_tile.items():
        if tile not in healthy_baseline:
            continue
        bl_mean = healthy_baseline[tile]["mean"]
        bl_sd = healthy_baseline[tile]["sd"]
        if bl_sd <= 0:
            continue
        z = (customer_data["a_score"] - bl_mean) / bl_sd
        if abs(z) > threshold:
            flagged[tile] = {"customer_a": customer_data["a_score"], "z": z, "threshold": threshold}
    return flagged


# ═══════════════════════════════════════════════════════════════════════════
# TIER CLASSIFICATION (for customer-facing output)
# ═══════════════════════════════════════════════════════════════════════════

def classify_tier(a_score: float) -> str:
    if not math.isfinite(a_score):
        return "INVALID"
    for tier, (lo, hi) in TIER_THRESHOLDS.items():
        lo_ok = (lo is None) or (a_score >= lo)
        hi_ok = (hi is None) or (a_score < hi)
        if lo_ok and hi_ok:
            return tier
    return "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOMER REPORT BUILDER (skeleton)
# ═══════════════════════════════════════════════════════════════════════════

def build_customer_report(scoring_engine: ScoringEngine, customer_betas: pd.Series, customer_age: int) -> dict:
    """
    Run all stages on a customer's β-matrix and produce a customer-facing
    report dict. Production engine renders this into PDF / web JSON.
    """
    report = {
        "customer_age": customer_age,
        "platform": "EPIC",  # determined from IDAT manifest
        "stages": {},
    }

    # Stage 2 — Loyfer/Moss per-tile
    stage2_loyfer = scoring_engine.compute_stage_2_per_tile(customer_betas, atlas_name="loyfer_moss")
    report["stages"]["stage_2_loyfer"] = stage2_loyfer

    # Stage 2 — EpiSCORE (14 tissues, when integration VAL lands)
    stage2_episcore = scoring_engine.compute_stage_2_per_tile(customer_betas, atlas_name="episcore")
    report["stages"]["stage_2_episcore"] = stage2_episcore

    # Stage 3 — both Salas (production) and UniLIFE (Queue-1 #1 head-to-head)
    report["stages"]["stage_3_salas"] = scoring_engine.compute_stage_3_immune_fractions(customer_betas, panel="salas")
    report["stages"]["stage_3_unilife"] = scoring_engine.compute_stage_3_immune_fractions(customer_betas, panel="unilife")

    # Tier classification per Stage 2 tile
    report["tier_per_tile"] = {
        tile: classify_tier(data["a_score"])
        for tile, data in stage2_loyfer.items()
        if "a_score" in data
    }

    # Cellular age — production uses Horvath/Hannum until 17-tissue Ageing Atlas integration
    report["cellular_age"] = "TODO: Horvath/Hannum clock not in skeleton"

    # Customer-facing summary
    report["customer_summary"] = {
        "cellular_age": report["cellular_age"],
        "immune_class_signature": "TODO: derive from Stage 3 fractions",
        "tiles_above_normal": [t for t, tier in report["tier_per_tile"].items()
                               if tier in ("MARGINAL", "DETECTABLE", "URGENT", "FLOOR_BREACH")],
        "tiles_below_normal": [t for t, tier in report["tier_per_tile"].items()
                               if tier == "BELOW_NORMAL"],
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP — what the production server does at boot
# ═══════════════════════════════════════════════════════════════════════════

def server_startup():
    """
    Production server boot sequence:
      1. Verify atlas vault integrity (SHA-256 against INVENTORY.json)
      2. Load every atlas / reference matrix into memory
      3. Initialize scoring engine
      4. Ready to accept customer IDATs
    """
    logger.info("=" * 70)
    logger.info("EDEAR Commercial Scoring Engine — startup")
    logger.info("=" * 70)
    vault = AtlasVault()
    if not vault.verify_integrity():
        raise RuntimeError("Vault integrity check FAILED — refusing to start server")
    engine = ScoringEngine(vault)
    logger.info("=" * 70)
    logger.info("Server ready. Awaiting customer IDATs.")
    logger.info("=" * 70)
    return engine


if __name__ == "__main__":
    # Demo: at startup, verify vault integrity and load every reference matrix
    engine = server_startup()
    print()
    print("Atlases loaded:")
    for k, v in engine.vault.atlases.items():
        if isinstance(v, dict):
            print(f"  {k}: {len(v)} sub-matrices ({list(v.keys())[:3]}...)")
        elif isinstance(v, pd.DataFrame):
            print(f"  {k}: {v.shape[0]} markers × {v.shape[1]} cell types")
    print()
    print(f"H_min anchors: {len(H_MIN_ANCHORS)} architecture classes")
    print(f"Tier thresholds: {list(TIER_THRESHOLDS.keys())}")
