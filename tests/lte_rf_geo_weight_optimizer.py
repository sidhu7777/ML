from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import RepeatedKFold

from tests.lte_rf_debug_lab import DEFAULT_PROJECT_ID, _metric_bundle


OUTPUT_ROOT = Path("tests/output")

CLUTTER_FEATURE = "clutter_class"
NUMERIC_FEATURES = [
    "morphology_cluster",
    "building_area_ratio",
    "building_count",
    "road_length_m",
    "green_ratio",
    "water_ratio",
    "avg_building_area_m2",
    "site_count_250m",
    "site_count_500m",
    "serving_distance_m",
    "nearest_site_distance_m",
    "azimuth_delta_deg",
    "mean_nearest3_site_distance_m",
    "best_interferer_distance_m",
    "best_interferer_azimuth_delta_deg",
    "serving_proxy_rsrp_dbm",
    "best_interferer_proxy_rsrp_dbm",
    "serving_proxy_rsrp_phys_dbm",
    "best_interferer_proxy_phys_dbm",
    "interference_gap_db",
    "interference_ratio_linear",
    "interference_sum_proxy_dbm",
    "sinr_proxy_db",
    "rsrq_proxy_db",
    "effective_tx_height_m",
    "los_blocker_count",
    "los_blocked_length_m",
    "los_blocked_ratio",
    "mean_blocker_height_m",
    "max_blocker_height_m",
    "nlos_flag",
    "diffraction_proxy_db",
    "terrain_elevation_m",
    "terrain_slope_deg",
    "proxy_site_elevation_m",
    "terrain_relief_to_site_m",
]

CURRENT_WEIGHTS = {
    "clutter_Dense Urban": -4.5,
    "clutter_Urban": -2.5,
    "clutter_Suburban": -1.0,
    "clutter_Vegetation": -1.8,
    "clutter_Water": 1.0,
    "clutter_Rural/Open": 0.8,
    "morphology_cluster": -0.35,
    "building_area_ratio": -9.0,
    "building_count": -0.08,
    "road_length_m": -0.003,
    "green_ratio": -2.0,
    "water_ratio": 1.2,
    "avg_building_area_m2": -0.0008,
    "site_count_250m": 0.15,
    "site_count_500m": 0.08,
    "serving_distance_m": -0.0035,
    "nearest_site_distance_m": -0.0015,
    "azimuth_delta_deg": -0.018,
    "mean_nearest3_site_distance_m": 0.0008,
    "dense_urban_far_base": -2.8,
    "dense_urban_far_slope": -0.004,
    "urban_off_axis_slope": -0.015,
    "far_serving_off_axis_base": -1.2,
    "far_serving_distance_slope": -0.004,
    "far_serving_azimuth_slope": -0.010,
    "high_building_far_base": -1.1,
    "high_building_area_slope": -0.0012,
    "high_building_distance_slope": -0.0030,
    "vegetation_far_base": -0.8,
    "vegetation_green_slope": -2.2,
    "water_open_base": 0.9,
    "water_open_distance_slope": 0.0015,
    "dense_site_base": 0.7,
    "dense_site_count_slope": 0.06,
    "cluster_dense_urban_base": -1.4,
    "cluster_dense_urban_slope": -0.35,
    "nlos_flag": -2.4,
    "los_blocker_count": -0.9,
    "los_blocked_ratio": -5.5,
    "max_blocker_height_m": -0.05,
    "diffraction_proxy_db": -0.55,
    "terrain_slope_deg": -0.08,
    "terrain_relief_to_site_m": -0.028,
    "interference_gap_penalty_slope": -0.55,
    "interference_gap_bonus_slope": 0.10,
    "interference_ratio_linear": -1.6,
    "rsrp_phys_weight": 0.28,
    "rsrp_geo_weight": 0.55,
    "rsrq_phys_weight": 0.24,
    "rsrq_geo_weight": 0.18,
    "rsrq_geo_fallback_weight": 0.22,
    "sinr_phys_weight": 0.32,
    "sinr_geo_weight": 0.24,
    "sinr_geo_fallback_weight": 0.35,
}

SEARCH_SPACE = {
    "clutter_Dense Urban": (-8.0, -1.0),
    "clutter_Urban": (-5.0, 0.0),
    "clutter_Suburban": (-3.0, 1.5),
    "clutter_Vegetation": (-4.0, 0.0),
    "clutter_Water": (-1.0, 2.5),
    "clutter_Rural/Open": (-1.0, 2.0),
    "morphology_cluster": (-1.2, 0.8),
    "building_area_ratio": (-16.0, 2.0),
    "building_count": (-0.25, 0.1),
    "road_length_m": (-0.01, 0.002),
    "green_ratio": (-4.0, 1.0),
    "water_ratio": (-1.0, 3.5),
    "avg_building_area_m2": (-0.003, 0.0005),
    "site_count_250m": (-0.2, 0.4),
    "site_count_500m": (-0.15, 0.25),
    "serving_distance_m": (-0.008, 0.001),
    "nearest_site_distance_m": (-0.004, 0.001),
    "azimuth_delta_deg": (-0.06, 0.005),
    "mean_nearest3_site_distance_m": (-0.001, 0.002),
    "dense_urban_far_base": (-5.0, 0.0),
    "dense_urban_far_slope": (-0.01, 0.002),
    "urban_off_axis_slope": (-0.04, 0.002),
    "far_serving_off_axis_base": (-3.5, 0.0),
    "far_serving_distance_slope": (-0.01, 0.001),
    "far_serving_azimuth_slope": (-0.03, 0.001),
    "high_building_far_base": (-3.0, 0.0),
    "high_building_area_slope": (-0.004, 0.0005),
    "high_building_distance_slope": (-0.01, 0.0005),
    "vegetation_far_base": (-2.0, 0.5),
    "vegetation_green_slope": (-4.0, 0.5),
    "water_open_base": (-0.5, 2.5),
    "water_open_distance_slope": (-0.002, 0.005),
    "dense_site_base": (-0.5, 2.5),
    "dense_site_count_slope": (-0.05, 0.20),
    "cluster_dense_urban_base": (-3.0, 0.0),
    "cluster_dense_urban_slope": (-1.0, 0.2),
    "nlos_flag": (-5.0, 0.0),
    "los_blocker_count": (-2.0, 0.1),
    "los_blocked_ratio": (-10.0, 1.0),
    "max_blocker_height_m": (-0.15, 0.02),
    "diffraction_proxy_db": (-1.5, 0.1),
    "terrain_slope_deg": (-0.30, 0.02),
    "terrain_relief_to_site_m": (-0.08, 0.01),
    "interference_gap_penalty_slope": (-1.5, -0.05),
    "interference_gap_bonus_slope": (-0.1, 0.4),
    "interference_ratio_linear": (-4.0, 0.1),
    "rsrp_phys_weight": (0.0, 0.7),
    "rsrp_geo_weight": (0.0, 1.0),
    "rsrq_phys_weight": (0.0, 0.7),
    "rsrq_geo_weight": (0.0, 0.8),
    "rsrq_geo_fallback_weight": (0.0, 0.8),
    "sinr_phys_weight": (0.0, 0.8),
    "sinr_geo_weight": (0.0, 0.8),
    "sinr_geo_fallback_weight": (0.0, 0.9),
}


@dataclass
class OptimizationConfig:
    project_id: int = DEFAULT_PROJECT_ID
    run_dir: Path | None = None
    output_root: Path = OUTPUT_ROOT
    iterations: int = 40
    seed: int = 42
    holdout_fraction: float = 0.2
    cv_splits: int = 5
    cv_repeats: int = 3
    regularization_lambda: float = 0.05
    patience: int = 12
    min_improvement: float = 0.002
    warmup_random: int = 12
    candidate_pool_size: int = 256
    search_method: str = "bayes"
    top_k: int = 5


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _list_runs(project_id: int, output_root: Path) -> List[Path]:
    root = output_root / f"project_{project_id}"
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _resolve_run_dir(config: OptimizationConfig) -> Path:
    if config.run_dir is not None:
        return config.run_dir
    runs = _list_runs(config.project_id, config.output_root)
    if not runs:
        raise FileNotFoundError(f"No saved runs found for project_id={config.project_id}")
    return runs[0]


def _load_summary(run_dir: Path) -> Dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _prepare_optimizer_frame(run_dir: Path) -> pd.DataFrame:
    dt_eval = pd.read_csv(run_dir / "rf_accuracy_points.csv")
    grid_features = pd.read_csv(run_dir / "analysis_grid_features.csv")
    feature_cols = ["grid_id", CLUTTER_FEATURE] + [col for col in NUMERIC_FEATURES if col in grid_features.columns]
    frame = dt_eval.merge(grid_features[feature_cols], on="grid_id", how="left", suffixes=("", "_grid"))
    for col in NUMERIC_FEATURES:
        frame[col] = _safe_numeric(frame, col)
    if CLUTTER_FEATURE not in frame.columns:
        frame[CLUTTER_FEATURE] = "missing"
    frame[CLUTTER_FEATURE] = frame[CLUTTER_FEATURE].fillna("missing").astype(str)
    required_cols = ["RSRP_meas", "RSRQ_meas", "SINR_meas", "pred_rsrp", "pred_rsrq", "pred_sinr"]
    frame = frame.dropna(subset=required_cols).copy()
    if frame.empty:
        raise ValueError("Optimizer input frame is empty after merging DT evaluation with grid features.")
    return frame.reset_index(drop=True)


def _parameter_names() -> List[str]:
    return list(SEARCH_SPACE.keys())


def _weights_to_vector(weights: Dict[str, float]) -> np.ndarray:
    return np.array([float(weights[name]) for name in _parameter_names()], dtype=float)


def _vector_to_weights(vector: np.ndarray) -> Dict[str, float]:
    weights = dict(CURRENT_WEIGHTS)
    for idx, name in enumerate(_parameter_names()):
        low, high = SEARCH_SPACE[name]
        weights[name] = float(np.clip(vector[idx], low, high))
    return weights


def _normalized_l2_penalty(weights: Dict[str, float]) -> float:
    parts = []
    for name in _parameter_names():
        low, high = SEARCH_SPACE[name]
        width = max(high - low, 1e-9)
        parts.append(((float(weights[name]) - float(CURRENT_WEIGHTS[name])) / width) ** 2)
    return float(np.mean(parts)) if parts else 0.0


def _sample_weights(rng: np.random.Generator) -> Dict[str, float]:
    weights = dict(CURRENT_WEIGHTS)
    for key, (low, high) in SEARCH_SPACE.items():
        weights[key] = float(rng.uniform(low, high))
    return weights


def _apply_weighted_adjustment(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    clutter_penalty = out[CLUTTER_FEATURE].map(
        {
            "Dense Urban": weights["clutter_Dense Urban"],
            "Urban": weights["clutter_Urban"],
            "Suburban": weights["clutter_Suburban"],
            "Vegetation": weights["clutter_Vegetation"],
            "Water": weights["clutter_Water"],
            "Rural/Open": weights["clutter_Rural/Open"],
        }
    ).fillna(0.0)

    cluster_center = float(out["morphology_cluster"].mean()) if len(out) else 0.0
    cluster_offset = (out["morphology_cluster"] - cluster_center) * float(weights["morphology_cluster"])
    geo_offset = clutter_penalty + cluster_offset
    geo_offset = geo_offset + (_safe_numeric(out, "building_area_ratio").clip(0, 0.8) * float(weights["building_area_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(out, "building_count").clip(0, 30) * float(weights["building_count"]))
    geo_offset = geo_offset + (_safe_numeric(out, "road_length_m").clip(0, 400) * float(weights["road_length_m"]))
    geo_offset = geo_offset + (_safe_numeric(out, "green_ratio").clip(0, 1.0) * float(weights["green_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(out, "water_ratio").clip(0, 1.0) * float(weights["water_ratio"]))
    geo_offset = geo_offset + (_safe_numeric(out, "avg_building_area_m2").clip(0, 3000) * float(weights["avg_building_area_m2"]))
    geo_offset = geo_offset + (_safe_numeric(out, "site_count_250m").clip(0, 12) * float(weights["site_count_250m"]))
    geo_offset = geo_offset + (_safe_numeric(out, "site_count_500m").clip(0, 25) * float(weights["site_count_500m"]))
    geo_offset = geo_offset + (_safe_numeric(out, "serving_distance_m").clip(0, 1200) * float(weights["serving_distance_m"]))
    geo_offset = geo_offset + (_safe_numeric(out, "nearest_site_distance_m").clip(0, 1000) * float(weights["nearest_site_distance_m"]))
    geo_offset = geo_offset + (((_safe_numeric(out, "azimuth_delta_deg").clip(0, 180) / 10.0) ** 1.2) * float(weights["azimuth_delta_deg"]))
    geo_offset = geo_offset + (_safe_numeric(out, "mean_nearest3_site_distance_m").clip(0, 1500) * float(weights["mean_nearest3_site_distance_m"]))

    clutter_series = out[CLUTTER_FEATURE].astype(str)
    nearest_site = _safe_numeric(out, "nearest_site_distance_m")
    serving_distance = _safe_numeric(out, "serving_distance_m")
    azimuth_delta = _safe_numeric(out, "azimuth_delta_deg")
    avg_building_area = _safe_numeric(out, "avg_building_area_m2")
    green_ratio = _safe_numeric(out, "green_ratio")
    site_count_250m = _safe_numeric(out, "site_count_250m")
    morphology_cluster = _safe_numeric(out, "morphology_cluster")

    dense_urban_far_penalty = np.where(
        (clutter_series == "Dense Urban") & (nearest_site > 180.0),
        float(weights["dense_urban_far_base"])
        + (float(weights["dense_urban_far_slope"]) * (nearest_site.clip(180.0, 700.0) - 180.0)),
        0.0,
    )
    urban_off_axis_penalty = np.where(
        azimuth_delta > 45.0,
        float(weights["urban_off_axis_slope"]) * (azimuth_delta.clip(45.0, 180.0) - 45.0),
        0.0,
    )
    far_serving_off_axis_penalty = np.where(
        (serving_distance > 250.0) & (azimuth_delta > 35.0),
        float(weights["far_serving_off_axis_base"])
        + (float(weights["far_serving_distance_slope"]) * (serving_distance.clip(250.0, 1200.0) - 250.0))
        + (float(weights["far_serving_azimuth_slope"]) * (azimuth_delta.clip(35.0, 180.0) - 35.0)),
        0.0,
    )
    high_building_far_penalty = np.where(
        (avg_building_area > 250.0) & (nearest_site > 160.0),
        float(weights["high_building_far_base"])
        + (float(weights["high_building_area_slope"]) * (avg_building_area.clip(250.0, 3000.0) - 250.0))
        + (float(weights["high_building_distance_slope"]) * (nearest_site.clip(160.0, 1000.0) - 160.0)),
        0.0,
    )
    vegetation_far_penalty = np.where(
        (clutter_series == "Vegetation") & (serving_distance > 220.0),
        float(weights["vegetation_far_base"])
        + (float(weights["vegetation_green_slope"]) * green_ratio.clip(0.2, 1.0)),
        0.0,
    )
    water_open_bonus = np.where(
        clutter_series.isin(["Water", "Rural/Open"]) & (azimuth_delta < 20.0) & (nearest_site < 220.0),
        float(weights["water_open_base"])
        + (float(weights["water_open_distance_slope"]) * (220.0 - nearest_site.clip(0.0, 220.0))),
        0.0,
    )
    dense_site_bonus = np.where(
        (site_count_250m >= 4.0) & (nearest_site < 120.0),
        float(weights["dense_site_base"])
        + (float(weights["dense_site_count_slope"]) * site_count_250m.clip(4.0, 12.0)),
        0.0,
    )
    cluster_dense_urban_penalty = np.where(
        (morphology_cluster >= (cluster_center + 1.0)) & (clutter_series == "Dense Urban"),
        float(weights["cluster_dense_urban_base"])
        + (float(weights["cluster_dense_urban_slope"]) * (morphology_cluster - cluster_center).clip(lower=0.0, upper=4.0)),
        0.0,
    )
    nlos_penalty = float(weights["nlos_flag"]) * _safe_numeric(out, "nlos_flag").clip(0, 1)
    blocker_penalty = (
        float(weights["los_blocker_count"]) * _safe_numeric(out, "los_blocker_count").clip(0, 10)
        + float(weights["los_blocked_ratio"]) * _safe_numeric(out, "los_blocked_ratio").clip(0, 1.0)
        + float(weights["max_blocker_height_m"]) * _safe_numeric(out, "max_blocker_height_m").clip(0, 80.0)
    )
    diffraction_penalty = float(weights["diffraction_proxy_db"]) * _safe_numeric(out, "diffraction_proxy_db").clip(0, 25.0)
    terrain_penalty = (
        float(weights["terrain_slope_deg"]) * _safe_numeric(out, "terrain_slope_deg").clip(0, 35.0)
        + float(weights["terrain_relief_to_site_m"]) * _safe_numeric(out, "terrain_relief_to_site_m").clip(lower=0.0, upper=180.0)
    )
    interference_gap = _safe_numeric(out, "interference_gap_db")
    interference_penalty = np.where(
        interference_gap < 6.0,
        float(weights["interference_gap_penalty_slope"]) * (6.0 - interference_gap.clip(-15.0, 6.0)),
        float(weights["interference_gap_bonus_slope"]) * (interference_gap.clip(6.0, 18.0) - 6.0),
    )
    interference_ratio_penalty = (
        float(weights["interference_ratio_linear"]) * _safe_numeric(out, "interference_ratio_linear").clip(0.0, 2.5)
    )

    geo_offset = (
        geo_offset
        + pd.Series(dense_urban_far_penalty, index=out.index, dtype=float)
        + pd.Series(urban_off_axis_penalty, index=out.index, dtype=float)
        + pd.Series(far_serving_off_axis_penalty, index=out.index, dtype=float)
        + pd.Series(high_building_far_penalty, index=out.index, dtype=float)
        + pd.Series(vegetation_far_penalty, index=out.index, dtype=float)
        + pd.Series(water_open_bonus, index=out.index, dtype=float)
        + pd.Series(dense_site_bonus, index=out.index, dtype=float)
        + pd.Series(cluster_dense_urban_penalty, index=out.index, dtype=float)
        + nlos_penalty
        + blocker_penalty
        + diffraction_penalty
        + terrain_penalty
        + pd.Series(interference_penalty, index=out.index, dtype=float)
        + interference_ratio_penalty
    )

    rsrp_base = pd.to_numeric(out["pred_rsrp"], errors="coerce")
    rsrq_base = pd.to_numeric(out["pred_rsrq"], errors="coerce")
    sinr_base = pd.to_numeric(out["pred_sinr"], errors="coerce")
    rsrp_phys = pd.to_numeric(out.get("serving_proxy_rsrp_phys_dbm"), errors="coerce")
    rsrq_phys = pd.to_numeric(out.get("rsrq_proxy_db"), errors="coerce")
    sinr_phys = pd.to_numeric(out.get("sinr_proxy_db"), errors="coerce")

    rsrp_base_weight = max(0.0, 1.0 - float(weights["rsrp_phys_weight"]))
    rsrq_base_weight = max(0.0, 1.0 - float(weights["rsrq_phys_weight"]))
    sinr_base_weight = max(0.0, 1.0 - float(weights["sinr_phys_weight"]))

    out["tuned_geo_offset"] = geo_offset
    out["pred_rsrp_tuned"] = rsrp_base.copy()
    has_rsrp_phys = rsrp_phys.notna()
    out.loc[has_rsrp_phys, "pred_rsrp_tuned"] = (
        (rsrp_base_weight * rsrp_base[has_rsrp_phys])
        + (float(weights["rsrp_phys_weight"]) * rsrp_phys[has_rsrp_phys])
        + (float(weights["rsrp_geo_weight"]) * geo_offset[has_rsrp_phys])
    )
    out.loc[~has_rsrp_phys, "pred_rsrp_tuned"] = rsrp_base[~has_rsrp_phys] + geo_offset[~has_rsrp_phys]

    out["pred_rsrq_tuned"] = rsrq_base.copy()
    has_rsrq_phys = rsrq_phys.notna()
    out.loc[has_rsrq_phys, "pred_rsrq_tuned"] = (
        (rsrq_base_weight * rsrq_base[has_rsrq_phys])
        + (float(weights["rsrq_phys_weight"]) * rsrq_phys[has_rsrq_phys])
        + (float(weights["rsrq_geo_weight"]) * geo_offset[has_rsrq_phys])
    )
    out.loc[~has_rsrq_phys, "pred_rsrq_tuned"] = rsrq_base[~has_rsrq_phys] + (
        geo_offset[~has_rsrq_phys] * float(weights["rsrq_geo_fallback_weight"])
    )

    out["pred_sinr_tuned"] = sinr_base.copy()
    has_sinr_phys = sinr_phys.notna()
    out.loc[has_sinr_phys, "pred_sinr_tuned"] = (
        (sinr_base_weight * sinr_base[has_sinr_phys])
        + (float(weights["sinr_phys_weight"]) * sinr_phys[has_sinr_phys])
        + (float(weights["sinr_geo_weight"]) * geo_offset[has_sinr_phys])
    )
    out.loc[~has_sinr_phys, "pred_sinr_tuned"] = sinr_base[~has_sinr_phys] + (
        geo_offset[~has_sinr_phys] * float(weights["sinr_geo_fallback_weight"])
    )

    out["pred_rsrp_tuned"] = out["pred_rsrp_tuned"].clip(-140, -44)
    out["pred_rsrq_tuned"] = out["pred_rsrq_tuned"].clip(-20, -3)
    out["pred_sinr_tuned"] = out["pred_sinr_tuned"].clip(-10, 30)
    return out


def _evaluate_frame(df: pd.DataFrame, pred_suffix: str) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    specs = [
        ("RSRP_meas", f"pred_rsrp{pred_suffix}"),
        ("RSRQ_meas", f"pred_rsrq{pred_suffix}"),
        ("SINR_meas", f"pred_sinr{pred_suffix}"),
    ]
    for meas_col, pred_col in specs:
        valid = df.dropna(subset=[meas_col, pred_col])
        if not valid.empty:
            metrics[meas_col] = _metric_bundle(valid[meas_col], valid[pred_col], metric_key=meas_col)
    return metrics


def _score_metrics(metrics: Dict[str, Dict[str, float]]) -> float:
    norms = {"RSRP_meas": 10.0, "RSRQ_meas": 3.0, "SINR_meas": 6.0}
    parts: List[float] = []
    for metric_name, values in metrics.items():
        mae = float(values.get("mae", 1e9))
        within_key = f"within_{str(norms[metric_name]).replace('.', '_')}"
        within_score = float(values.get(within_key, 0.0))
        parts.append((mae / norms[metric_name]) - within_score)
    return float(np.mean(parts)) if parts else float("inf")


def _score_candidate_cv(
    train_df: pd.DataFrame,
    weights: Dict[str, float],
    cv_splits: int,
    cv_repeats: int,
    regularization_lambda: float,
    seed: int,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    rkf = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed)
    fold_scores: List[float] = []
    metric_rows: Dict[str, List[Dict[str, float]]] = {"RSRP_meas": [], "RSRQ_meas": [], "SINR_meas": []}

    for _, valid_idx in rkf.split(train_df):
        fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)
        tuned_valid = _apply_weighted_adjustment(fold_valid, weights)
        valid_metrics = _evaluate_frame(tuned_valid, "_tuned")
        fold_scores.append(_score_metrics(valid_metrics))
        for metric_name in metric_rows:
            if metric_name in valid_metrics:
                metric_rows[metric_name].append(valid_metrics[metric_name])

    averaged_metrics: Dict[str, Dict[str, float]] = {}
    for metric_name, rows in metric_rows.items():
        if not rows:
            continue
        averaged_metrics[metric_name] = {
            key: round(float(np.mean([row[key] for row in rows if row.get(key) is not None])), 4)
            for key in rows[0].keys()
            if any(row.get(key) is not None for row in rows)
        }

    cv_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
    penalty = regularization_lambda * _normalized_l2_penalty(weights)
    return cv_score + penalty, averaged_metrics


def _propose_bayesian_candidate(
    tried_vectors: List[np.ndarray],
    tried_scores: List[float],
    rng: np.random.Generator,
    candidate_pool_size: int,
) -> Dict[str, float]:
    if len(tried_vectors) < 5:
        return _sample_weights(rng)

    X = np.vstack(tried_vectors)
    y = np.array(tried_scores, dtype=float)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
        normalize_y=True,
        random_state=0,
    )
    gp.fit(X, y)

    pool = np.vstack([_weights_to_vector(_sample_weights(rng)) for _ in range(candidate_pool_size)])
    mean_pred, std_pred = gp.predict(pool, return_std=True)
    acquisition = mean_pred - (0.35 * std_pred)
    return _vector_to_weights(pool[int(np.argmin(acquisition))])


def _format_weight_table(weights: Dict[str, float]) -> pd.DataFrame:
    rows = [{"parameter": key, "value": value} for key, value in weights.items() if key != "cluster_center_mode"]
    return pd.DataFrame(rows).sort_values("parameter").reset_index(drop=True)


def run_geo_weight_optimizer(config: OptimizationConfig) -> Path:
    run_dir = _resolve_run_dir(config)
    summary = _load_summary(run_dir)
    optimizer_dir = run_dir / f"optimizer_{_timestamp()}"
    optimizer_dir.mkdir(parents=True, exist_ok=True)

    frame = _prepare_optimizer_frame(run_dir)
    rng = np.random.default_rng(config.seed)
    shuffled_idx = rng.permutation(len(frame))
    holdout_size = max(1, int(len(frame) * config.holdout_fraction))
    holdout_idx = shuffled_idx[:holdout_size]
    train_idx = shuffled_idx[holdout_size:]
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    holdout_df = frame.iloc[holdout_idx].reset_index(drop=True)
    if train_df.empty or holdout_df.empty:
        raise ValueError("Optimizer split failed: train or holdout split is empty.")

    baseline_train = _evaluate_frame(train_df, "")
    baseline_holdout = _evaluate_frame(holdout_df, "")
    baseline_full = _evaluate_frame(frame, "")

    current_train = _evaluate_frame(_apply_weighted_adjustment(train_df, CURRENT_WEIGHTS), "_tuned")
    current_holdout = _evaluate_frame(_apply_weighted_adjustment(holdout_df, CURRENT_WEIGHTS), "_tuned")
    current_full = _evaluate_frame(_apply_weighted_adjustment(frame, CURRENT_WEIGHTS), "_tuned")
    current_cv_score, current_cv_metrics = _score_candidate_cv(
        train_df,
        CURRENT_WEIGHTS,
        cv_splits=config.cv_splits,
        cv_repeats=config.cv_repeats,
        regularization_lambda=config.regularization_lambda,
        seed=config.seed,
    )

    leaderboard: List[Dict[str, float]] = []
    tried_vectors: List[np.ndarray] = [_weights_to_vector(CURRENT_WEIGHTS)]
    tried_scores: List[float] = [current_cv_score]
    best_weights = dict(CURRENT_WEIGHTS)
    best_cv_score = current_cv_score
    no_improvement_rounds = 0

    for idx in range(config.iterations):
        if idx < config.warmup_random or config.search_method == "random":
            weights = _sample_weights(rng)
            search_label = "random"
        else:
            weights = _propose_bayesian_candidate(tried_vectors, tried_scores, rng, config.candidate_pool_size)
            search_label = "bayes"

        tuned_train = _apply_weighted_adjustment(train_df, weights)
        tuned_holdout = _apply_weighted_adjustment(holdout_df, weights)
        train_metrics = _evaluate_frame(tuned_train, "_tuned")
        holdout_metrics = _evaluate_frame(tuned_holdout, "_tuned")
        cv_score, cv_metrics = _score_candidate_cv(
            train_df,
            weights,
            cv_splits=config.cv_splits,
            cv_repeats=config.cv_repeats,
            regularization_lambda=config.regularization_lambda,
            seed=config.seed + idx + 1,
        )
        holdout_score = _score_metrics(holdout_metrics)

        row = {
            "candidate": idx + 1,
            "search_method": search_label,
            "train_score": _score_metrics(train_metrics),
            "cv_score": cv_score,
            "holdout_score": holdout_score,
            "train_rsrp_mae": train_metrics.get("RSRP_meas", {}).get("mae"),
            "cv_rsrp_mae": cv_metrics.get("RSRP_meas", {}).get("mae"),
            "holdout_rsrp_mae": holdout_metrics.get("RSRP_meas", {}).get("mae"),
            "train_rsrq_mae": train_metrics.get("RSRQ_meas", {}).get("mae"),
            "cv_rsrq_mae": cv_metrics.get("RSRQ_meas", {}).get("mae"),
            "holdout_rsrq_mae": holdout_metrics.get("RSRQ_meas", {}).get("mae"),
            "train_sinr_mae": train_metrics.get("SINR_meas", {}).get("mae"),
            "cv_sinr_mae": cv_metrics.get("SINR_meas", {}).get("mae"),
            "holdout_sinr_mae": holdout_metrics.get("SINR_meas", {}).get("mae"),
            "regularization_penalty": round(config.regularization_lambda * _normalized_l2_penalty(weights), 6),
        }
        for key, value in weights.items():
            if key != "cluster_center_mode":
                row[key] = value
        leaderboard.append(row)

        tried_vectors.append(_weights_to_vector(weights))
        tried_scores.append(cv_score)

        if cv_score < (best_cv_score - config.min_improvement):
            best_cv_score = cv_score
            best_weights = dict(weights)
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1
            if no_improvement_rounds >= config.patience:
                print(f"[OPT] Early stop at candidate {idx + 1} due to no CV improvement.")
                break

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(["cv_score", "holdout_score", "train_score"]).reset_index(drop=True)
    leaderboard_df.to_csv(optimizer_dir / "leaderboard.csv", index=False)

    best_train = _evaluate_frame(_apply_weighted_adjustment(train_df, best_weights), "_tuned")
    best_holdout = _evaluate_frame(_apply_weighted_adjustment(holdout_df, best_weights), "_tuned")
    best_full_df = _apply_weighted_adjustment(frame, best_weights)
    best_full = _evaluate_frame(best_full_df, "_tuned")
    best_cv_score_final, best_cv_metrics = _score_candidate_cv(
        train_df,
        best_weights,
        cv_splits=config.cv_splits,
        cv_repeats=config.cv_repeats,
        regularization_lambda=config.regularization_lambda,
        seed=config.seed,
    )

    export_cols = [
        "session_id",
        "lat",
        "lon",
        "grid_id",
        "Node_Cell_ID",
        "RSRP_meas",
        "RSRQ_meas",
        "SINR_meas",
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_tuned",
        "pred_rsrq_tuned",
        "pred_sinr_tuned",
        "tuned_geo_offset",
        CLUTTER_FEATURE,
    ] + [col for col in NUMERIC_FEATURES if col in best_full_df.columns]
    best_full_df[[col for col in export_cols if col in best_full_df.columns]].to_csv(optimizer_dir / "tuned_eval_points.csv", index=False)
    _format_weight_table(best_weights).to_csv(optimizer_dir / "best_weights.csv", index=False)

    result_summary = {
        "source_run_dir": str(run_dir),
        "source_run_name": run_dir.name,
        "optimizer_dir": str(optimizer_dir),
        "project_id": config.project_id,
        "iterations_requested": config.iterations,
        "iterations_completed": len(leaderboard_df),
        "seed": config.seed,
        "holdout_fraction": config.holdout_fraction,
        "cv_splits": config.cv_splits,
        "cv_repeats": config.cv_repeats,
        "regularization_lambda": config.regularization_lambda,
        "patience": config.patience,
        "min_improvement": config.min_improvement,
        "warmup_random": config.warmup_random,
        "candidate_pool_size": config.candidate_pool_size,
        "search_method": config.search_method,
        "rows": {"full": len(frame), "train": len(train_df), "holdout": len(holdout_df)},
        "baseline_metrics": {"train": baseline_train, "holdout": baseline_holdout, "full": baseline_full},
        "current_fixed_weight_metrics": {
            "train": current_train,
            "cv": current_cv_metrics,
            "cv_score": current_cv_score,
            "holdout": current_holdout,
            "holdout_score": _score_metrics(current_holdout),
            "full": current_full,
        },
        "best_tuned_metrics": {
            "train": best_train,
            "cv": best_cv_metrics,
            "cv_score": best_cv_score_final,
            "holdout": best_holdout,
            "holdout_score": _score_metrics(best_holdout),
            "full": best_full,
        },
        "best_weights": {key: value for key, value in best_weights.items() if key != "cluster_center_mode"},
        "top_candidates": leaderboard_df.head(config.top_k).to_dict(orient="records"),
        "source_summary_metrics": summary.get("full_metrics", {}),
    }
    (optimizer_dir / "summary.json").write_text(json.dumps(result_summary, indent=2, default=str), encoding="utf-8")

    print(f"[OPT] Source run: {run_dir}")
    print(f"[OPT] Optimizer output: {optimizer_dir}")
    print(f"[OPT] Baseline RSRP MAE(full)={baseline_full.get('RSRP_meas', {}).get('mae')}")
    print(f"[OPT] Current Geo RSRP MAE(holdout)={current_holdout.get('RSRP_meas', {}).get('mae')}")
    print(f"[OPT] Tuned Geo RSRP MAE(holdout)={best_holdout.get('RSRP_meas', {}).get('mae')}")
    print(f"[OPT] Current Geo CV score={round(current_cv_score, 6)}")
    print(f"[OPT] Tuned Geo CV score={round(best_cv_score_final, 6)}")
    return optimizer_dir


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test-only optimizer for LTE RF geo-adjustment weights")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--regularization-lambda", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-improvement", type=float, default=0.002)
    parser.add_argument("--warmup-random", type=int, default=12)
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--search-method", type=str, choices=["random", "bayes"], default="bayes")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    config = OptimizationConfig(
        project_id=args.project_id,
        run_dir=args.run_dir,
        output_root=args.output_root,
        iterations=args.iterations,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        regularization_lambda=args.regularization_lambda,
        patience=args.patience,
        min_improvement=args.min_improvement,
        warmup_random=args.warmup_random,
        candidate_pool_size=args.candidate_pool_size,
        search_method=args.search_method,
        top_k=args.top_k,
    )
    optimizer_dir = run_geo_weight_optimizer(config)
    print(f"[OPT] Artifacts saved under {optimizer_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
