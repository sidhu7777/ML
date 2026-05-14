from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.lte_rf_debug_lab import DEFAULT_PROJECT_ID, DEFAULT_REGION, _write_json
from tools.lte_prediction_optimised import ml_engine as opt_ml


OUTPUT_ROOT = Path("tests/output")

MIN_BAD_SAMPLE_COUNT_FOR_ACTION = 25
MEDIUM_CONFIDENCE_BAD_SAMPLE_COUNT = 120
HIGH_CONFIDENCE_BAD_SAMPLE_COUNT = 250
VERY_HIGH_CONFIDENCE_BAD_SAMPLE_COUNT = 600

FAR_EDGE_MEAN_SERVING_DISTANCE_M = 175.0
FAR_EDGE_P90_SERVING_DISTANCE_M = 260.0
MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M = 130.0
MEDIUM_EDGE_P90_SERVING_DISTANCE_M = 220.0

HIGH_NLOS_SHARE_GATE = 0.7046681119665223
HIGH_LOS_BLOCKED_RATIO_GATE = 0.2139142952055874
HIGH_BUILDING_AREA_RATIO_GATE = 0.24481831460941722
MEDIUM_NLOS_SHARE_GATE = 0.4834199705913421
MEDIUM_LOS_BLOCKED_RATIO_GATE = 0.13621508948087993

DENSE_OVERLAP_NEAREST_SITE_M = 180.0
DENSE_OVERLAP_SITE_COUNT_250M = 2.0
SMALL_AZIMUTH_DELTA_DEG = 35.0

MIN_AZIMUTH_MISMATCH_DEG = 15.0
MAX_AZIMUTH_MISMATCH_DEG = 45.0
MAX_AZIMUTH_STEP_DEG = 10.0
MIN_BEARING_SAMPLE_COUNT = 30
MAX_BEARING_SPREAD_DEG = 40.0
AZIMUTH_NLOS_HARD_BLOCK_GATE = 0.85
BEARING_BIN_SIZE_DEG = 10.0
MIN_PEAK_SHARE_FOR_AZIMUTH = 0.27944004818646667
MIN_SAFE_ETILT_DEG = 2.0
MAX_SAFE_ETILT_DEG = 12.0
MAX_ETILT_INCREASE_PER_RUN_DEG = 2.0
MAX_ETILT_DECREASE_PER_RUN_DEG = 2.0
RELAXED_AZIMUTH_BAD_SAMPLE_COUNT = 183
RELAXED_AZIMUTH_PEAK_SHARE = 0.5721666812548345
RELAXED_AZIMUTH_MAX_SPREAD_DEG = 32.264493787350155
MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH = 1.2734413438634005
STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH = 2.542748090042536
MIN_CANDIDATE_SCORE_GAP = 4.654858443626446
SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO = 0.055289002855823374

COVERAGE_ETILT_ACTION_SCORE = 51.49084372010145
OVERLAP_ETILT_ACTION_SCORE = 47.31004233364023
AZIMUTH_ACTION_SCORE = 71.61142299538227
TX_POWER_ACTION_SCORE = 46.25093121890758


@dataclass
class TiltRecommendationTestConfig:
    project_id: int = DEFAULT_PROJECT_ID
    region: str = DEFAULT_REGION
    operator: Optional[str] = None
    rsrp_threshold: float = -105.0
    rsrq_threshold: float = -15.0
    sinr_threshold: float = 0.0
    output_root: Path = OUTPUT_ROOT


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _derive_antenna_cell_key(antenna_df: pd.DataFrame) -> pd.Series:
    node_cell_col = TILT_SRC._find_col(antenna_df, ["Node_Cell_ID", "node_cell_id"], required=False)
    if node_cell_col:
        node_cell = antenna_df[node_cell_col].map(TILT_SRC._norm_cell_id)
        if (node_cell != "").any():
            return node_cell

    nodeb_col = TILT_SRC._find_col(antenna_df, ["nodeb_id", "nodeb", "site_id"], required=False)
    cell_col = TILT_SRC._find_col(antenna_df, ["cell_id", "eci", "local_cell_id"], required=False)

    if nodeb_col and cell_col:
        nodeb = antenna_df[nodeb_col].map(TILT_SRC._norm_cell_id)
        cell = antenna_df[cell_col].map(TILT_SRC._norm_cell_id)
        combined = pd.Series(
            np.where((nodeb != "") & (cell != ""), nodeb + "_" + cell, ""),
            index=antenna_df.index,
        )
        if (combined != "").any():
            return combined
        if (cell != "").any():
            return cell

    if cell_col:
        return antenna_df[cell_col].map(TILT_SRC._norm_cell_id)

    return pd.Series("", index=antenna_df.index)


def _load_tilt_module():
    module_path = PROJECT_ROOT / "tools" / "lte_tilt_recommandation" / "etilt_optimizer_cd2.py"
    module_name = "tests._tilt_source_module"
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(module_path),
            "dummy_log.csv",
            "dummy_antenna.csv",
            "-105",
            "-15",
            "0",
        ]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load tilt source module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


TILT_SRC = _load_tilt_module()


def _fetch_baseline_log_df(project_id: int, region: str, operator: Optional[str]) -> pd.DataFrame:
    baseline_df = opt_ml.fetch_baseline(project_id, region=region).copy()
    if operator:
        operator_norm = str(operator).strip().lower()
        op_col = next((c for c in ["operator", "Operator"] if c in baseline_df.columns), None)
        if op_col:
            baseline_df = baseline_df.loc[
                baseline_df[op_col].astype(str).str.strip().str.lower() == operator_norm
            ].copy()

    if baseline_df.empty:
        raise FileNotFoundError(f"No baseline rows found for project_id={project_id} region={region}")

    baseline_df["node_b_id"] = baseline_df["Node_Cell_ID"].astype(str).str.split("_").str[0]
    baseline_df["nodeb_id"] = baseline_df["Node_Cell_ID"].astype(str).str.split("_").str[0]
    baseline_df["cell_id"] = baseline_df["Node_Cell_ID"].astype(str).str.split("_").str[-1]
    baseline_df["local_cell_id"] = baseline_df["Node_Cell_ID"].astype(str).str.split("_").str[-1]
    if "operator" not in baseline_df.columns:
        baseline_df["operator"] = operator or "Unknown"
    baseline_df["pred_rsrp"] = pd.to_numeric(baseline_df["pred_rsrp"], errors="coerce")
    baseline_df["pred_rsrq"] = pd.to_numeric(baseline_df["pred_rsrq"], errors="coerce")
    baseline_df["pred_sinr"] = pd.to_numeric(baseline_df["pred_sinr"], errors="coerce")
    return baseline_df


def _fetch_antenna_df(project_id: int, region: str, operator: Optional[str]) -> pd.DataFrame:
    site_df = opt_ml.fetch_site_data(project_id, region=region, operator=operator).copy()
    if site_df.empty:
        raise FileNotFoundError(f"No site rows found for project_id={project_id} region={region}")
    return site_df


def _enrich_log_with_antenna_context(log_df: pd.DataFrame, antenna_df: pd.DataFrame) -> pd.DataFrame:
    out = log_df.copy()
    ant = antenna_df.copy()
    ant["Node_Cell_ID"] = ant["Node_Cell_ID"].astype(str).str.strip()
    merge_cols = [
        col for col in [
            "Node_Cell_ID",
            "Technology",
            "lat",
            "lon",
            "azimuth",
            "electrical_tilt",
            "mechanical_tilt",
            "tx_power",
            "antenna_height",
            "dashboard_site_id",
        ] if col in ant.columns
    ]
    ant = ant[merge_cols].drop_duplicates(subset=["Node_Cell_ID"], keep="last")
    out = out.merge(ant, on="Node_Cell_ID", how="left", suffixes=("", "_site"))
    if "Technology" not in out.columns:
        out["Technology"] = "4G"
    else:
        out["Technology"] = out["Technology"].fillna("4G")
    return out


def _fetch_geo_df(project_id: int, region: str, operator: Optional[str], antenna_df: pd.DataFrame) -> pd.DataFrame:
    affected_cells = sorted(antenna_df["Node_Cell_ID"].astype(str).unique().tolist())
    geo_df = opt_ml.fetch_geo_features(project_id, region=region, affected_cells=affected_cells).copy()
    if geo_df.empty:
        return geo_df
    if operator:
        site_cells = antenna_df[["Node_Cell_ID"]].copy()
        site_cells["Node_Cell_ID"] = site_cells["Node_Cell_ID"].astype(str)
        geo_df = geo_df.merge(site_cells.drop_duplicates(), on="Node_Cell_ID", how="inner")
    return geo_df


def _compute_dominant_bearing_summary(log_df: pd.DataFrame, antenna_df: pd.DataFrame) -> pd.DataFrame:
    log_work = TILT_SRC._normalize_columns(log_df).copy()
    ant_work = TILT_SRC._normalize_columns(antenna_df).copy()

    log_cell_col = TILT_SRC._find_col(log_work, ["Node_Cell_ID", "node_cell_id"], required=False)
    ant_cell_col = TILT_SRC._find_col(ant_work, ["Node_Cell_ID", "node_cell_id"], required=False)
    log_work["Cell ID"] = (
        log_work[log_cell_col].astype(str).map(TILT_SRC._norm_cell_id)
        if log_cell_col else TILT_SRC._get_cell_key_from_log(log_work)
    )
    ant_work["Cell ID"] = (
        ant_work[ant_cell_col].astype(str).map(TILT_SRC._norm_cell_id)
        if ant_cell_col else _derive_antenna_cell_key(ant_work)
    )

    lat_col = TILT_SRC._find_col(log_work, ["lat", "latitude"], required=False)
    lon_col = TILT_SRC._find_col(log_work, ["lon", "longitude"], required=False)
    rsrp_col = TILT_SRC._find_col(log_work, ["rsrp", "pred_rsrp", "csi_rsrp"], required=False)
    rsrq_col = TILT_SRC._find_col(log_work, ["rsrq", "pred_rsrq", "csi_rsrq"], required=False)
    sinr_col = TILT_SRC._find_col(log_work, ["sinr", "pred_sinr", "csi_sinr"], required=False)

    ant_lat_col = TILT_SRC._find_col(ant_work, ["latitude", "lat"], required=False)
    ant_lon_col = TILT_SRC._find_col(ant_work, ["longitude", "lon"], required=False)
    az_col = TILT_SRC._find_col(ant_work, ["azimuth", "azi"], required=False)

    log_work["RSRP_eval"] = pd.to_numeric(log_work[rsrp_col], errors="coerce") if rsrp_col else np.nan
    log_work["RSRQ_eval"] = pd.to_numeric(log_work[rsrq_col], errors="coerce") if rsrq_col else np.nan
    log_work["SINR_eval"] = pd.to_numeric(log_work[sinr_col], errors="coerce") if sinr_col else np.nan
    log_work["Lat_eval"] = pd.to_numeric(log_work[lat_col], errors="coerce") if lat_col else np.nan
    log_work["Lon_eval"] = pd.to_numeric(log_work[lon_col], errors="coerce") if lon_col else np.nan
    log_work["severity_score"] = (
        (float(TILT_SRC.RSRP_THRESH) - log_work["RSRP_eval"]).clip(lower=0).fillna(0)
        + (float(TILT_SRC.RSRQ_THRESH) - log_work["RSRQ_eval"]).clip(lower=0).fillna(0)
        + (float(TILT_SRC.SINR_THRESH) - log_work["SINR_eval"]).clip(lower=0).fillna(0)
    )

    ant_map = (
        ant_work.drop_duplicates(subset=["Cell ID"])
        .assign(
            Azimuth_cfg=lambda d: pd.to_numeric(d[az_col], errors="coerce").map(TILT_SRC._normalize_azimuth)
            if az_col else np.nan,
            AntLat=lambda d: pd.to_numeric(d[ant_lat_col], errors="coerce") if ant_lat_col else np.nan,
            AntLon=lambda d: pd.to_numeric(d[ant_lon_col], errors="coerce") if ant_lon_col else np.nan,
        )
        .set_index("Cell ID")[["Azimuth_cfg", "AntLat", "AntLon"]]
        .to_dict("index")
    )

    rows: List[Dict[str, object]] = []
    for cell_id, group in log_work.groupby("Cell ID", dropna=False):
        if not cell_id or cell_id not in ant_map:
            continue
        ant = ant_map[cell_id]
        if pd.isna(ant.get("AntLat")) or pd.isna(ant.get("AntLon")):
            continue
        g2 = group.dropna(subset=["RSRP_eval", "Lat_eval", "Lon_eval"]).copy()
        if g2.empty:
            continue

        bad_dir = g2[g2["severity_score"] > 0].copy()
        if bad_dir.empty:
            continue
        if len(bad_dir) >= MIN_BEARING_SAMPLE_COUNT:
            use_dir = bad_dir
        else:
            use_dir = g2.nlargest(min(len(g2), MIN_BEARING_SAMPLE_COUNT), "severity_score").copy()
            use_dir = use_dir[use_dir["severity_score"] > 0]
            if use_dir.empty:
                continue

        use_dir["bearing"] = use_dir.apply(
            lambda r: TILT_SRC._bearing_deg(
                ant["AntLat"],
                ant["AntLon"],
                r["Lat_eval"],
                r["Lon_eval"],
            ),
            axis=1,
        )
        peak_summary = _directional_peak_summary(use_dir["bearing"], use_dir["severity_score"])
        dominant_bearing = peak_summary["peak_bearing_deg"]
        bearing_spread = peak_summary["peak_spread_deg"]
        rows.append(
            {
                "Cell ID": cell_id,
                "dominant_bearing_deg": dominant_bearing,
                "configured_azimuth_deg": ant.get("Azimuth_cfg", np.nan),
                "bearing_mismatch_deg": TILT_SRC._angular_diff(dominant_bearing, ant.get("Azimuth_cfg", np.nan)),
                "bearing_sample_count": int(len(use_dir)),
                "bearing_spread_deg": bearing_spread,
                "bearing_peak_share": peak_summary["peak_share"],
                "bearing_peak_weight": peak_summary["peak_weight"],
                "bearing_second_peak_weight": peak_summary["second_peak_weight"],
                "bearing_directional_contrast": peak_summary["directional_contrast"],
            }
        )

    return pd.DataFrame(rows)


def _attach_geo_to_bad_samples(bad_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    out = bad_df.copy()
    if geo_df.empty:
        return out

    geo_work = geo_df.copy()
    out["Cell ID"] = out["Cell ID"].map(TILT_SRC._norm_cell_id)
    geo_work["Cell ID"] = geo_work["Node_Cell_ID"].astype(str).map(TILT_SRC._norm_cell_id)

    lat_col = TILT_SRC._find_col(out, ["lat", "latitude"], required=False)
    lon_col = TILT_SRC._find_col(out, ["lon", "longitude"], required=False)
    if not lat_col or not lon_col:
        return out

    out["lat_6dp"] = pd.to_numeric(out[lat_col], errors="coerce").round(6)
    out["lon_6dp"] = pd.to_numeric(out[lon_col], errors="coerce").round(6)
    geo_work["lat_6dp"] = pd.to_numeric(geo_work["lat"], errors="coerce").round(6)
    geo_work["lon_6dp"] = pd.to_numeric(geo_work["lon"], errors="coerce").round(6)

    geo_cols = [
        "Cell ID",
        "lat_6dp",
        "lon_6dp",
        "clutter_class",
        "morphology_cluster",
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "los_blocker_count",
        "los_blocked_ratio",
        "max_blocker_height_m",
        "diffraction_proxy_db",
        "nlos_flag",
        "terrain_elevation_m",
        "terrain_slope_deg",
        "proxy_site_elevation_m",
        "terrain_relief_to_site_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "azimuth_delta_deg",
    ]
    geo_cols = [c for c in geo_cols if c in geo_work.columns]
    return out.merge(
        geo_work[geo_cols].drop_duplicates(subset=["Cell ID", "lat_6dp", "lon_6dp"], keep="last"),
        on=["Cell ID", "lat_6dp", "lon_6dp"],
        how="left",
    )


def _mode_or_blank(series: pd.Series) -> str:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return ""
    mode = vals.mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else ""


def _fmt_num(value: float, decimals: int) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{decimals}f}"


def _norm_reason_token(value: str) -> str:
    token = str(value or "").strip().lower()
    token = token.replace("/", "_").replace("-", "_").replace(" ", "_")
    return token or "unknown"


def _values_changed(curr, rec) -> bool:
    curr_num = pd.to_numeric(pd.Series([curr]), errors="coerce").iloc[0]
    rec_num = pd.to_numeric(pd.Series([rec]), errors="coerce").iloc[0]
    if pd.notna(curr_num) and pd.notna(rec_num):
        return not np.isclose(float(curr_num), float(rec_num), equal_nan=True)
    return str(curr) != str(rec)


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(float(value)))))


def _confidence_label(score: int) -> str:
    if score >= 75:
        return "high"
    if score >= 55:
        return "medium"
    return "low"


def _changed_recommendation_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    return df.apply(
        lambda r: _values_changed(r.get("Current Value"), r.get("Recommended Value")),
        axis=1,
    )


def _circular_spread_deg(values: pd.Series, center_deg: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty or pd.isna(center_deg):
        return np.nan
    diffs = vals.map(lambda v: TILT_SRC._angular_diff(v, center_deg))
    return float(pd.to_numeric(diffs, errors="coerce").mean())


def _directional_peak_summary(
    values: pd.Series,
    weights: pd.Series,
    bin_size_deg: float = BEARING_BIN_SIZE_DEG,
) -> Dict[str, float]:
    bearings = pd.to_numeric(values, errors="coerce")
    bearing_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    work = pd.DataFrame({"bearing": bearings, "weight": bearing_weights}).dropna(subset=["bearing"])
    if work.empty:
        return {
            "peak_bearing_deg": np.nan,
            "peak_spread_deg": np.nan,
            "peak_share": np.nan,
            "peak_weight": 0.0,
            "second_peak_weight": 0.0,
            "directional_contrast": np.nan,
            "total_weight": 0.0,
        }

    work["bearing"] = work["bearing"].astype(float).map(TILT_SRC._normalize_azimuth)
    work["weight"] = work["weight"].astype(float).clip(lower=0.0)
    work["weight"] = work["weight"].where(work["weight"] > 0.0, 1.0)

    bin_count = max(1, int(round(360.0 / float(bin_size_deg))))
    bin_idx = np.floor(work["bearing"].to_numpy() / float(bin_size_deg)).astype(int) % bin_count
    work["bin_idx"] = bin_idx
    bin_weights = np.bincount(bin_idx, weights=work["weight"].to_numpy(), minlength=bin_count).astype(float)
    smooth_weights = bin_weights.copy()
    if bin_count > 1:
        smooth_weights = bin_weights + np.roll(bin_weights, 1) + np.roll(bin_weights, -1)

    peak_bin = int(np.argmax(smooth_weights))
    bin_distance = ((work["bin_idx"] - peak_bin + bin_count / 2.0) % bin_count) - (bin_count / 2.0)
    local_lobe = work.loc[bin_distance.abs() <= 1].copy()
    if local_lobe.empty:
        local_lobe = work.copy()

    peak_bearing = TILT_SRC._circular_mean_deg(local_lobe["bearing"])
    peak_spread = _circular_spread_deg(local_lobe["bearing"], peak_bearing)
    peak_weight = float(local_lobe["weight"].sum())
    total_weight = float(work["weight"].sum())
    peak_share = peak_weight / total_weight if total_weight > 0 else np.nan
    protected_bins = {(peak_bin - 1) % bin_count, peak_bin % bin_count, (peak_bin + 1) % bin_count}
    second_peak_weight = float(
        max((smooth_weights[idx] for idx in range(bin_count) if idx not in protected_bins), default=0.0)
    )
    directional_contrast = peak_weight / second_peak_weight if second_peak_weight > 0 else np.inf
    return {
        "peak_bearing_deg": peak_bearing,
        "peak_spread_deg": peak_spread,
        "peak_share": peak_share,
        "peak_weight": peak_weight,
        "second_peak_weight": second_peak_weight,
        "directional_contrast": directional_contrast,
        "total_weight": total_weight,
    }


def _aggregate_bad_geo_context(bad_geo_df: pd.DataFrame) -> pd.DataFrame:
    if bad_geo_df.empty:
        return pd.DataFrame()

    work = bad_geo_df.copy()
    work["nlos_flag_num"] = pd.to_numeric(work.get("nlos_flag"), errors="coerce").fillna(0.0)
    grouped = work.groupby(["Cell ID", "Technology"], dropna=False)
    summary = grouped.agg(
        bad_sample_count=("Cell ID", "size"),
        mean_serving_distance_m=("serving_distance_m", "mean"),
        p90_serving_distance_m=(
            "serving_distance_m",
            lambda s: pd.to_numeric(s, errors="coerce").dropna().quantile(0.90)
            if pd.to_numeric(s, errors="coerce").dropna().size
            else np.nan,
        ),
        mean_nearest_site_distance_m=("nearest_site_distance_m", "mean"),
        mean_nearest3_site_distance_m=("mean_nearest3_site_distance_m", "mean"),
        mean_site_count_250m=("site_count_250m", "mean"),
        mean_site_count_500m=("site_count_500m", "mean"),
        mean_azimuth_delta_deg=("azimuth_delta_deg", "mean"),
        mean_building_area_ratio=("building_area_ratio", "mean"),
        mean_building_count=("building_count", "mean"),
        mean_los_blocked_ratio=("los_blocked_ratio", "mean"),
        mean_los_blocker_count=("los_blocker_count", "mean"),
        mean_green_ratio=("green_ratio", "mean"),
        mean_water_ratio=("water_ratio", "mean"),
        mean_road_length_m=("road_length_m", "mean"),
        mean_terrain_slope_deg=("terrain_slope_deg", "mean"),
        mean_terrain_relief_to_site_m=("terrain_relief_to_site_m", "mean"),
        nlos_share=("nlos_flag_num", "mean"),
        clutter_mode=("clutter_class", _mode_or_blank),
    ).reset_index()
    return summary


def _signed_azimuth_delta(target_deg: float, current_deg: float) -> float:
    if pd.isna(target_deg) or pd.isna(current_deg):
        return np.nan
    return ((float(target_deg) - float(current_deg) + 180.0) % 360.0) - 180.0


def _bounded_etilt_target(current_etilt: float, requested_etilt: float) -> float:
    if pd.isna(current_etilt) or pd.isna(requested_etilt):
        return np.nan
    lower_bound = max(float(MIN_SAFE_ETILT_DEG), float(current_etilt) - float(MAX_ETILT_DECREASE_PER_RUN_DEG))
    upper_bound = min(float(MAX_SAFE_ETILT_DEG), float(current_etilt) + float(MAX_ETILT_INCREASE_PER_RUN_DEG))
    return float(np.clip(float(requested_etilt), lower_bound, upper_bound))


def _select_best_candidate(candidates: List[Dict[str, object]]) -> Dict[str, object]:
    return max(candidates, key=lambda c: (float(c["score"]), -abs(float(c.get("delta", 0.0)))))


def _select_best_candidate_with_gap(
    candidates: List[Dict[str, object]],
    min_gap: float = MIN_CANDIDATE_SCORE_GAP,
) -> Dict[str, object]:
    ranked = sorted(
        candidates,
        key=lambda c: (float(c["score"]), -abs(float(c.get("delta", 0.0)))),
        reverse=True,
    )
    best = ranked[0]
    if len(ranked) == 1 or str(best.get("mode")) == "hold":
        return best
    second = ranked[1]
    if float(best["score"]) - float(second["score"]) < float(min_gap):
        hold = next((c for c in candidates if str(c.get("mode")) == "hold"), None)
        if hold is not None:
            return hold
    return best


def _estimate_etilt_validation_gain(
    mode: str,
    delta: float,
    coverage_score: float,
    overlap_score: float,
    mean_dist: float,
    p90_dist: float,
    overlap_dense: bool,
    interference_signal_present: bool,
    coverage_signal_present: bool,
    coverage_dominant: bool,
    medium_edge_issue: bool,
    far_edge_issue: bool,
    nlos_share: float,
    los_blocked: float,
) -> float:
    gain = 0.0
    abs_delta = abs(float(delta))
    if mode == "coverage":
        gain += max(0.0, coverage_score - COVERAGE_ETILT_ACTION_SCORE)
        if far_edge_issue:
            gain += 8.0
        elif medium_edge_issue:
            gain += 5.0
        if coverage_dominant:
            gain += 4.0
        if not pd.isna(mean_dist):
            gain += min(6.0, max(0.0, (float(mean_dist) - MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M) / 12.0))
        if not pd.isna(p90_dist):
            gain += min(6.0, max(0.0, (float(p90_dist) - MEDIUM_EDGE_P90_SERVING_DISTANCE_M) / 16.0))
        if overlap_dense and interference_signal_present:
            gain -= 5.0 * abs_delta
        if not pd.isna(nlos_share) and nlos_share >= HIGH_NLOS_SHARE_GATE:
            gain -= 5.0
        if not pd.isna(los_blocked) and los_blocked >= HIGH_LOS_BLOCKED_RATIO_GATE:
            gain -= 4.0
    elif mode == "overlap":
        gain += max(0.0, overlap_score - OVERLAP_ETILT_ACTION_SCORE)
        if overlap_dense:
            gain += 8.0
        if interference_signal_present:
            gain += 5.0
        if coverage_signal_present and (far_edge_issue or medium_edge_issue):
            gain -= 4.0 * abs_delta
        if not pd.isna(mean_dist) and mean_dist >= FAR_EDGE_MEAN_SERVING_DISTANCE_M:
            gain -= 4.0
        if not pd.isna(p90_dist) and p90_dist >= FAR_EDGE_P90_SERVING_DISTANCE_M:
            gain -= 4.0
        if not pd.isna(nlos_share) and nlos_share >= HIGH_NLOS_SHARE_GATE:
            gain -= 2.0
    return gain


def _estimate_azimuth_validation_gain(
    current_az: float,
    target_az: float,
    dominant_bearing: float,
    azimuth_score: float,
    bearing_mismatch: float,
    bearing_peak_share: float,
    directional_contrast: float,
    bearing_spread: float,
    nlos_share: float,
    los_blocked: float,
    overlap_dense: bool,
    interference_signal_present: bool,
) -> float:
    if pd.isna(current_az) or pd.isna(target_az) or pd.isna(dominant_bearing):
        return -999.0
    current_residual = TILT_SRC._angular_diff(dominant_bearing, current_az)
    target_residual = TILT_SRC._angular_diff(dominant_bearing, target_az)
    correction_gain = max(0.0, float(current_residual) - float(target_residual))
    gain = max(0.0, azimuth_score - AZIMUTH_ACTION_SCORE)
    gain += correction_gain * 1.8
    if not pd.isna(bearing_mismatch):
        gain += min(6.0, float(bearing_mismatch) / 8.0)
    if not pd.isna(bearing_peak_share):
        gain += max(0.0, (float(bearing_peak_share) - 0.25) * 20.0)
    if not pd.isna(directional_contrast):
        gain += max(0.0, (float(directional_contrast) - 1.0) * 5.0)
    if not pd.isna(bearing_spread):
        gain += max(0.0, (MAX_BEARING_SPREAD_DEG - float(bearing_spread)) / 5.0)
    if overlap_dense and interference_signal_present:
        gain += 2.0
    if not pd.isna(nlos_share) and nlos_share >= MEDIUM_NLOS_SHARE_GATE:
        gain -= 6.0
    if not pd.isna(los_blocked) and los_blocked >= HIGH_LOS_BLOCKED_RATIO_GATE:
        gain -= 5.0
    if target_residual > current_residual:
        gain -= 12.0
    return gain


def _build_geo_aware_recommendations(
    bad_summary: pd.DataFrame,
    antenna_df: pd.DataFrame,
    swap_dict: Dict[str, str],
    geo_cell_summary: pd.DataFrame,
    bearing_summary: pd.DataFrame,
) -> pd.DataFrame:
    bad_summary = TILT_SRC._normalize_columns(bad_summary).copy()
    antenna_df = TILT_SRC._normalize_columns(antenna_df).copy()
    geo_cell_summary = geo_cell_summary.copy()
    bearing_summary = bearing_summary.copy()

    bad_summary["Cell ID"] = bad_summary["Cell ID"].map(TILT_SRC._norm_cell_id)
    antenna_df["Cell ID"] = _derive_antenna_cell_key(antenna_df)
    if not geo_cell_summary.empty:
        geo_cell_summary["Cell ID"] = geo_cell_summary["Cell ID"].map(TILT_SRC._norm_cell_id)
    if not bearing_summary.empty:
        bearing_summary["Cell ID"] = bearing_summary["Cell ID"].map(TILT_SRC._norm_cell_id)

    ant_use = antenna_df.drop_duplicates(subset=["Cell ID"]).copy()
    ant_use["Cell ID Suffix"] = ant_use["Cell ID"].map(TILT_SRC._cell_id_suffix)
    ant_local_cell_col = TILT_SRC._find_col(antenna_df, ["cell_id", "eci", "local_cell_id"], required=False)
    if ant_local_cell_col:
        ant_use["Antenna Local Cell"] = ant_use[ant_local_cell_col].map(TILT_SRC._norm_cell_id)
    else:
        ant_use["Antenna Local Cell"] = ant_use["Cell ID Suffix"]

    tech_col = TILT_SRC._find_col(antenna_df, ["Technology", "technology"], required=False)
    az_col = TILT_SRC._find_col(antenna_df, ["azimuth", "azi"], required=False)
    etilt_col = TILT_SRC._find_col(antenna_df, ["e_tilt", "etilt", "electrical_tilt"], required=False)
    power_col = TILT_SRC._find_col(antenna_df, ["tx_power", "real_transmit_power_of_resource", "reference_signal_power"], required=False)

    geo_map = geo_cell_summary.set_index("Cell ID").to_dict("index") if not geo_cell_summary.empty else {}
    bearing_map = bearing_summary.set_index("Cell ID").to_dict("index") if not bearing_summary.empty else {}
    site_stats: Dict[str, Dict[str, float]] = {}
    if not bad_summary.empty:
        tmp = bad_summary.copy()
        tmp["site_key"] = tmp["Cell ID"].astype(str).str.split("_").str[0]
        tmp["total_bad_samples"] = (
            pd.to_numeric(tmp.get("Bad RSRP", 0), errors="coerce").fillna(0)
            + pd.to_numeric(tmp.get("Bad RSRQ", 0), errors="coerce").fillna(0)
            + pd.to_numeric(tmp.get("Bad SINR", 0), errors="coerce").fillna(0)
        )
        site_stats = (
            tmp.groupby("site_key", dropna=False)
            .agg(
                site_sector_count=("Cell ID", "nunique"),
                site_total_bad_samples=("total_bad_samples", "sum"),
                site_mean_bad_samples=("total_bad_samples", "mean"),
                site_max_bad_samples=("total_bad_samples", "max"),
            )
            .to_dict("index")
        )

    rec_rows: List[Dict[str, object]] = []
    for _, row in bad_summary.iterrows():
        cell_id = TILT_SRC._norm_cell_id(row["Cell ID"])
        site_key = cell_id.split("_")[0] if "_" in cell_id else cell_id
        tech = TILT_SRC._safe_str(row.get("Technology", "")) or "UNKNOWN"
        bad_rsrp = int(row.get("Bad RSRP", 0) or 0)
        bad_rsrq = int(row.get("Bad RSRQ", 0) or 0)
        bad_sinr = int(row.get("Bad SINR", 0) or 0)
        total_bad = bad_rsrp + bad_rsrq + bad_sinr
        swap_flag = swap_dict.get(cell_id, "No")

        ant_row = ant_use.loc[ant_use["Cell ID"] == cell_id]
        if ant_row.empty:
            cell_suffix = TILT_SRC._cell_id_suffix(cell_id)
            if cell_suffix:
                ant_row = ant_use.loc[ant_use["Antenna Local Cell"] == cell_suffix]
            if len(ant_row) != 1 and cell_suffix:
                ant_row = ant_use.loc[ant_use["Cell ID Suffix"] == cell_suffix]
        if ant_row.empty:
            continue
        if len(ant_row) > 1:
            ant_row = ant_row.iloc[[0]]
        ant = ant_row.iloc[0]

        ant_tech = TILT_SRC._safe_str(ant[tech_col]) if tech_col and tech_col in ant else tech
        curr_az = pd.to_numeric(ant[az_col], errors="coerce") if az_col else np.nan
        curr_etilt = pd.to_numeric(ant[etilt_col], errors="coerce") if etilt_col else np.nan
        curr_power = pd.to_numeric(ant[power_col], errors="coerce") if power_col else np.nan

        geo_ctx = geo_map.get(cell_id, {})
        bearing_ctx = bearing_map.get(cell_id, {})

        bad_sample_count = int(pd.to_numeric(pd.Series([geo_ctx.get("bad_sample_count", total_bad)]), errors="coerce").fillna(total_bad).iloc[0])
        mean_dist = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_serving_distance_m")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        p90_dist = float(pd.to_numeric(pd.Series([geo_ctx.get("p90_serving_distance_m")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        mean_az_delta = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_azimuth_delta_deg")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        nlos_share = float(pd.to_numeric(pd.Series([geo_ctx.get("nlos_share")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        los_blocked = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_los_blocked_ratio")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        building_ratio = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_building_area_ratio")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        nearest_site = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_nearest_site_distance_m")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        site_count_250m = float(pd.to_numeric(pd.Series([geo_ctx.get("mean_site_count_250m")]), errors="coerce").iloc[0]) if geo_ctx else np.nan
        clutter_mode = str(geo_ctx.get("clutter_mode", "") or "")

        dominant_bearing = float(pd.to_numeric(pd.Series([bearing_ctx.get("dominant_bearing_deg")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        bearing_mismatch = float(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_mismatch_deg")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        bearing_sample_count = int(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_sample_count", 0)]), errors="coerce").fillna(0).iloc[0]) if bearing_ctx else 0
        bearing_spread = float(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_spread_deg")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        bearing_peak_share = float(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_peak_share")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        directional_contrast = float(pd.to_numeric(pd.Series([bearing_ctx.get("bearing_directional_contrast")]), errors="coerce").iloc[0]) if bearing_ctx else np.nan
        site_ctx = site_stats.get(site_key, {})
        site_sector_count = int(pd.to_numeric(pd.Series([site_ctx.get("site_sector_count", 1)]), errors="coerce").fillna(1).iloc[0])
        site_mean_bad_samples = float(pd.to_numeric(pd.Series([site_ctx.get("site_mean_bad_samples", bad_sample_count)]), errors="coerce").fillna(bad_sample_count).iloc[0])
        site_total_bad_samples = float(pd.to_numeric(pd.Series([site_ctx.get("site_total_bad_samples", bad_sample_count)]), errors="coerce").fillna(bad_sample_count).iloc[0])

        low_signal = bad_sample_count < MIN_BAD_SAMPLE_COUNT_FOR_ACTION
        interference_signal_present = (bad_sinr + bad_rsrq) >= max(20, int(total_bad * 0.35))
        coverage_signal_ratio = bad_rsrp / max(total_bad, 1)
        interference_signal_ratio = (bad_sinr + bad_rsrq) / max(total_bad, 1)
        sinr_dominant = bad_sinr > max(bad_rsrp, bad_rsrq) and bad_sinr > 0
        rsrq_dominant = bad_rsrq > max(bad_rsrp, bad_sinr) and bad_rsrq > 0
        far_edge_issue = (
            (not pd.isna(mean_dist) and mean_dist >= FAR_EDGE_MEAN_SERVING_DISTANCE_M)
            or (not pd.isna(p90_dist) and p90_dist >= FAR_EDGE_P90_SERVING_DISTANCE_M)
        )
        medium_edge_issue = (
            (not pd.isna(mean_dist) and mean_dist >= MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M)
            or (not pd.isna(p90_dist) and p90_dist >= MEDIUM_EDGE_P90_SERVING_DISTANCE_M)
        )
        high_blockage = (
            (not pd.isna(nlos_share) and nlos_share >= HIGH_NLOS_SHARE_GATE)
            or (not pd.isna(los_blocked) and los_blocked >= HIGH_LOS_BLOCKED_RATIO_GATE)
            or (not pd.isna(building_ratio) and building_ratio >= HIGH_BUILDING_AREA_RATIO_GATE)
        )
        medium_blockage = (
            (not pd.isna(nlos_share) and nlos_share >= MEDIUM_NLOS_SHARE_GATE)
            or (not pd.isna(los_blocked) and los_blocked >= MEDIUM_LOS_BLOCKED_RATIO_GATE)
        )
        close_site_overlap = not pd.isna(nearest_site) and nearest_site <= DENSE_OVERLAP_NEAREST_SITE_M
        dense_site_overlap = not pd.isna(site_count_250m) and site_count_250m >= DENSE_OVERLAP_SITE_COUNT_250M
        overlap_dense = close_site_overlap and dense_site_overlap
        small_az_delta = pd.isna(mean_az_delta) or mean_az_delta <= SMALL_AZIMUTH_DELTA_DEG
        site_persistent_issue = (
            site_sector_count >= 3
            and site_mean_bad_samples >= MEDIUM_CONFIDENCE_BAD_SAMPLE_COUNT
            and site_total_bad_samples >= HIGH_CONFIDENCE_BAD_SAMPLE_COUNT * 3
        )
        persistent_coverage_signal = (
            site_persistent_issue
            and bad_rsrp >= max(12, int(total_bad * 0.20))
        )
        coverage_signal_present = (
            bad_rsrp >= max(20, int(total_bad * 0.30))
            or persistent_coverage_signal
        )
        coverage_dominant = coverage_signal_present and bad_rsrp >= max(bad_sinr, bad_rsrq)
        high_bad_volume_override = (
            bad_sample_count >= VERY_HIGH_CONFIDENCE_BAD_SAMPLE_COUNT
            and medium_edge_issue
            and not high_blockage
        )
        stable_bearing_case = (
            not pd.isna(dominant_bearing)
            and not pd.isna(bearing_mismatch)
            and not pd.isna(bearing_spread)
            and not pd.isna(bearing_peak_share)
            and bearing_sample_count >= MIN_BEARING_SAMPLE_COUNT
            and MIN_AZIMUTH_MISMATCH_DEG <= bearing_mismatch <= MAX_AZIMUTH_MISMATCH_DEG
            and bearing_spread <= MAX_BEARING_SPREAD_DEG
            and bearing_peak_share >= MIN_PEAK_SHARE_FOR_AZIMUTH
            and (pd.isna(nlos_share) or nlos_share < AZIMUTH_NLOS_HARD_BLOCK_GATE)
        )
        blockage_limited_case = coverage_signal_present and high_blockage and bad_rsrp >= max(10, int(total_bad * 0.20))
        site_symmetric_overlap_case = (
            site_sector_count >= 3
            and overlap_dense
            and interference_signal_present
            and site_mean_bad_samples > 0
            and abs(float(bad_sample_count) - float(site_mean_bad_samples)) <= float(site_mean_bad_samples) * SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO
        )

        coverage_etilt_score = 0.0
        overlap_etilt_score = 0.0
        azimuth_score = 0.0
        tx_power_score = 0.0

        if bad_sample_count >= VERY_HIGH_CONFIDENCE_BAD_SAMPLE_COUNT:
            coverage_etilt_score += 30.0
            overlap_etilt_score += 28.0
        elif bad_sample_count >= HIGH_CONFIDENCE_BAD_SAMPLE_COUNT:
            coverage_etilt_score += 22.0
            overlap_etilt_score += 20.0
        elif bad_sample_count >= MEDIUM_CONFIDENCE_BAD_SAMPLE_COUNT:
            coverage_etilt_score += 12.0
            overlap_etilt_score += 10.0

        if far_edge_issue:
            coverage_etilt_score += 22.0
            tx_power_score += 14.0
        elif medium_edge_issue:
            coverage_etilt_score += 14.0
            tx_power_score += 8.0

        if site_persistent_issue:
            coverage_etilt_score += 12.0
            overlap_etilt_score += 10.0
            azimuth_score += 8.0

        if coverage_signal_present:
            coverage_etilt_score += 12.0
            tx_power_score += 10.0
        if coverage_dominant:
            coverage_etilt_score += 8.0
            tx_power_score += 6.0
        if persistent_coverage_signal:
            coverage_etilt_score += 10.0

        if overlap_dense:
            overlap_etilt_score += 10.0
        if small_az_delta:
            overlap_etilt_score += 5.0
        if interference_signal_present:
            overlap_etilt_score += 10.0
            azimuth_score += 10.0
        if sinr_dominant:
            overlap_etilt_score += 6.0
            azimuth_score += 8.0
        elif rsrq_dominant:
            overlap_etilt_score += 3.0
            azimuth_score += 4.0

        if medium_edge_issue and coverage_signal_ratio >= 0.22 and not high_blockage:
            coverage_etilt_score += 12.0
        if medium_edge_issue and interference_signal_ratio < 0.72:
            coverage_etilt_score += 8.0
        if medium_edge_issue and not high_blockage and bad_sample_count >= HIGH_CONFIDENCE_BAD_SAMPLE_COUNT:
            coverage_etilt_score += 6.0
        if not pd.isna(p90_dist) and p90_dist >= 210.0 and not high_blockage:
            coverage_etilt_score += 4.0
        if overlap_dense and coverage_signal_ratio >= 0.28 and not high_blockage:
            coverage_etilt_score += 6.0
            overlap_etilt_score -= 6.0
        if medium_blockage and not high_blockage and medium_edge_issue:
            coverage_etilt_score += 5.0
        if interference_signal_ratio >= 0.72 and overlap_dense:
            overlap_etilt_score += 4.0
        if site_symmetric_overlap_case:
            overlap_etilt_score -= 2.0

        if stable_bearing_case:
            azimuth_score += 28.0
            azimuth_score += min(12.0, max(0.0, (MAX_BEARING_SPREAD_DEG - float(bearing_spread)) * 0.4))
            azimuth_score += min(12.0, max(0.0, (float(bearing_mismatch) - MIN_AZIMUTH_MISMATCH_DEG) * 0.4))
            azimuth_score += min(10.0, max(0.0, (float(bearing_peak_share) - MIN_PEAK_SHARE_FOR_AZIMUTH) * 40.0))
            if not pd.isna(directional_contrast):
                azimuth_score += min(12.0, max(0.0, (float(directional_contrast) - 1.0) * 6.0))
        elif bearing_sample_count > 0:
            azimuth_score -= 12.0

        if high_blockage:
            coverage_etilt_score -= 18.0
            tx_power_score -= 18.0
            if not overlap_dense:
                overlap_etilt_score -= 10.0
        elif medium_blockage:
            coverage_etilt_score -= 8.0
            tx_power_score -= 8.0

        if not pd.isna(nlos_share) and nlos_share >= AZIMUTH_NLOS_HARD_BLOCK_GATE:
            azimuth_score = -100.0

        if high_bad_volume_override:
            coverage_etilt_score += 10.0

        if interference_signal_present or overlap_dense:
            tx_power_score -= 18.0
        if not coverage_signal_present:
            tx_power_score -= 10.0
        if coverage_signal_present and not interference_signal_present and not overlap_dense:
            tx_power_score += 10.0
        if far_edge_issue and coverage_dominant and not high_blockage:
            tx_power_score += 8.0

        coverage_open_case = (
            coverage_signal_present
            and far_edge_issue
            and not high_blockage
            and not interference_signal_present
            and coverage_etilt_score >= COVERAGE_ETILT_ACTION_SCORE
        )
        coverage_moderate_case = (
            coverage_etilt_score >= COVERAGE_ETILT_ACTION_SCORE
            and medium_edge_issue
            and not high_blockage
        )
        overlap_limited_case = (
            overlap_etilt_score >= OVERLAP_ETILT_ACTION_SCORE
            and overlap_etilt_score >= coverage_etilt_score + 10.0
            and interference_signal_present
            and overlap_dense
            and small_az_delta
        )
        overlap_moderate_case = (
            overlap_etilt_score >= OVERLAP_ETILT_ACTION_SCORE
            and overlap_etilt_score >= coverage_etilt_score + 8.0
            and overlap_dense
            and (interference_signal_present or sinr_dominant or rsrq_dominant)
        )

        blockage_score = 0.0
        if high_blockage:
            blockage_score += 30.0
        elif medium_blockage:
            blockage_score += 16.0
        if not pd.isna(nlos_share):
            blockage_score += min(18.0, max(0.0, (float(nlos_share) - MEDIUM_NLOS_SHARE_GATE) * 60.0))

        directional_misalignment_case = (
            stable_bearing_case
            and not high_blockage
            and (pd.isna(nlos_share) or nlos_share < MEDIUM_NLOS_SHARE_GATE)
            and not pd.isna(directional_contrast)
            and directional_contrast >= MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH
            and bearing_peak_share >= max(MIN_PEAK_SHARE_FOR_AZIMUTH - 0.08, 0.30)
        )
        if directional_misalignment_case:
            azimuth_score += 6.0
            if site_symmetric_overlap_case:
                azimuth_score += 4.0
        strong_azimuth_override = (
            stable_bearing_case
            and not pd.isna(directional_contrast)
            and directional_contrast >= 1.35
            and not pd.isna(bearing_peak_share)
            and bearing_peak_share >= 0.34
            and not pd.isna(bearing_mismatch)
            and 20.0 <= bearing_mismatch <= 35.0
            and bad_sample_count >= 80
            and (pd.isna(nlos_share) or nlos_share < AZIMUTH_NLOS_HARD_BLOCK_GATE)
            and (pd.isna(los_blocked) or los_blocked < HIGH_LOS_BLOCKED_RATIO_GATE)
        )
        if strong_azimuth_override:
            azimuth_score += 4.0

        dominant_root_cause = "mixed"
        if low_signal:
            dominant_root_cause = "low_signal_confidence"
        else:
            root_scores = {
                "blockage_dominated": blockage_score,
                "coverage_candidate": coverage_etilt_score,
                "interference_candidate": overlap_etilt_score,
                "directional_misalignment": azimuth_score if directional_misalignment_case else 0.0,
                "interference_soft": 18.0 if interference_signal_present else 0.0,
                "coverage_soft": 16.0 if coverage_signal_present else 0.0,
            }
            dominant_root_cause = max(root_scores, key=root_scores.get)
            if dominant_root_cause == "interference_candidate" and overlap_etilt_score >= max(OVERLAP_ETILT_ACTION_SCORE, coverage_etilt_score + 10.0):
                dominant_root_cause = "interference_limited"

        action_family = "mixed"
        if dominant_root_cause == "coverage_candidate":
            action_family = "coverage"
        elif dominant_root_cause in {"interference_candidate", "interference_limited"}:
            action_family = "interference"
        elif dominant_root_cause == "directional_misalignment":
            action_family = "azimuth"
        elif dominant_root_cause == "blockage_dominated":
            action_family = "blockage"
        if strong_azimuth_override and azimuth_score >= AZIMUTH_ACTION_SCORE:
            action_family = "azimuth"

        relaxed_azimuth_case = (
            stable_bearing_case
            and bearing_sample_count >= max(20, MIN_BEARING_SAMPLE_COUNT - 10)
            and bearing_peak_share >= max(0.34, RELAXED_AZIMUTH_PEAK_SHARE - 0.12)
            and bearing_spread <= RELAXED_AZIMUTH_MAX_SPREAD_DEG
            and bad_sample_count >= RELAXED_AZIMUTH_BAD_SAMPLE_COUNT
            and not pd.isna(directional_contrast)
            and directional_contrast >= max(1.15, MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH - 0.15)
        )
        azimuth_preferred_case = (
            directional_misalignment_case
            and azimuth_score >= AZIMUTH_ACTION_SCORE
            and action_family == "azimuth"
            and bearing_peak_share >= max(0.36, RELAXED_AZIMUTH_PEAK_SHARE - 0.10)
            and not pd.isna(directional_contrast)
            and directional_contrast >= max(1.35, STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH - 0.45)
            and (pd.isna(nlos_share) or nlos_share < MEDIUM_NLOS_SHARE_GATE)
            and azimuth_score >= max(overlap_etilt_score, coverage_etilt_score) + 3.0
        )

        family_priority = {"coverage": 1.5, "interference": 1.5, "azimuth": 1.5, "blockage": 0.5, "mixed": 1.0}

        def _family_bonus(candidate_family: str) -> float:
            if action_family == candidate_family:
                return 6.0
            if action_family == "mixed":
                return 2.0
            if action_family == "blockage":
                return -2.0 if candidate_family in {"coverage", "interference", "azimuth"} else 0.0
            return 0.0

        rec_etilt = curr_etilt
        etilt_reason = "No ETilt change required."
        etilt_status = "no_change"
        etilt_confidence_raw = max(coverage_etilt_score, overlap_etilt_score)
        etilt_confidence_score = 0.0
        best_etilt_score = 0.0
        if not pd.isna(curr_etilt):
            if low_signal:
                etilt_reason = "Bad sample volume is too small for a reliable tilt change; collect more evidence before acting."
                etilt_status = "low_confidence_hold"
            else:
                etilt_candidates: List[Dict[str, object]] = [
                    {"mode": "hold", "family": "hold", "delta": 0.0, "target": float(curr_etilt), "score": 0.0, "reason": "No ETilt change required."}
                ]
                if overlap_etilt_score >= max(OVERLAP_ETILT_ACTION_SCORE, coverage_etilt_score + 2.0):
                    overlap_gain_1 = _estimate_etilt_validation_gain(
                        mode="overlap",
                        delta=1.0,
                        coverage_score=coverage_etilt_score,
                        overlap_score=overlap_etilt_score,
                        mean_dist=mean_dist,
                        p90_dist=p90_dist,
                        overlap_dense=overlap_dense,
                        interference_signal_present=interference_signal_present,
                        coverage_signal_present=coverage_signal_present,
                        coverage_dominant=coverage_dominant,
                        medium_edge_issue=medium_edge_issue,
                        far_edge_issue=far_edge_issue,
                        nlos_share=nlos_share,
                        los_blocked=los_blocked,
                    )
                    etilt_candidates.append(
                        {
                            "mode": "overlap",
                            "family": "interference",
                            "delta": 1.0,
                            "target": _bounded_etilt_target(curr_etilt, curr_etilt + 1.0),
                            "score": overlap_etilt_score + overlap_gain_1 + _family_bonus("interference"),
                            "reason": "Dense overlap and persistent interference evidence favor footprint tightening within safe ETilt bounds; increasing ETilt should reduce spillover.",
                        }
                    )
                    if bad_sinr >= max(8, bad_rsrp * 1.75) and overlap_etilt_score >= coverage_etilt_score + 8.0:
                        overlap_gain_2 = _estimate_etilt_validation_gain(
                            mode="overlap",
                            delta=2.0,
                            coverage_score=coverage_etilt_score,
                            overlap_score=overlap_etilt_score,
                            mean_dist=mean_dist,
                            p90_dist=p90_dist,
                            overlap_dense=overlap_dense,
                            interference_signal_present=interference_signal_present,
                            coverage_signal_present=coverage_signal_present,
                            coverage_dominant=coverage_dominant,
                            medium_edge_issue=medium_edge_issue,
                            far_edge_issue=far_edge_issue,
                            nlos_share=nlos_share,
                            los_blocked=los_blocked,
                        )
                        etilt_candidates.append(
                            {
                                "mode": "overlap",
                                "family": "interference",
                                "delta": 2.0,
                                "target": _bounded_etilt_target(curr_etilt, curr_etilt + 2.0),
                                "score": overlap_etilt_score + overlap_gain_2 + _family_bonus("interference"),
                                "reason": "Dense overlap and persistent interference evidence strongly favor footprint tightening within safe ETilt bounds; a larger ETilt increase should reduce spillover.",
                            }
                        )
                if coverage_etilt_score >= COVERAGE_ETILT_ACTION_SCORE:
                    coverage_gain_1 = _estimate_etilt_validation_gain(
                        mode="coverage",
                        delta=-1.0,
                        coverage_score=coverage_etilt_score,
                        overlap_score=overlap_etilt_score,
                        mean_dist=mean_dist,
                        p90_dist=p90_dist,
                        overlap_dense=overlap_dense,
                        interference_signal_present=interference_signal_present,
                        coverage_signal_present=coverage_signal_present,
                        coverage_dominant=coverage_dominant,
                        medium_edge_issue=medium_edge_issue,
                        far_edge_issue=far_edge_issue,
                        nlos_share=nlos_share,
                        los_blocked=los_blocked,
                    )
                    etilt_candidates.append(
                        {
                            "mode": "coverage",
                            "family": "coverage",
                            "delta": -1.0,
                            "target": _bounded_etilt_target(curr_etilt, curr_etilt - 1.0),
                            "score": coverage_etilt_score + coverage_gain_1 + _family_bonus("coverage"),
                            "reason": "Persistent urban coverage weakness is present with edge-distance evidence, bounded by safe ETilt limits; reducing ETilt should open the footprint toward underserved users.",
                        }
                    )
                    if (
                        not pd.isna(p90_dist)
                        and p90_dist >= 280.0
                        and coverage_etilt_score >= overlap_etilt_score + 6.0
                    ):
                        coverage_gain_2 = _estimate_etilt_validation_gain(
                            mode="coverage",
                            delta=-2.0,
                            coverage_score=coverage_etilt_score,
                            overlap_score=overlap_etilt_score,
                            mean_dist=mean_dist,
                            p90_dist=p90_dist,
                            overlap_dense=overlap_dense,
                            interference_signal_present=interference_signal_present,
                            coverage_signal_present=coverage_signal_present,
                            coverage_dominant=coverage_dominant,
                            medium_edge_issue=medium_edge_issue,
                            far_edge_issue=far_edge_issue,
                            nlos_share=nlos_share,
                            los_blocked=los_blocked,
                        )
                        etilt_candidates.append(
                            {
                                "mode": "coverage",
                                "family": "coverage",
                                "delta": -2.0,
                                "target": _bounded_etilt_target(curr_etilt, curr_etilt - 2.0),
                                "score": coverage_etilt_score + coverage_gain_2 + _family_bonus("coverage"),
                                "reason": "Coverage-opening evidence is very strong at the cell edge, so a larger ETilt reduction is justified within safe bounds.",
                            }
                        )
                if action_family == "blockage" and blockage_limited_case and not high_blockage and coverage_etilt_score >= COVERAGE_ETILT_ACTION_SCORE - 4.0:
                    etilt_candidates.append(
                        {
                            "mode": "coverage",
                            "family": "coverage",
                            "delta": -1.0,
                            "target": _bounded_etilt_target(curr_etilt, curr_etilt - 1.0),
                            "score": coverage_etilt_score + 1.0,
                            "reason": "Blocked geometry is present, but a bounded 1 deg ETilt opening is allowed as a cautious secondary recovery path.",
                        }
                    )
                best_etilt = _select_best_candidate_with_gap(etilt_candidates, min_gap=2.0)
                rec_etilt = float(best_etilt["target"])
                best_etilt_score = float(best_etilt["score"])
                if best_etilt["mode"] != "hold" and not np.isclose(rec_etilt, float(curr_etilt), equal_nan=True):
                    etilt_reason = str(best_etilt["reason"])
                    etilt_status = "action_change"
                elif blockage_limited_case and overlap_etilt_score < OVERLAP_ETILT_ACTION_SCORE:
                    etilt_reason = (
                        "Bad RSRP occurs in blocked/NLOS geometry, so tilt is probably not the first fix; "
                        "building or terrain blockage dominates the problem area."
                    )
                    etilt_status = "blocked_by_blockage"
                elif action_family == "blockage":
                    etilt_reason = "Blockage dominates this cell, so ETilt is held unless a cautious secondary recovery path shows enough value."
                    etilt_status = "blocked_by_blockage"
        etilt_confidence_floor = 0.0

        rec_az = curr_az
        az_reason = "No azimuth change required."
        az_status = "no_change"
        az_confidence_raw = max(0.0, azimuth_score)
        az_confidence_score = 0
        best_az_score = 0.0
        if not pd.isna(curr_az):
            if low_signal:
                az_reason = "Bad sample volume is too small for a reliable azimuth change; hold azimuth until more evidence is available."
                az_status = "low_confidence_hold"
            elif str(swap_flag).strip().upper() == "YES":
                az_reason = "Swap sector is suspected, so azimuth should be held until mapping is validated."
                az_status = "hold_swap"
                az_confidence_raw = 0.0
            elif not pd.isna(nlos_share) and nlos_share >= AZIMUTH_NLOS_HARD_BLOCK_GATE:
                az_reason = "Extreme NLOS geometry makes dominant-bearing direction unreliable here, so azimuth is held."
                az_status = "blocked_by_blockage"
                az_confidence_raw = 0.0
            elif blockage_limited_case and not strong_azimuth_override:
                az_reason = "Blocked geometry makes directional steering unreliable here, so azimuth is held."
                az_status = "blocked_by_blockage"
                az_confidence_raw = 0.0
            elif azimuth_score >= AZIMUTH_ACTION_SCORE and (
                action_family == "azimuth"
                or site_persistent_issue
                or relaxed_azimuth_case
                or azimuth_preferred_case
                or strong_azimuth_override
                or directional_misalignment_case
            ):
                signed_delta = _signed_azimuth_delta(dominant_bearing, curr_az)
                signed_delta = float(np.clip(signed_delta, -MAX_AZIMUTH_STEP_DEG, MAX_AZIMUTH_STEP_DEG))
                if abs(signed_delta) >= 5.0:
                    az_gain_5 = _estimate_azimuth_validation_gain(
                        current_az=curr_az,
                        target_az=TILT_SRC._normalize_azimuth(curr_az + (5.0 if signed_delta > 0 else -5.0)),
                        dominant_bearing=dominant_bearing,
                        azimuth_score=azimuth_score,
                        bearing_mismatch=bearing_mismatch,
                        bearing_peak_share=bearing_peak_share,
                        directional_contrast=directional_contrast,
                        bearing_spread=bearing_spread,
                        nlos_share=nlos_share,
                        los_blocked=los_blocked,
                        overlap_dense=overlap_dense,
                        interference_signal_present=interference_signal_present,
                    )
                    az_candidates: List[Dict[str, object]] = [
                        {"mode": "hold", "family": "hold", "delta": 0.0, "target": float(curr_az), "score": 0.0},
                        {
                            "mode": "steer",
                            "family": "azimuth",
                            "delta": 5.0 if signed_delta > 0 else -5.0,
                            "target": TILT_SRC._normalize_azimuth(curr_az + (5.0 if signed_delta > 0 else -5.0)),
                            "score": azimuth_score + az_gain_5 + _family_bonus("azimuth"),
                        },
                    ]
                    if abs(signed_delta) > 7.5:
                        az_target_full = TILT_SRC._normalize_azimuth(
                            curr_az + float(np.clip(signed_delta, -MAX_AZIMUTH_STEP_DEG, MAX_AZIMUTH_STEP_DEG))
                        )
                        az_gain_full = _estimate_azimuth_validation_gain(
                            current_az=curr_az,
                            target_az=az_target_full,
                            dominant_bearing=dominant_bearing,
                            azimuth_score=azimuth_score,
                            bearing_mismatch=bearing_mismatch,
                            bearing_peak_share=bearing_peak_share,
                            directional_contrast=directional_contrast,
                            bearing_spread=bearing_spread,
                            nlos_share=nlos_share,
                            los_blocked=los_blocked,
                            overlap_dense=overlap_dense,
                            interference_signal_present=interference_signal_present,
                        )
                        az_candidates.append(
                            {
                                "mode": "steer",
                                "family": "azimuth",
                                "delta": float(np.clip(signed_delta, -MAX_AZIMUTH_STEP_DEG, MAX_AZIMUTH_STEP_DEG)),
                                "target": az_target_full,
                                "score": azimuth_score + az_gain_full + _family_bonus("azimuth"),
                            }
                        )
                    best_az = _select_best_candidate_with_gap(az_candidates, min_gap=2.0)
                    rec_az = float(best_az["target"])
                    best_az_score = float(best_az["score"])
                    az_reason = (
                        f"Stable degraded directional peak is offset from configured azimuth by {bearing_mismatch:.1f} deg "
                        f"with bearing spread {bearing_spread:.1f} deg, so a bounded azimuth correction is applied."
                    )
                    if np.isclose(rec_az, float(curr_az), equal_nan=True):
                        az_reason = "Observed dominant bearing is close enough to configured azimuth after candidate validation; no azimuth change needed."
                    else:
                        az_status = "action_change"
                else:
                    az_reason = "Observed dominant bearing is close enough to configured azimuth; no azimuth change needed."
            elif overlap_etilt_score >= OVERLAP_ETILT_ACTION_SCORE:
                az_reason = "Interference evidence is stronger for ETilt tightening than for azimuth steering, so azimuth is held."
                az_status = "prefer_tilt_first"
            elif not stable_bearing_case and bearing_sample_count > 0:
                az_reason = "Directional evidence is not stable enough for safe azimuth steering; hold azimuth until the corridor is narrower and more consistent."

        if etilt_status == "action_change" and az_status == "action_change":
            etilt_priority = best_etilt_score + family_priority.get(str(action_family), 1.0)
            az_priority = best_az_score + family_priority.get("azimuth" if action_family == "azimuth" else "mixed", 1.0)
            if action_family == "azimuth" or azimuth_preferred_case or (strong_azimuth_override and az_priority >= etilt_priority - 2.0):
                etilt_reason = "Azimuth candidate validated better than ETilt for the selected root-cause family, so tilt is held first."
                etilt_status = "prefer_azimuth_first"
                rec_etilt = curr_etilt
            elif best_etilt_score >= best_az_score + 4.0:
                az_reason = "ETilt candidate validated better than azimuth for this cell, so azimuth is held to avoid stacking a weaker change."
                az_status = "prefer_tilt_first"
                rec_az = curr_az
            elif best_az_score > best_etilt_score:
                etilt_reason = "Azimuth candidate showed the stronger validated directional correction, so ETilt is held first."
                etilt_status = "prefer_azimuth_first"
                rec_etilt = curr_etilt
            else:
                az_reason = "ETilt candidate showed the stronger validated improvement, so azimuth is held first."
                az_status = "prefer_tilt_first"
                rec_az = curr_az
        elif etilt_status == "action_change" and azimuth_preferred_case:
            etilt_reason = "Directional misalignment evidence is cleaner than ETilt evidence here, so tilt is held while azimuth is preferred first."
            etilt_status = "prefer_azimuth_first"
            rec_etilt = curr_etilt
        elif az_status == "action_change" and etilt_status == "action_change":
            pass
        elif etilt_status == "action_change" and az_status == "no_change" and action_family == "azimuth" and best_az_score >= AZIMUTH_ACTION_SCORE - 2.0:
            etilt_reason = "Directional evidence points toward azimuth first, so ETilt is held until a clearer non-azimuth winner appears."
            etilt_status = "prefer_azimuth_first"
            rec_etilt = curr_etilt

        if etilt_confidence_raw >= COVERAGE_ETILT_ACTION_SCORE or etilt_confidence_raw >= OVERLAP_ETILT_ACTION_SCORE:
            if etilt_status in {"no_change", "prefer_azimuth_first", "prefer_tilt_first"}:
                etilt_confidence_floor = 55.0
        etilt_confidence_score = _clamp_score(
            max(
                0.0,
                etilt_confidence_raw if etilt_status == "action_change" else max(min(etilt_confidence_raw, 45.0), etilt_confidence_floor),
            )
        )
        etilt_confidence_label = _confidence_label(etilt_confidence_score)

        az_confidence_floor = 0.0
        if az_confidence_raw >= AZIMUTH_ACTION_SCORE and az_status in {"no_change", "prefer_tilt_first"}:
            az_confidence_floor = 55.0
        az_confidence_score = _clamp_score(
            max(
                0.0,
                az_confidence_raw if az_status == "action_change" else max(min(az_confidence_raw, 45.0), az_confidence_floor),
            )
        )
        az_confidence_label = _confidence_label(az_confidence_score)

        rec_power = curr_power
        pwr_reason = "No TX power change required."
        pwr_status = "no_change"
        pwr_confidence_raw = max(0.0, tx_power_score)
        pwr_confidence_score = 0
        if not pd.isna(curr_power):
            if low_signal:
                pwr_reason = "Bad sample volume is too small for a reliable power change recommendation."
                pwr_status = "low_confidence_hold"
            elif (
                tx_power_score >= TX_POWER_ACTION_SCORE
                and coverage_etilt_score >= COVERAGE_ETILT_ACTION_SCORE - 4.0
                and overlap_etilt_score <= OVERLAP_ETILT_ACTION_SCORE - 10.0
                and not interference_signal_present
                and not overlap_dense
                and not sinr_dominant
                and etilt_status != "action_change"
            ):
                rec_power = curr_power + 1.0
                pwr_reason = (
                    "Coverage weakness looks distance-driven with low overlap risk and no stronger ETilt winner, so a small +1 dB increase is allowed as a secondary action."
                )
                pwr_status = "action_change"
            elif interference_signal_present or overlap_dense:
                pwr_reason = "Power increase is avoided because the problem is interference/overlap driven."
                pwr_status = "held_for_interference"
        pwr_confidence_floor = 0.0
        if pwr_confidence_raw >= TX_POWER_ACTION_SCORE and pwr_status == "no_change":
            pwr_confidence_floor = 50.0
        pwr_confidence_score = _clamp_score(
            max(
                0.0,
                pwr_confidence_raw if pwr_status == "action_change" else max(min(pwr_confidence_raw, 40.0), pwr_confidence_floor),
            )
        )
        pwr_confidence_label = _confidence_label(pwr_confidence_score)

        base_context = (
            f"Root cause={dominant_root_cause}. "
            f"Bad sample count={bad_sample_count}. "
            f"Geo context: mean_serving_distance_m={_fmt_num(mean_dist, 1)}, "
            f"p90_serving_distance_m={_fmt_num(p90_dist, 1)}, "
            f"nlos_share={_fmt_num(nlos_share, 2)}, "
            f"mean_los_blocked_ratio={_fmt_num(los_blocked, 2)}, "
            f"clutter_mode={clutter_mode or 'n/a'}. "
            f"Scores: coverage_etilt={_fmt_num(coverage_etilt_score, 1)}, "
            f"overlap_etilt={_fmt_num(overlap_etilt_score, 1)}, "
            f"azimuth={_fmt_num(azimuth_score, 1)}, "
            f"tx_power={_fmt_num(tx_power_score, 1)}. "
            f"Bearing: peak_share={_fmt_num(bearing_peak_share, 2)}, "
            f"spread_deg={_fmt_num(bearing_spread, 1)}, "
            f"directional_contrast={_fmt_num(directional_contrast, 2)}."
        )

        rec_rows.extend([
            {
                "Cell ID": cell_id,
                "Technology": ant_tech or tech,
                "Parameter": "ETilt",
                "Current Value": "" if pd.isna(curr_etilt) else float(curr_etilt),
                "Recommended Value": "" if pd.isna(rec_etilt) else float(rec_etilt),
                "Reason": f"{etilt_reason} Confidence={etilt_confidence_label}({etilt_confidence_score}). {base_context}",
                "Swap Sector Detected": swap_flag,
                "Bad Sample Count": bad_sample_count,
                "P90 Serving Distance (m)": "" if pd.isna(p90_dist) else float(p90_dist),
                "Root Cause Category": dominant_root_cause,
                "Recommendation Status": etilt_status,
                "Recommendation Confidence": etilt_confidence_label,
                "Confidence Score": etilt_confidence_score,
                "Coverage ETilt Score": round(float(coverage_etilt_score), 2),
                "Overlap ETilt Score": round(float(overlap_etilt_score), 2),
                "Azimuth Score": round(float(azimuth_score), 2),
                "TX Power Score": round(float(tx_power_score), 2),
                "Bearing Peak Share": "" if pd.isna(bearing_peak_share) else round(float(bearing_peak_share), 4),
                "Bearing Spread Deg": "" if pd.isna(bearing_spread) else round(float(bearing_spread), 2),
                "Directional Contrast": "" if pd.isna(directional_contrast) else round(float(directional_contrast), 4),
            },
            {
                "Cell ID": cell_id,
                "Technology": ant_tech or tech,
                "Parameter": "Azimuth",
                "Current Value": "" if pd.isna(curr_az) else float(curr_az),
                "Recommended Value": "" if pd.isna(rec_az) else float(rec_az),
                "Reason": f"{az_reason} Confidence={az_confidence_label}({az_confidence_score}). {base_context}",
                "Swap Sector Detected": swap_flag,
                "Bad Sample Count": bad_sample_count,
                "P90 Serving Distance (m)": "" if pd.isna(p90_dist) else float(p90_dist),
                "Root Cause Category": dominant_root_cause,
                "Recommendation Status": az_status,
                "Recommendation Confidence": az_confidence_label,
                "Confidence Score": az_confidence_score,
                "Coverage ETilt Score": round(float(coverage_etilt_score), 2),
                "Overlap ETilt Score": round(float(overlap_etilt_score), 2),
                "Azimuth Score": round(float(azimuth_score), 2),
                "TX Power Score": round(float(tx_power_score), 2),
                "Bearing Peak Share": "" if pd.isna(bearing_peak_share) else round(float(bearing_peak_share), 4),
                "Bearing Spread Deg": "" if pd.isna(bearing_spread) else round(float(bearing_spread), 2),
                "Directional Contrast": "" if pd.isna(directional_contrast) else round(float(directional_contrast), 4),
            },
            {
                "Cell ID": cell_id,
                "Technology": ant_tech or tech,
                "Parameter": "TX Power",
                "Current Value": "" if pd.isna(curr_power) else float(curr_power),
                "Recommended Value": "" if pd.isna(rec_power) else float(rec_power),
                "Reason": f"{pwr_reason} Confidence={pwr_confidence_label}({pwr_confidence_score}). {base_context}",
                "Swap Sector Detected": swap_flag,
                "Bad Sample Count": bad_sample_count,
                "P90 Serving Distance (m)": "" if pd.isna(p90_dist) else float(p90_dist),
                "Root Cause Category": dominant_root_cause,
                "Recommendation Status": pwr_status,
                "Recommendation Confidence": pwr_confidence_label,
                "Confidence Score": pwr_confidence_score,
                "Coverage ETilt Score": round(float(coverage_etilt_score), 2),
                "Overlap ETilt Score": round(float(overlap_etilt_score), 2),
                "Azimuth Score": round(float(azimuth_score), 2),
                "TX Power Score": round(float(tx_power_score), 2),
                "Bearing Peak Share": "" if pd.isna(bearing_peak_share) else round(float(bearing_peak_share), 4),
                "Bearing Spread Deg": "" if pd.isna(bearing_spread) else round(float(bearing_spread), 2),
                "Directional Contrast": "" if pd.isna(directional_contrast) else round(float(directional_contrast), 4),
            },
        ])

    return pd.DataFrame(
        rec_rows,
        columns=[
            "Cell ID",
            "Technology",
            "Parameter",
            "Current Value",
            "Recommended Value",
            "Reason",
            "Swap Sector Detected",
            "Bad Sample Count",
            "P90 Serving Distance (m)",
            "Root Cause Category",
            "Recommendation Status",
            "Recommendation Confidence",
            "Confidence Score",
            "Coverage ETilt Score",
            "Overlap ETilt Score",
            "Azimuth Score",
            "TX Power Score",
            "Bearing Peak Share",
            "Bearing Spread Deg",
            "Directional Contrast",
        ],
    )


def _prepare_recommendation_exports(recommendations_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recommendations_df.empty:
        return recommendations_df.copy(), recommendations_df.copy()

    full_df = recommendations_df.copy()
    changed_mask = _changed_recommendation_mask(full_df)
    status = full_df["Recommendation Status"].astype(str).map(_norm_reason_token)
    keep_mask = changed_mask | status.isin({"blocked_by_blockage", "hold_swap"})
    filtered_df = full_df.loc[keep_mask].copy()
    return full_df, filtered_df


def _summarize_recommendation_run(
    config: TiltRecommendationTestConfig,
    log_df: pd.DataFrame,
    antenna_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    bad_samples_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    geo_cell_summary: pd.DataFrame,
    bearing_summary: pd.DataFrame,
    recommendations_all_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    swap_dict: Dict[str, str],
    run_dir: Path,
    excel_path: Path,
    total_runtime_sec: float,
) -> dict:
    swap_yes = int(sum(1 for v in swap_dict.values() if str(v).strip().upper() == "YES"))
    status_counts = (
        recommendations_all_df["Recommendation Status"].astype(str).map(_norm_reason_token).value_counts().to_dict()
        if not recommendations_all_df.empty else {}
    )
    root_cause_counts = (
        recommendations_all_df["Root Cause Category"].astype(str).map(_norm_reason_token).value_counts().to_dict()
        if not recommendations_all_df.empty else {}
    )
    confidence_counts = (
        recommendations_all_df["Recommendation Confidence"].astype(str).map(_norm_reason_token).value_counts().to_dict()
        if not recommendations_all_df.empty and "Recommendation Confidence" in recommendations_all_df.columns else {}
    )

    parameter_change_counts: dict[str, int] = {}
    top_parameter_changes: list[dict[str, object]] = []
    changed_rows = pd.DataFrame()
    if not recommendations_df.empty:
        changed_mask = _changed_recommendation_mask(recommendations_df)
        changed_rows = recommendations_df.loc[changed_mask].copy()
        if not changed_rows.empty:
            parameter_change_counts = changed_rows["Parameter"].astype(str).value_counts().to_dict()
            for _, rec in changed_rows.sort_values(["Bad Sample Count", "Cell ID"], ascending=[False, True]).head(12).iterrows():
                top_parameter_changes.append(
                    {
                        "cell_id": str(rec["Cell ID"]),
                        "parameter": str(rec["Parameter"]),
                        "current_value": rec["Current Value"],
                        "recommended_value": rec["Recommended Value"],
                        "bad_sample_count": int(pd.to_numeric(pd.Series([rec["Bad Sample Count"]]), errors="coerce").fillna(0).iloc[0]),
                        "root_cause_category": str(rec["Root Cause Category"]),
                        "recommendation_status": str(rec["Recommendation Status"]),
                        "reason": str(rec["Reason"]),
                    }
                )

    geo_context_overview = {}
    if not geo_cell_summary.empty:
        geo_context_overview = {
            "mean_serving_distance_m_range": {
                "min": round(float(pd.to_numeric(geo_cell_summary["mean_serving_distance_m"], errors="coerce").min()), 2),
                "max": round(float(pd.to_numeric(geo_cell_summary["mean_serving_distance_m"], errors="coerce").max()), 2),
            },
            "p90_serving_distance_m_range": {
                "min": round(float(pd.to_numeric(geo_cell_summary["p90_serving_distance_m"], errors="coerce").min()), 2),
                "max": round(float(pd.to_numeric(geo_cell_summary["p90_serving_distance_m"], errors="coerce").max()), 2),
            },
            "nlos_share_range": {
                "min": round(float(pd.to_numeric(geo_cell_summary["nlos_share"], errors="coerce").min()), 3),
                "max": round(float(pd.to_numeric(geo_cell_summary["nlos_share"], errors="coerce").max()), 3),
            },
            "los_blocked_ratio_range": {
                "min": round(float(pd.to_numeric(geo_cell_summary["mean_los_blocked_ratio"], errors="coerce").min()), 3),
                "max": round(float(pd.to_numeric(geo_cell_summary["mean_los_blocked_ratio"], errors="coerce").max()), 3),
            },
            "clutter_mode_counts": geo_cell_summary["clutter_mode"].astype(str).replace("", "UNKNOWN").value_counts().to_dict(),
        }

    top_bad_cells = []
    if not geo_cell_summary.empty:
        top_bad = geo_cell_summary.sort_values(["bad_sample_count", "p90_serving_distance_m"], ascending=[False, False]).head(10)
        for _, row in top_bad.iterrows():
            top_bad_cells.append(
                {
                    "cell_id": str(row["Cell ID"]),
                    "technology": str(row["Technology"]),
                    "bad_sample_count": int(pd.to_numeric(pd.Series([row["bad_sample_count"]]), errors="coerce").fillna(0).iloc[0]),
                    "mean_serving_distance_m": None if pd.isna(row["mean_serving_distance_m"]) else round(float(row["mean_serving_distance_m"]), 2),
                    "p90_serving_distance_m": None if pd.isna(row["p90_serving_distance_m"]) else round(float(row["p90_serving_distance_m"]), 2),
                    "nlos_share": None if pd.isna(row["nlos_share"]) else round(float(row["nlos_share"]), 3),
                    "mean_los_blocked_ratio": None if pd.isna(row["mean_los_blocked_ratio"]) else round(float(row["mean_los_blocked_ratio"]), 3),
                    "clutter_mode": str(row["clutter_mode"] or "UNKNOWN"),
                }
            )

    forecast_highlights = []
    if not forecast_df.empty:
        for _, row in forecast_df.sort_values(["Improvement %", "Pre-Change"], ascending=[False, False]).head(10).iterrows():
            forecast_highlights.append(
                {
                    "cell_id": str(row["Cell ID"]),
                    "kpi": str(row["KPI"]),
                    "pre_change": int(row["Pre-Change"]),
                    "est_post_change": int(row["Est. Post-Change"]),
                    "improvement_pct": float(row["Improvement %"]),
                    "forecast_type": "heuristic_not_rf_validated",
                }
            )

    return {
        "run_type": "tilt_recommendation_test",
        "project_id": int(config.project_id),
        "region": config.region,
        "operator": config.operator,
        "thresholds": {
            "rsrp": float(config.rsrp_threshold),
            "rsrq": float(config.rsrq_threshold),
            "sinr": float(config.sinr_threshold),
        },
        "logic_profile": {
            "min_bad_sample_count_for_action": MIN_BAD_SAMPLE_COUNT_FOR_ACTION,
            "medium_confidence_bad_sample_count": MEDIUM_CONFIDENCE_BAD_SAMPLE_COUNT,
            "high_confidence_bad_sample_count": HIGH_CONFIDENCE_BAD_SAMPLE_COUNT,
            "very_high_confidence_bad_sample_count": VERY_HIGH_CONFIDENCE_BAD_SAMPLE_COUNT,
            "far_edge_mean_serving_distance_m": FAR_EDGE_MEAN_SERVING_DISTANCE_M,
            "far_edge_p90_serving_distance_m": FAR_EDGE_P90_SERVING_DISTANCE_M,
            "medium_edge_mean_serving_distance_m": MEDIUM_EDGE_MEAN_SERVING_DISTANCE_M,
            "medium_edge_p90_serving_distance_m": MEDIUM_EDGE_P90_SERVING_DISTANCE_M,
            "high_nlos_share_gate": HIGH_NLOS_SHARE_GATE,
            "high_los_blocked_ratio_gate": HIGH_LOS_BLOCKED_RATIO_GATE,
            "high_building_area_ratio_gate": HIGH_BUILDING_AREA_RATIO_GATE,
            "medium_nlos_share_gate": MEDIUM_NLOS_SHARE_GATE,
            "medium_los_blocked_ratio_gate": MEDIUM_LOS_BLOCKED_RATIO_GATE,
            "dense_overlap_nearest_site_m": DENSE_OVERLAP_NEAREST_SITE_M,
            "dense_overlap_site_count_250m": DENSE_OVERLAP_SITE_COUNT_250M,
            "small_azimuth_delta_deg": SMALL_AZIMUTH_DELTA_DEG,
            "min_azimuth_mismatch_deg": MIN_AZIMUTH_MISMATCH_DEG,
            "max_azimuth_mismatch_deg": MAX_AZIMUTH_MISMATCH_DEG,
            "max_azimuth_step_deg": MAX_AZIMUTH_STEP_DEG,
            "min_bearing_sample_count": MIN_BEARING_SAMPLE_COUNT,
            "max_bearing_spread_deg": MAX_BEARING_SPREAD_DEG,
            "azimuth_nlos_hard_block_gate": AZIMUTH_NLOS_HARD_BLOCK_GATE,
            "relaxed_azimuth_bad_sample_count": RELAXED_AZIMUTH_BAD_SAMPLE_COUNT,
            "relaxed_azimuth_peak_share": RELAXED_AZIMUTH_PEAK_SHARE,
            "relaxed_azimuth_max_spread_deg": RELAXED_AZIMUTH_MAX_SPREAD_DEG,
            "min_directional_contrast_for_azimuth": MIN_DIRECTIONAL_CONTRAST_FOR_AZIMUTH,
            "strong_directional_contrast_for_azimuth": STRONG_DIRECTIONAL_CONTRAST_FOR_AZIMUTH,
            "min_candidate_score_gap": MIN_CANDIDATE_SCORE_GAP,
            "site_symmetry_penalty_bad_sample_ratio": SITE_SYMMETRY_PENALTY_BAD_SAMPLE_RATIO,
            "min_safe_etilt_deg": MIN_SAFE_ETILT_DEG,
            "max_safe_etilt_deg": MAX_SAFE_ETILT_DEG,
            "max_etilt_increase_per_run_deg": MAX_ETILT_INCREASE_PER_RUN_DEG,
            "max_etilt_decrease_per_run_deg": MAX_ETILT_DECREASE_PER_RUN_DEG,
            "coverage_etilt_action_score": COVERAGE_ETILT_ACTION_SCORE,
            "overlap_etilt_action_score": OVERLAP_ETILT_ACTION_SCORE,
            "azimuth_action_score": AZIMUTH_ACTION_SCORE,
            "tx_power_action_score": TX_POWER_ACTION_SCORE,
        },
        "model_explanation": {
            "what_this_test_is_doing": (
                "This test harness reuses the current baseline prediction output, current antenna configuration, "
                "and saved geo-feature table to build a geo-aware recommendation layer. It now assigns confidence "
                "scores and can surface moderate-confidence actions for persistent degradation clusters, but it still "
                "does not run a fresh RF recompute for each recommendation candidate."
            ),
            "reused_source_functions": [
                "tools.lte_prediction_optimised.ml_engine.fetch_baseline",
                "tools.lte_prediction_optimised.ml_engine.fetch_site_data",
                "tools.lte_prediction_optimised.ml_engine.fetch_geo_features",
                "tools.lte_tilt_recommandation.etilt_optimizer_cd2.filter_bad_samples",
                "tools.lte_tilt_recommandation.etilt_optimizer_cd2.detect_swap_sector",
                "tools.lte_tilt_recommandation.etilt_optimizer_cd2.build_forecast",
                "tools.lte_tilt_recommandation.etilt_optimizer_cd2.export_to_excel",
            ],
            "decision_flow": [
                "Load current-state baseline prediction rows from the DB.",
                "Load current site and antenna settings from site_prediction.",
                "Load saved baseline geo features from lte_prediction_geo_features.",
                "Filter bad KPI samples using the configured RSRP/RSRQ/SINR thresholds.",
                "Attach geo context only to the bad-sample rows.",
                "Aggregate bad-sample geo context per cell.",
                "Apply geo-aware recommendation rules for ETilt, azimuth, and TX power.",
                "Score each candidate with parameter-specific confidence so persistent medium-geometry problems can surface as moderate-confidence actions.",
                "Rank bounded ETilt and azimuth candidates before final recommendation output.",
                "Estimate expected gain with the source heuristic forecast function.",
            ],
            "what_the_model_is_not_doing_yet": [
                "It is not validating each candidate with a localized optimized RF recomputation loop.",
                "It is not ranking multiple parameter candidates through true KPI recompute and geo-corrected comparison.",
                "The forecast highlights are heuristic estimates from the reused source function and are not RF-validated recompute results.",
            ],
        },
        "counts": {
            "baseline_rows": int(len(log_df)),
            "antenna_rows": int(len(antenna_df)),
            "geo_rows": int(len(geo_df)),
            "bad_samples": int(len(bad_samples_df)),
            "bad_cells": int(summary_df["Cell ID"].nunique()) if not summary_df.empty else 0,
            "bearing_cells": int(bearing_summary["Cell ID"].nunique()) if not bearing_summary.empty else 0,
            "geo_context_cells": int(geo_cell_summary["Cell ID"].nunique()) if not geo_cell_summary.empty else 0,
            "recommendation_rows_all": int(len(recommendations_all_df)),
            "recommendation_rows_actionable": int(len(changed_rows)),
            "recommendation_rows_exported": int(len(recommendations_df)),
            "forecast_rows": int(len(forecast_df)),
            "swap_sector_yes": swap_yes,
        },
        "recommendation_summary": {
            "status_counts": status_counts,
            "root_cause_counts": root_cause_counts,
            "confidence_counts": confidence_counts,
            "parameter_change_counts": parameter_change_counts,
            "top_parameter_changes": top_parameter_changes,
        },
        "geo_context_overview": geo_context_overview,
        "top_bad_cells": top_bad_cells,
        "forecast_highlights": forecast_highlights,
        "artifacts": {
            "baseline_log_input": str(run_dir / "baseline_log_input.csv"),
            "antenna_input": str(run_dir / "antenna_input.csv"),
            "geo_features_input": str(run_dir / "geo_features_input.csv"),
            "bad_samples": str(run_dir / "bad_samples.csv"),
            "bad_samples_with_geo": str(run_dir / "bad_samples_with_geo.csv"),
            "bad_summary": str(run_dir / "bad_summary.csv"),
            "bad_geo_cell_summary": str(run_dir / "bad_geo_cell_summary.csv"),
            "dominant_bearing_summary": str(run_dir / "dominant_bearing_summary.csv"),
            "recommendations_all": str(run_dir / "recommendations_all.csv"),
            "recommendations": str(run_dir / "recommendations.csv"),
            "forecast": str(run_dir / "forecast.csv"),
            "excel_report": str(excel_path),
        },
        "total_runtime_sec": round(float(total_runtime_sec), 4),
    }


def run_tilt_recommendation_test(config: TiltRecommendationTestConfig) -> Path:
    start = time.perf_counter()
    run_dir = _ensure_dir(config.output_root / f"project_{config.project_id}" / f"tilt_{_timestamp()}")

    TILT_SRC.RSRP_THRESH = float(config.rsrp_threshold)
    TILT_SRC.RSRQ_THRESH = float(config.rsrq_threshold)
    TILT_SRC.SINR_THRESH = float(config.sinr_threshold)

    print(
        f"[TILT_TEST][START] project_id={config.project_id} region={config.region} "
        f"operator={config.operator} rsrp={config.rsrp_threshold} "
        f"rsrq={config.rsrq_threshold} sinr={config.sinr_threshold}"
    )

    log_df = _fetch_baseline_log_df(config.project_id, config.region, config.operator)
    antenna_df = _fetch_antenna_df(config.project_id, config.region, config.operator)
    log_df = _enrich_log_with_antenna_context(log_df, antenna_df)
    geo_df = _fetch_geo_df(config.project_id, config.region, config.operator, antenna_df)

    bad_samples_df, summary_df = TILT_SRC.filter_bad_samples(log_df.copy(), TILT_SRC.ALLOWED_TECHS)
    swap_dict = TILT_SRC.detect_swap_sector(log_df.copy(), antenna_df.copy())
    bearing_summary = _compute_dominant_bearing_summary(log_df, antenna_df)
    bad_geo_df = _attach_geo_to_bad_samples(bad_samples_df, geo_df)
    geo_cell_summary = _aggregate_bad_geo_context(bad_geo_df)
    recommendations_all_df = _build_geo_aware_recommendations(
        summary_df,
        antenna_df,
        swap_dict,
        geo_cell_summary,
        bearing_summary,
    )
    recommendations_all_df, recommendations_df = _prepare_recommendation_exports(recommendations_all_df)
    forecast_df = TILT_SRC.build_forecast(summary_df, recommendations_all_df)

    excel_path = run_dir / "RF_Optimization_Report_Test.xlsx"
    TILT_SRC.export_to_excel(
        summary_df=summary_df,
        recommendations_df=recommendations_df,
        forecast_df=forecast_df,
        bad_samples_df=bad_geo_df,
        output_path=str(excel_path),
    )

    log_df.to_csv(run_dir / "baseline_log_input.csv", index=False)
    antenna_df.to_csv(run_dir / "antenna_input.csv", index=False)
    geo_df.to_csv(run_dir / "geo_features_input.csv", index=False)
    bad_samples_df.to_csv(run_dir / "bad_samples.csv", index=False)
    bad_geo_df.to_csv(run_dir / "bad_samples_with_geo.csv", index=False)
    summary_df.to_csv(run_dir / "bad_summary.csv", index=False)
    geo_cell_summary.to_csv(run_dir / "bad_geo_cell_summary.csv", index=False)
    bearing_summary.to_csv(run_dir / "dominant_bearing_summary.csv", index=False)
    recommendations_all_df.to_csv(run_dir / "recommendations_all.csv", index=False)
    recommendations_df.to_csv(run_dir / "recommendations.csv", index=False)
    forecast_df.to_csv(run_dir / "forecast.csv", index=False)

    total_runtime_sec = time.perf_counter() - start
    summary_payload = _summarize_recommendation_run(
        config=config,
        log_df=log_df,
        antenna_df=antenna_df,
        geo_df=geo_df,
        bad_samples_df=bad_samples_df,
        summary_df=summary_df,
        geo_cell_summary=geo_cell_summary,
        bearing_summary=bearing_summary,
        recommendations_all_df=recommendations_all_df,
        recommendations_df=recommendations_df,
        forecast_df=forecast_df,
        swap_dict=swap_dict,
        run_dir=run_dir,
        excel_path=excel_path,
        total_runtime_sec=total_runtime_sec,
    )
    _write_json(run_dir / "summary.json", summary_payload)
    print(f"[TILT_TEST][DONE] run_dir={run_dir} total_runtime_sec={summary_payload['total_runtime_sec']}")
    return run_dir


def _parse_args() -> TiltRecommendationTestConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--operator", type=str, default=None)
    parser.add_argument("--rsrp", type=float, default=-105.0)
    parser.add_argument("--rsrq", type=float, default=-15.0)
    parser.add_argument("--sinr", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    return TiltRecommendationTestConfig(
        project_id=args.project_id,
        region=args.region,
        operator=args.operator,
        rsrp_threshold=args.rsrp,
        rsrq_threshold=args.rsrq,
        sinr_threshold=args.sinr,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    run_dir = run_tilt_recommendation_test(_parse_args())
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
