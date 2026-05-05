import os
import time

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from .Sector_wise_prediction_code_copy import (
    calibrate_site,
    compute_predictions_parallel,
    generate_grid,
)


load_dotenv()

engine = {
    "india": create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL") else None,
    "taiwan": create_engine(
        os.getenv("DATABASE_URL_Taiwan"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}


def _safe_nunique(df, col):
    return int(df[col].nunique(dropna=True)) if col in df.columns else "n/a"


def _safe_non_null(df, col):
    return int(df[col].notna().sum()) if col in df.columns else "n/a"


def _safe_minmax(df, col):
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.4f}..{series.max():.4f}"


def _print_fetch_summary(stage, table_name, filters, df, extra=None):
    print(f"[LTE_OPT][{stage}] source_table={table_name}")
    print(f"[LTE_OPT][{stage}] filters={filters}")
    print(f"[LTE_OPT][{stage}] row_count={len(df)}")
    print(f"[LTE_OPT][{stage}] columns={list(df.columns)}")
    if extra:
        for key, value in extra.items():
            print(f"[LTE_OPT][{stage}] {key}={value}")


def fetch_baseline(project_id, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT lat, lon, pred_rsrp as rsrp, cell_id
    FROM lte_prediction_baseline_results
    WHERE project_id = {project_id}
    """

    df = pd.read_sql(query, current_engine)
    df["Node_Cell_ID"] = df["cell_id"].astype(str)
    _print_fetch_summary(
        "BASELINE_FETCH",
        "lte_prediction_baseline_results",
        {"project_id": project_id, "region": region},
        df,
        extra={
            "distinct_cell_id": _safe_nunique(df, "cell_id"),
            "distinct_node_cell_id": _safe_nunique(df, "Node_Cell_ID"),
            "rsrp_range": _safe_minmax(df, "rsrp"),
        }
    )
    return df


def fetch_site_data(project_id, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT *
    FROM site_prediction
    WHERE tbl_project_id = {project_id}
    """

    df = pd.read_sql(query, current_engine)
    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "e_tilt": "Etilt",
        "m_tilt": "Mtilt",
        "height": "Height"
    })

    df["Node_Cell_ID"] = df["cell_id"].astype(str)
    if "frequency_mhz" not in df.columns:
        df["frequency_mhz"] = 1800

    _print_fetch_summary(
        "SITE_FETCH",
        "site_prediction",
        {"project_id": project_id, "region": region},
        df,
        extra={
            "distinct_pci": _safe_nunique(df, "pci"),
            "distinct_cell_id": _safe_nunique(df, "cell_id"),
            "distinct_nodeb_id": _safe_nunique(df, "nodeb_id"),
        }
    )
    return df


def fetch_optimized_sites(project_id, operator, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT *
    FROM site_prediction_optimized
    WHERE tbl_project_id = {project_id}
    AND cluster = '{operator}'
    """

    df = pd.read_sql(query, current_engine)
    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "e_tilt": "electrical_tilt",
        "m_tilt": "mechanical_tilt",
        "height": "antenna_height"
    })

    required_cols = [
        "lat", "lon", "azimuth", "tx_power",
        "electrical_tilt", "mechanical_tilt", "antenna_height"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df["Node_Cell_ID"] = df["cell_id"]
    if "frequency_mhz" not in df.columns:
        df["frequency_mhz"] = 1800

    _print_fetch_summary(
        "OPTIMIZED_SITE_FETCH",
        "site_prediction_optimized",
        {"project_id": project_id, "operator": operator, "region": region},
        df,
        extra={
            "distinct_pci": _safe_nunique(df, "pci"),
            "distinct_cell_id": _safe_nunique(df, "cell_id"),
            "distinct_nodeb_id": _safe_nunique(df, "nodeb_id"),
            "non_null_cluster": _safe_non_null(df, "cluster"),
        }
    )
    return df


def compute_k1k2(baseline_df, site_df):
    k1k2_map = {}
    total_cells = int(site_df["Node_Cell_ID"].nunique())
    print(f"[LTE_OPT][K1K2] total_site_cells={total_cells}")

    for cid in site_df["Node_Cell_ID"].unique():
        site_rows = site_df[site_df["Node_Cell_ID"] == cid]
        dt_rows = baseline_df[baseline_df["Node_Cell_ID"] == cid]
        print(
            f"[LTE_OPT][K1K2] cell={cid} site_rows={len(site_rows)} "
            f"baseline_rows={len(dt_rows)}"
        )

        if len(dt_rows) < 10:
            print(f"[LTE_OPT][K1K2] cell={cid} skipped_reason=baseline_rows_lt_10")
            continue

        freq = site_rows["frequency_mhz"].iloc[0]
        k1, k2 = calibrate_site(
            dt_rows,
            site_rows,
            site_rows["tx_power"].iloc[0],
            18,
            2,
            freq
        )
        k1k2_map[cid] = (k1, k2)
        print(f"[LTE_OPT][K1K2] cell={cid} calibrated_k1={k1:.4f} calibrated_k2={k2:.4f}")

    print(f"[LTE_OPT][K1K2] calibrated_cells={len(k1k2_map)}")
    return k1k2_map


def run_prediction_only_optimized(opt_sites, k1k2_map, params):
    final_list = []
    opt_site_records = opt_sites.to_dict("records")
    unique_cells = opt_sites["Node_Cell_ID"].unique()

    print(f"[LTE_OPT][RUN] total_cells_to_process={len(unique_cells)}")

    for cid in unique_cells:
        print(f"[LTE_OPT][RUN] cell_start={cid}")
        site_rows = opt_sites[opt_sites["Node_Cell_ID"] == cid]
        k1, k2 = k1k2_map.get(cid, (0, 0))

        print(
            f"[LTE_OPT][RUN] cell={cid} site_rows={len(site_rows)} "
            f"k1={k1} k2={k2} radius={params.get('radius')} "
            f"grid_resolution={params.get('grid_resolution')}"
        )

        cell_params = params.copy()
        cell_params.update({
            "k1": k1,
            "k2": k2,
            "all_sites_rows": opt_site_records
        })

        pts = generate_grid(
            site_rows,
            cell_params["radius"],
            cell_params["grid_resolution"]
        )
        print(f"[LTE_OPT][RUN] cell={cid} grid_points={len(pts)}")

        start = time.time()
        rsrp, rsrq, sinr = compute_predictions_parallel(
            pts,
            site_rows,
            cell_params,
            n_workers=cell_params.get("n_workers")
        )
        elapsed = round(time.time() - start, 2)

        pts["pred_rsrp"] = rsrp
        pts["pred_rsrq"] = rsrq
        pts["pred_sinr"] = sinr
        pts["Node_Cell_ID"] = cid

        print(
            f"[LTE_OPT][RUN] cell={cid} elapsed_sec={elapsed} "
            f"pred_rsrp_range={_safe_minmax(pts, 'pred_rsrp')} "
            f"pred_rsrq_range={_safe_minmax(pts, 'pred_rsrq')} "
            f"pred_sinr_range={_safe_minmax(pts, 'pred_sinr')}"
        )
        final_list.append(pts)

    final_df = pd.concat(final_list, ignore_index=True)
    _print_fetch_summary(
        "OPTIMIZED_RF_OUTPUT",
        "in_memory_optimized_prediction",
        {"cells_processed": len(unique_cells)},
        final_df,
        extra={
            "distinct_node_cell_id": _safe_nunique(final_df, "Node_Cell_ID"),
            "pred_rsrp_range": _safe_minmax(final_df, "pred_rsrp"),
            "pred_rsrq_range": _safe_minmax(final_df, "pred_rsrq"),
            "pred_sinr_range": _safe_minmax(final_df, "pred_sinr"),
        }
    )
    return final_df


def replace_cells(baseline_df, optimized_df):
    replace_ids = optimized_df["Node_Cell_ID"].unique()
    baseline_df = baseline_df[
        ~baseline_df["Node_Cell_ID"].isin(replace_ids)
    ]
    final_df = pd.concat([baseline_df, optimized_df], ignore_index=True)
    print(
        f"[LTE_OPT][REPLACE] replace_cell_count={len(replace_ids)} "
        f"remaining_baseline_rows={len(baseline_df)} final_rows={len(final_df)}"
    )
    return final_df
