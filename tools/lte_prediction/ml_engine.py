import os

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from .Sector_wise_prediction_code_copy import run_prediction_from_api
from .lte_ml_correction_final import run_ml_from_api


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
    print(f"[LTE][{stage}] source_table={table_name}")
    print(f"[LTE][{stage}] filters={filters}")
    print(f"[LTE][{stage}] row_count={len(df)}")
    print(f"[LTE][{stage}] columns={list(df.columns)}")
    if extra:
        for key, value in extra.items():
            print(f"[LTE][{stage}] {key}={value}")


def _print_df_profile(stage, df):
    pci_col = "PCI" if "PCI" in df.columns else "pci"
    print(f"[LTE][{stage}] df_shape={df.shape}")
    print(
        f"[LTE][{stage}] unique_cell_id={_safe_nunique(df, 'cell_id')} "
        f"unique_pci={_safe_nunique(df, pci_col)} "
        f"unique_nodeb_id={_safe_nunique(df, 'nodeb_id')}"
    )
    print(
        f"[LTE][{stage}] lat_range={_safe_minmax(df, 'lat')} "
        f"lon_range={_safe_minmax(df, 'lon')}"
    )
    if "network" in df.columns:
        top_network = df["network"].astype(str).value_counts(dropna=False).head(5).to_dict()
        print(f"[LTE][{stage}] top_network_counts={top_network}")


def _resolve_operator(df):
    for col in ("network", "cluster", "operator", "Technology"):
        if col in df.columns:
            series = df[col].dropna().astype(str).str.strip()
            series = series[series != ""]
            if not series.empty:
                operator = series.mode().iloc[0]
                print(f"[LTE][SITE_FETCH] resolved_operator_from={col} value={operator}")
                return operator
    raise ValueError("Unable to resolve operator from site_prediction data")


def fetch_site_data(project_id, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT *
    FROM site_prediction
    WHERE tbl_project_id = {project_id}
    """

    df = pd.read_sql(query, current_engine)

    if df.empty:
        raise ValueError("No site data found")

    _print_fetch_summary(
        "SITE_FETCH_RAW",
        "site_prediction",
        {"project_id": project_id, "region": region},
        df,
        extra={
            "distinct_pci": _safe_nunique(df, "pci"),
            "distinct_cell_id": _safe_nunique(df, "cell_id"),
            "distinct_nodeb_id": _safe_nunique(df, "nodeb_id"),
            "non_null_cluster": _safe_non_null(df, "cluster"),
        }
    )

    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "e_tilt": "Etilt",
        "m_tilt": "Mtilt",
        "height": "Height",
        "cell_id": "cell_id",
        "nodeb_id": "nodeb_id",
        "azimuth": "azimuth",
        "pci": "PCI",
        "rsrp": "rsrp",
        "tx_power": "tx_power",
        "band": "band",
        "earfcn": "earfcn",
        "reference_signal_power": "reference_signal_power",
        "site": "Site ID",
        "cluster": "network"
    })

    required = ["lat", "lon", "Etilt", "Mtilt", "Height", "tx_power"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["Etilt"] = pd.to_numeric(df["Etilt"], errors="coerce").fillna(3)
    df["Mtilt"] = pd.to_numeric(df["Mtilt"], errors="coerce").fillna(0)
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce").fillna(30)
    df["tx_power"] = pd.to_numeric(df["tx_power"], errors="coerce").fillna(46)

    df = df.dropna(subset=["lat", "lon"])

    print("[LTE][SITE_FETCH_READY] converted_to_prediction_engine_format=True")
    _print_df_profile("SITE_FETCH_READY", df)
    return df, _resolve_operator(df)


def fetch_drive_data(session_ids, operator, region="india"):
    session_str = ",".join(map(str, session_ids))
    key = f"{operator}_{session_str}"
    path = f"cache/drive_{key}.parquet"

    if os.path.exists(path):
        print("[LTE][DRIVE_FETCH_CACHE] cache_hit=True")
        cached = pd.read_parquet(path)
        _print_fetch_summary(
            "DRIVE_FETCH_CACHE",
            "cache/drive parquet",
            {"session_ids": session_ids, "operator": operator, "region": region},
            cached,
            extra={
                "distinct_sessions_requested": len(session_ids),
                "lat_range": _safe_minmax(cached, "lat"),
                "lon_range": _safe_minmax(cached, "lon"),
            }
        )
        return cached

    current_engine = engine.get(region.lower(), engine["india"])

    main_query = f"""
    SELECT lat, lon, rsrp, rsrq, sinr
    FROM tbl_network_log
    WHERE session_id IN ({session_str})
    AND LOWER(COALESCE(m_alpha_long, m_alpha_short)) = LOWER('{operator}')
    """

    neighbour_query = f"""
    SELECT lat, lon, rsrp, rsrq, sinr
    FROM tbl_network_log_neighbour
    WHERE session_id IN ({session_str})
    AND LOWER(COALESCE(m_alpha_long, m_alpha_short)) = LOWER('{operator}')
    """

    main_df = pd.read_sql(main_query, current_engine)
    neighbour_df = pd.read_sql(neighbour_query, current_engine)

    _print_fetch_summary(
        "DRIVE_FETCH_MAIN",
        "tbl_network_log",
        {"session_ids": session_ids, "operator": operator, "region": region},
        main_df,
        extra={"distinct_sessions_requested": len(session_ids)}
    )
    _print_fetch_summary(
        "DRIVE_FETCH_NEIGHBOUR",
        "tbl_network_log_neighbour",
        {"session_ids": session_ids, "operator": operator, "region": region},
        neighbour_df,
        extra={"distinct_sessions_requested": len(session_ids)}
    )

    df = pd.concat([main_df, neighbour_df], ignore_index=True)
    _print_df_profile("DRIVE_FETCH_COMBINED", df)

    os.makedirs("cache", exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_building_data(project_id, region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT *
    FROM tbl_savepolygon
    WHERE project_id = {project_id}
    """

    df = pd.read_sql(query, current_engine)
    _print_fetch_summary(
        "BUILDING_FETCH",
        "tbl_savepolygon",
        {"project_id": project_id, "region": region},
        df,
        extra={
            "distinct_project_id": _safe_nunique(df, "project_id"),
            "non_null_region": _safe_non_null(df, "region"),
        }
    )
    return df


def fetch_polygon_data(project_id):
    return {
        "type": "Polygon",
        "coordinates": [[[77.1, 28.6], [77.2, 28.6], [77.2, 28.7], [77.1, 28.7], [77.1, 28.6]]]
    }


def run_rf_prediction_fast(site_df, drive_df, building_df, params):
    temp_dir = "temp_rf"
    os.makedirs(temp_dir, exist_ok=True)

    site_path = f"{temp_dir}/site.csv"
    drive_path = f"{temp_dir}/drive.csv"
    building_path = f"{temp_dir}/building.csv"

    site_df.to_csv(site_path, index=False)
    drive_df.to_csv(drive_path, index=False)
    building_df.to_csv(building_path, index=False)

    print(
        f"[LTE][RF_INPUT] site_rows={len(site_df)} drive_rows={len(drive_df)} "
        f"building_rows={len(building_df)} radius={params['radius']} "
        f"grid={params['grid']} workers={params['workers']}"
    )
    print(
        f"[LTE][RF_INPUT] unique_cells={_safe_nunique(site_df, 'cell_id')} "
        f"unique_pci={_safe_nunique(site_df, 'PCI') if 'PCI' in site_df.columns else _safe_nunique(site_df, 'pci')}"
    )

    run_prediction_from_api({
        "site": site_path,
        "drive": drive_path,
        "building": building_path,
        "polygon_area": None,
        "radius": params["radius"],
        "grid_resolution": params["grid"],
        "frequency": params.get("frequency_mhz", 1800),
        "bandwidth": params.get("bandwidth_mhz", 10),
        "antenna_gain": params.get("antenna_gain", 18),
        "cable_loss": params.get("cable_loss", 2),
        "ue_height": params.get("ue_height", 1.5),
        "outdir": temp_dir,
        "n_workers": params["workers"],
        "calibrate": True
    })

    pred_df = pd.read_csv(f"{temp_dir}/prediction_ALL_SITES.csv")
    _print_fetch_summary(
        "RF_OUTPUT",
        "temp_rf/prediction_ALL_SITES.csv",
        {"radius": params["radius"], "grid": params["grid"]},
        pred_df,
        extra={
            "unique_predicted_cells": _safe_nunique(pred_df, "Node_Cell_ID"),
            "pred_rsrp_range": _safe_minmax(pred_df, "pred_rsrp"),
            "pred_rsrq_range": _safe_minmax(pred_df, "pred_rsrq"),
            "pred_sinr_range": _safe_minmax(pred_df, "pred_sinr"),
        }
    )
    return pred_df


def run_ml_fast(pred_df, drive_df):
    print(
        f"[LTE][ML_INPUT] pred_rows={len(pred_df)} drive_rows={len(drive_df)} "
        f"pred_cols={list(pred_df.columns)} drive_cols={list(drive_df.columns)}"
    )
    return run_ml_from_api(pred_df, drive_df)


def grid_drive_test(input_file, output_file):
    df = pd.read_csv(input_file)
    df.to_csv(output_file, index=False)
