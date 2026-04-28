import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# 🔥 Import your main RF functions
from .Sector_wise_prediction_code_copy import (
    calibrate_site,
    compute_predictions_parallel,
    generate_grid
)
from .Sector_wise_prediction_code_copy import run_prediction_from_api

# ==========================================================
# DB CONNECTION
# ==========================================================

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


# ==========================================================
# FETCH BASELINE (AS DRIVE TEST)
# ==========================================================

def fetch_baseline(project_id, region="india"):

    current_engine = engine.get(region.lower(), engine["india"])

    query = f"""
    SELECT lat, lon, pred_rsrp as rsrp, cell_id
    FROM lte_prediction_baseline_results
    WHERE project_id = {project_id}
    """

    df = pd.read_sql(query, current_engine)

    df["Node_Cell_ID"] = df["cell_id"].astype(str)

    return df


# ==========================================================
# FETCH ORIGINAL SITE DATA
# ==========================================================

def fetch_site_data(project_id,region="india"):
    current_engine = engine.get(region.lower(), engine["india"])
    query = f"""
    SELECT *
    FROM site_prediction
    WHERE tbl_project_id = {project_id}
    """

    df = pd.read_sql(query, current_engine)

    # 🔥 MATCH YOUR SCRIPT FORMAT
    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",
        "e_tilt": "Etilt",
        "m_tilt": "Mtilt",
        "height": "Height"
    })

    df["Node_Cell_ID"] = df["cell_id"].astype(str)

    # default frequency
    if "frequency_mhz" not in df.columns:
        df["frequency_mhz"] = 1800

    return df


# ==========================================================
# FETCH OPTIMIZED SITE DATA
# ==========================================================

# ==========================================================
# FETCH OPTIMIZED SITE DATA (OPERATOR-WISE)
# ==========================================================

def fetch_optimized_sites(project_id, operator,region="india"):
    current_engine = engine.get(region.lower(), engine["india"])

    # 🔥 Much simpler query! No JOIN needed since the operator is right here.
    query = f"""
    SELECT *
    FROM site_prediction_optimized 
    WHERE tbl_project_id = {project_id}
    AND cluster_name = '{operator}'
    """

    df = pd.read_sql(query, current_engine)

    df = df.rename(columns={
        "latitude": "lat",
        "longitude": "lon",

        # 🔥 CRITICAL FIX
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
            raise ValueError(f"❌ Missing required column: {col}")

    # ✅ CLEAN STRING (VERY IMPORTANT)
    df["cell_id"] = df["cell_id"].astype(str).str.strip()

    # ✅ Use directly (already combined ID)
    df["Node_Cell_ID"] = df["cell_id"]

    print(f"✅ Filtered for Operator: {operator}")
    print("✅ Total rows:", len(df))
    print("✅ Total cells:", df["Node_Cell_ID"].nunique())
    print("Sample IDs:", df["Node_Cell_ID"].unique()[:5])

    # default frequency
    if "frequency_mhz" not in df.columns:
        df["frequency_mhz"] = 1800

    return df


# ==========================================================
# K1 K2 CALCULATION
# ==========================================================

def compute_k1k2(baseline_df, site_df):

    k1k2_map = {}

    for cid in site_df["Node_Cell_ID"].unique():

        site_rows = site_df[site_df["Node_Cell_ID"] == cid]
        dt_rows   = baseline_df[baseline_df["Node_Cell_ID"] == cid]

        if len(dt_rows) < 10:
            continue

        freq = site_rows["frequency_mhz"].iloc[0]

        k1, k2 = calibrate_site(
            dt_rows,
            site_rows,
            site_rows["tx_power"].iloc[0],
            18, 2, freq
        )

        k1k2_map[cid] = (k1, k2)

    return k1k2_map


# ==========================================================
# OPTIMIZED PREDICTION ONLY
# ==========================================================

def run_prediction_only_optimized(opt_sites, k1k2_map, params):

    final_list = []

    # 🔥 ALL sites for interference
    opt_site_records = opt_sites.to_dict("records")

    unique_cells = opt_sites["Node_Cell_ID"].unique()

    print(f"🚀 Total cells to process: {len(unique_cells)}")

    for cid in unique_cells:

        print(f"\n⚡ Running optimized cell: {cid}")

        site_rows = opt_sites[opt_sites["Node_Cell_ID"] == cid]

        k1, k2 = k1k2_map.get(cid, (0, 0))

        if k1 != 0:
            print(f"   ✔ Using K1={k1:.2f}, K2={k2:.2f}")
        else:
            print(f"   ⚠ Using COST231")

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

        print(f"   📍 Grid points: {len(pts)}")

        # ⏱ START TIMER
        import time
        start = time.time()

        rsrp, rsrq, sinr = compute_predictions_parallel(
            pts,
            site_rows,
            cell_params,
            n_workers=cell_params.get("n_workers")
        )

        print(f"   ⏱ Time taken: {round(time.time() - start, 2)} sec")

        pts["pred_rsrp"] = rsrp
        pts["pred_rsrq"] = rsrq
        pts["pred_sinr"] = sinr
        pts["Node_Cell_ID"] = cid

        final_list.append(pts)

        print(f"✅ Completed cell: {cid}")   # 🔥 KEY LINE

    return pd.concat(final_list, ignore_index=True)


# ==========================================================
# REPLACE BASELINE CELLS
# ==========================================================

def replace_cells(baseline_df, optimized_df):

    replace_ids = optimized_df["Node_Cell_ID"].unique()

    baseline_df = baseline_df[
        ~baseline_df["Node_Cell_ID"].isin(replace_ids)
    ]

    final_df = pd.concat([baseline_df, optimized_df], ignore_index=True)

    return final_df
