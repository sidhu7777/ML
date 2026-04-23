# ============================================================
# services.py — Full LTE Prediction Pipeline (Final Version)
# ============================================================

import os
import json
import math
import uuid
import shutil
import numpy as np
import pandas as pd
from typing import List
import joblib
import gc
import re

# Geometry (optional)
try:
    from shapely import wkt
    from shapely.geometry import Point
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

# ML Imports
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

# Optional boosted models
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False


# ============================================================
# KPI Ranges
# ============================================================

KPI_RANGES = {
    "RSRP": (-140.0, -44.0),
    "RSRQ": (-19.5, -3.0),
    "SINR": (0.0, 30.0),
}


# ============================================================
# Helper Functions
# ============================================================

def clamp_array(arr, kpi_name):
    lo, hi = KPI_RANGES[kpi_name]
    return np.clip(arr, lo, hi)


def normcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df


def standardize_latlon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Your REAL column names
    if "lat_pred" in df.columns and "lon_pred" in df.columns:
        df = df.rename(columns={
            "lat_pred": "lat",
            "lon_pred": "lon"
        })

    return df


def to_rad(df: pd.DataFrame) -> np.ndarray:
    return np.radians(np.c_[df["lat"].values, df["lon"].values])


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2.0)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))


def make_regressor():
    if XGB_OK:
        return XGBRegressor(
            n_estimators=900, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, reg_alpha=0.0,
            tree_method="hist", random_state=42, n_jobs=-1
        )

    if LGB_OK:
        return lgb.LGBMRegressor(
            n_estimators=1400, num_leaves=63, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42
        )

    try:
        return GradientBoostingRegressor(random_state=42)
    except:
        return RandomForestRegressor(
            n_estimators=600, random_state=42, n_jobs=-1
        )


def build_preprocess(num_features, cat_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ],
        remainder="drop"
    )


# ============================================================
# FAST MATCH — nearest site-sector for each point
# ============================================================

def fast_match(work_site, points_df):
    if len(work_site) == 0:
        raise RuntimeError("work_site empty")

    site_rad = to_rad(work_site)

    n_neighbors = min(10, len(work_site))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="haversine")
    nbrs.fit(site_rad)

    pts_rad = to_rad(points_df)
    distances, indices = nbrs.kneighbors(pts_rad)
    distances_m = distances * 6371000.0

    best_idx, best_bearing, best_delta, best_dist = [], [], [], []

    for i in range(points_df.shape[0]):
        lat = points_df.iloc[i]["lat"]
        lon = points_df.iloc[i]["lon"]

        idxs = indices[i]
        candidate_sites = work_site.iloc[idxs]

        phi1 = np.radians(candidate_sites["lat"].values)
        phi2 = math.radians(lat)
        dlmb = np.radians(lon - candidate_sites["lon"].values)

        x = np.sin(dlmb) * np.cos(phi2)
        y = (
            np.cos(phi1) * np.sin(phi2)
            - np.sin(phi1) * np.cos(phi2) * np.cos(dlmb)
        )

        brngs = (np.degrees(np.arctan2(x, y)) + 360) % 360
        deltas = np.abs(
            (candidate_sites["azimuth"].values - brngs + 180) % 360 - 180
        )

        order = np.lexsort((distances_m[i], deltas))
        j = order[0]

        best_idx.append(idxs[j])
        best_bearing.append(brngs[j])
        best_delta.append(deltas[j])
        best_dist.append(distances_m[i][j])

    sel = work_site.iloc[np.array(best_idx)].reset_index(drop=True)

    return pd.DataFrame({
        "best_site": sel["SiteName"].astype(str),
        "best_sector": sel["sector"],
        "site_lat": sel["lat"],
        "site_lon": sel["lon"],
        "site_az": sel["azimuth"],
        "dist_m": np.array(best_dist),
        "bearing_tx_to_ue": np.array(best_bearing),
        "delta_az": np.array(best_delta),
    })


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_prediction_pipeline(
    db_connection,
    project_id,
    session_ids,
    outdir,
    indoor_mode="heuristic",
    pixel_size_meters=22.0
):
    os.makedirs(outdir, exist_ok=True)

    # ============================================================
    # 1) VALIDATE SESSIONS WITH GPS
    # ============================================================
    valid_sessions = []
    for sid in session_ids:
        gps_sql = f"""
            SELECT COUNT(*) AS c
            FROM tbl_network_log
            WHERE session_id='{sid}'
              AND lat IS NOT NULL AND lon IS NOT NULL
        """
        cnt = pd.read_sql(gps_sql, db_connection)["c"].iloc[0]
        if cnt > 0:
            valid_sessions.append(str(sid))

    if not valid_sessions:
        raise RuntimeError("No valid DT sessions contain GPS.")

    # ============================================================
    # 2) LOAD & CLEAN SITE DATA (Operator-Wise)
    # ============================================================
    # Added 'network' to the SELECT statement
    site_sql = f"""
        SELECT
            id, project_id, site_key_inferred, network,
            sector_count, lat_pred, lon_pred, azimuth_deg_5
        FROM site_noMl
        WHERE project_id='{project_id}'
    """

    site_df = pd.read_sql(site_sql, db_connection)
    site_df = normcols(site_df)
    site_df = standardize_latlon(site_df)
    site_df = site_df.dropna(subset=["lat", "lon"])

    # --- CLEAN NETWORK COLUMN ---
    # Convert to string, strip whitespace, fill NaNs
    site_df["network"] = site_df.get("network", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    
    # Drop rows where network is blank OR is entirely digits (e.g., '000000', '123')
    site_df = site_df[~site_df["network"].str.match(r'^\d*$')]
    site_df = site_df[site_df["network"] != "unknown"]
    site_df["network"] = site_df["network"].str.upper() # Standardize to uppercase (e.g., JIO, AIRTEL)

    site_df["azimuth"] = pd.to_numeric(site_df.get("azimuth_deg_5", 0), errors="coerce").fillna(0)
    site_df["SiteName"] = site_df["site_key_inferred"].astype(str)
    site_df["sector"] = site_df["sector_count"].fillna(1).astype(int)

    work_site = site_df[["lat", "lon", "azimuth", "SiteName", "sector", "network"]].copy()

    if work_site.empty:
        raise RuntimeError("site_noMl contains no usable sites after cleaning garbage networks.")

    # Get the valid list of operators (e.g., ['JIO', 'AIRTEL'])
    unique_operators = work_site["network"].unique().tolist()
    print(f"--- [Background] Valid Operators Found: {unique_operators} ---")

    # ============================================================
    # 3) LOAD DRIVE TEST
    # ============================================================
    valid_sql = ", ".join([f"'{s}'" for s in valid_sessions])
    dt_sql = f"""
        SELECT *
        FROM tbl_network_log
        WHERE session_id IN ({valid_sql})
          AND (rsrp IS NOT NULL OR rsrq IS NOT NULL OR sinr IS NOT NULL)
    """

    dt_df = pd.read_sql(dt_sql, db_connection)
    dt_df = normcols(dt_df)
    dt_df = standardize_latlon(dt_df)
    dt_core = dt_df.dropna(subset=["lat", "lon"])

    if dt_core.empty:
        raise RuntimeError("DT contains no valid GPS points.")

    # ============================================================
    # 4) MATCH DT TO SITE & INHERIT THE TRUE OPERATOR
    # ============================================================
    print("--- [Background] Performing Global DT-to-Tower Match... ---")
    
    # 1. Match EVERY drive test point to its nearest tower (like your original code)
    dt_match = fast_match(work_site, dt_core[["lat", "lon"]])
    dt_core = pd.concat([dt_core.reset_index(drop=True), dt_match.reset_index(drop=True)], axis=1)

    # 2. Map the "SiteName" to the actual Operator from the work_site dataframe
    site_network_map = work_site.set_index("SiteName")["network"].to_dict()
    
    # 3. Force the DT data to inherit the Operator of the tower it was matched to!
    dt_core["true_operator"] = dt_core["best_site"].map(site_network_map).fillna("UNKNOWN")
    
    # 4. Filter out any points that didn't match to a valid operator
    dt_core = dt_core[dt_core["true_operator"] != "UNKNOWN"]
    
    if dt_core.empty:
        raise RuntimeError("Failed to match DT points to any valid operator towers.")

    # Overwrite the garbage tech types (like GLTEANCHOR) with the actual operator name
    dt_core["network"] = dt_core["true_operator"]

    # --- Proceed with normal calculations ---
    dt_core["log10_dist"] = np.log10(dt_core["dist_m"].clip(lower=1.0))
    dt_core["angle_gain"] = np.cos(np.radians(dt_core["delta_az"])).clip(lower=0)

    def rough_dl_freq(row):
        earf = row.get("earfcn")
        try:
            earf = float(earf)
        except:
            return np.nan
        if earf < 600: return 2110 + earf * 0.1
        elif earf < 1200: return 1930 + (earf - 600) * 0.1
        elif earf < 1950: return 1805 + (earf - 1200) * 0.1
        return np.nan

    dt_core["dl_freq_mhz"] = dt_core.apply(rough_dl_freq, axis=1)

    for c in ["band", "network", "pci", "best_site", "best_sector"]:
        dt_core[c] = dt_core.get(c, "unknown").astype(str)

    dt_core["is_indoor"] = 0
    dt_core["est_indoor_loss_db"] = 0.0

    # ============================================================
    # 5) TRAIN MODELS (Global model handling network as categorical)
    # ============================================================
    TARGETS = [t for t in ["rsrp", "rsrq", "sinr"] if t in dt_core.columns]
    num_features = [
        "dl_freq_mhz", "log10_dist", "dist_m", "angle_gain",
        "delta_az", "bearing_tx_to_ue", "site_lat", "site_lon",
        "lat", "lon", "is_indoor", "est_indoor_loss_db"
    ]
    cat_features = ["band", "network", "pci", "best_site", "best_sector"]

    models = {}
    for tgt in TARGETS:
        y = pd.to_numeric(dt_core[tgt], errors="coerce")
        valid = y.notna()
        X = dt_core.loc[valid, num_features + cat_features]
        y = y.loc[valid]
        preprocess = build_preprocess(num_features, cat_features)
        reg = make_regressor()
        pipe = Pipeline(steps=[("prep", preprocess), ("reg", reg)])
        pipe.fit(X, y)
        models[tgt] = pipe

    # ============================================================
    # 6) LOAD TEST GRID BOUNDARIES
    # ============================================================
    valid_coords = dt_core[(dt_core["lat"] != 0.0) & (dt_core["lon"] != 0.0)]
    median_lat, median_lon = valid_coords["lat"].median(), valid_coords["lon"].median()

    clean_coords = valid_coords[
        (valid_coords["lat"] >= median_lat - 0.5) & (valid_coords["lat"] <= median_lat + 0.5) &
        (valid_coords["lon"] >= median_lon - 0.5) & (valid_coords["lon"] <= median_lon + 0.5)
    ]

    min_lat, max_lat = clean_coords["lat"].min() - 0.0005, clean_coords["lat"].max() + 0.0005
    min_lon, max_lon = clean_coords["lon"].min() - 0.0005, clean_coords["lon"].max() + 0.0005

    step_lat = pixel_size_meters / 111111.0
    avg_lat = np.radians((min_lat + max_lat) / 2)
    step_lon = pixel_size_meters / (111111.0 * np.cos(avg_lat))

    lat_steps, lon_steps = np.arange(min_lat, max_lat, step_lat), np.arange(min_lon, max_lon, step_lon)
    total_points = len(lat_steps) * len(lon_steps)

    if total_points > 20_000_000:  
        raise MemoryError(f"Grid generation aborted! Area is too massive ({total_points} points).")

    gv_lat, gv_lon = np.meshgrid(lat_steps, lon_steps)
    base_test_df = pd.DataFrame({"lat": gv_lat.ravel(), "lon": gv_lon.ravel()})
    
    del gv_lat, gv_lon
    gc.collect()

    # ============================================================
    # 7 & 8) PREDICT & SAVE --- OPERATOR WISE ---
    # ============================================================
    CHUNK_SIZE = 500_000 
    total_written = 0
    total_rows = len(base_test_df)

    # Loop through each valid operator (e.g., Airtel, Jio)
    for operator in unique_operators:
        print(f"--- [Background] Running Predictions for Network: {operator} ---")
        
        # Isolate sites just for this operator
        oper_site = work_site[work_site["network"] == operator].copy()
        if oper_site.empty:
            continue

        for start_idx in range(0, total_rows, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, total_rows)
            
            chunk_df = base_test_df.iloc[start_idx:end_idx].copy()
            
            # Match grid to THIS specific operator's towers
            test_match = fast_match(oper_site, chunk_df[["lat", "lon"]])
            chunk_df = pd.concat([chunk_df.reset_index(drop=True), test_match.reset_index(drop=True)], axis=1)

            chunk_df["log10_dist"] = np.log10(chunk_df["dist_m"].clip(lower=1.0))
            chunk_df["angle_gain"] = np.cos(np.radians(chunk_df["delta_az"])).clip(lower=0)

            # Set the network specifically for this loop pass
            chunk_df["network"] = operator

            # Prepare missing columns
            for col in num_features + cat_features:
                if col not in chunk_df.columns:
                    chunk_df[col] = 0.0 if col in num_features else "unknown"
                if col in cat_features:
                    chunk_df[col] = chunk_df[col].astype(str)

            chunk_df["is_indoor"] = 0
            chunk_df["est_indoor_loss_db"] = 0.0

            # Predict
            for tgt, pipe in models.items():
                preds = pipe.predict(chunk_df[num_features + cat_features])
                if tgt.upper() in KPI_RANGES:
                    preds = clamp_array(preds, tgt.upper())
                chunk_df[f"pred_{tgt}"] = preds

            # Format for Database
            # Format for Database
            final_out = pd.DataFrame({
                "tbl_project_id": int(project_id),
                "lat": chunk_df["lat"],
                "lon": chunk_df["lon"],
                "rsrp": chunk_df.get("pred_rsrp", np.nan),
                "rsrq": chunk_df.get("pred_rsrq", np.nan),
                "sinr": chunk_df.get("pred_sinr", np.nan),
                "serving_cell": chunk_df["best_site"],
                "band": chunk_df["band"],
                "earfcn": chunk_df.get("dl_freq_mhz", np.nan),
                "pci": chunk_df["pci"],
                "network": chunk_df["network"], # <--- The cleaned operator name is saved here!
                "azimuth": np.nan, "tx_power": np.nan, "height": np.nan,
                "reference_signal_power": np.nan, "mtilt": np.nan, "etilt": np.nan,
            })

            # ==========================================
            # PRODUCTION MODE: SAVE CHUNK TO DATABASE
            # ==========================================
            final_out.to_sql(
                "tbl_prediction_data",
                con=db_connection,
                index=False,
                if_exists="append",  # Important: Appends so we don't overwrite previous chunks/operators
                method="multi",
                chunksize=1000       # Adjust this up to 5000 if your database can handle faster inserts
            )

            total_written += len(final_out)
            print(f"--- [Background] Saved {operator} chunk to DB: {start_idx} to {end_idx} ---")

            # CRITICAL: Force clear the RAM before the next loop
            del chunk_df
            del test_match
            del final_out
            gc.collect()

    return outdir, total_written
            
            # ==========================================
            # TESTING MODE: SAVE TO CSV & PRINT PREVIEW
            # ==========================================
            # csv_filename = os.path.join(outdir, f"test_output_{operator}.csv")
            
            # # If the file doesn't exist yet, write the header. Otherwise, append.
            # write_header = not os.path.exists(csv_filename)
            # final_out.to_csv(csv_filename, mode='a', index=False, header=write_header)

            # # Print a quick 5-row preview to the terminal on the first chunk
            # if start_idx == 0:
            #     print(f"\n=== [TEST MODE] PREVIEW FOR {operator} ===")
            #     print(final_out[["lat", "lon", "network", "serving_cell", "rsrp", "rsrq", "sinr"]].head())
            #     print("=========================================\n")
            # ==========================================

    #         total_written += len(final_out)
    #         print(f"--- [Background] Saved {operator} chunk: {start_idx} to {end_idx} ---")

    #         del chunk_df, test_match, final_out
    #         gc.collect()

    # return outdir, total_written
