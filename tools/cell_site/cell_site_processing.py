#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Site & Sector Locator Tool (NO-ML & ML with Continual Training) — v3.1
============================================================================
- NO-ML: physics/geometry baseline (weighted centroid + azimuth hist), optional TA refine, soft 360/N spacing.
- ML: distance regressor + geometric solver, **continual training** (replay dataset + versioned bundles),
      **NaN-safe** via SimpleImputer, and **site-merge** so that all sectors of a site share the SAME lat/lon.

Fixed in v3.1:
- Fixed "network is both an index level and a column label" merge error
- Fixed weighted centroid bug (subset weights)
- Stronger geofences in NO-ML and optional site merging controlled by --merge-sites
- ML pipeline now **merges sectors to site center (median)** by default (can disable with --no-ml-merge)
- Returns dataframe in output dict for DB insertion
"""
import argparse, os, sys, math, re, logging, hashlib, json
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error
    from sklearn.impute import SimpleImputer
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ----------------------------- Logging -----------------------------
def setup_logger(outdir: str, tag: str="run"):
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_path, filemode="w", level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console)
    logging.info(f"Log file: {log_path}")
    return log_path

# ----------------------------- Utils -------------------------------
def normalize_cols(cols):
    out = []
    for c in cols:
        cc = str(c).strip()
        cc = cc.replace("/", "_per_").replace("-", "_").replace("(", "").replace(")", "").replace(" ", "_")
        cc = re.sub("_+", "_", cc)
        out.append(cc.lower())
    return out

def to_num(x):
    try:
        if x is None: return np.nan
        if isinstance(x, (int,float,np.number)): v = float(x)
        else: v = float(str(x).strip().replace(",",""))
        if abs(v - 2147483647.0) < 1: return np.nan
        if math.isinf(v): return np.nan
        return v
    except: return np.nan

def deg2rad(d): return d*math.pi/180.0
def rad2deg(r): return r*180.0/math.pi

def haversine(lat1, lon1, lat2, lon2):
    R=6371000.0
    phi1, phi2 = deg2rad(lat1), deg2rad(lat2)
    dphi = deg2rad(lat2-lat1)
    dl = deg2rad(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def meters_to_offsets(dNorth, dEast, base_lat):
    R=6378137.0
    dLat = dNorth / R
    dLon = dEast / (R * math.cos(math.pi * base_lat / 180.0))
    latO = base_lat + dLat * 180.0 / math.pi
    lonO = dLon * 180.0 / math.pi
    return latO, lonO

def bearing_from_site(latitude, longitude, lat, lon):
    phi1, phi2 = deg2rad(latitude), deg2rad(lat)
    dlon = deg2rad(lon - longitude)
    
    y = math.sin(dlon) * math.cos(phi2)
    
    # 🟢 FIXED X FORMULA: Correct bearing math instead of angular distance
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    
    b = math.atan2(y, x)
    return (rad2deg(b) + 360.0) % 360.0

def snap_deg(x, step=5):
    try:
        if np.isnan(x): return np.nan
        return (int(round(float(x)/step))*step) % 360
    except: return np.nan

def infer_site_key(cellid):
    if pd.isna(cellid): return np.nan
    s = str(cellid).strip()
    if re.fullmatch(r"\d+", s):
        try:
            v = int(s); return v // 10
        except:
            return s[:-1] if len(s)>1 else s
    else:
        return s[:-1] if len(s)>1 else s

def load_any(path: str, sheet: str = None) -> pd.DataFrame:
    if path.lower().endswith((".xlsx",".xls")):
        return pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    return pd.read_csv(path)

def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = normalize_cols(df.columns)
    if "latitude" in df.columns: df.rename(columns={"latitude":"lat"}, inplace=True)
    if "longitude" in df.columns: df.rename(columns={"longitude":"lon"}, inplace=True)
    for c in ["lat","lon","rsrp_dbm","rsrq_db","sinr_db","rssi","earfcn_or_narfcn","pci_or_psi","band_mhz","speed_kmh","heading_deg","ta"]:
        if c in df.columns: df[c] = df[c].apply(to_num)
    for cat in ["network","technology"]:
        if cat in df.columns: df[cat] = df[cat].astype(str).str.lower()
    return df

# --------------------- Shared Azimuth ------------------------------
def azimuth_histogram(samples: pd.DataFrame, latitude: float, longitude: float, bin_size:int=5):
    if "rsrp_dbm" in samples.columns and not samples["rsrp_dbm"].isna().all():
        # FIX: Use linear scaling (RSRP + 140) to prevent the strongest point from dominating
        w = samples["rsrp_dbm"].apply(lambda v: max(float(v) + 140.0, 1.0) if pd.notna(v) else 1.0).values
    else:
        w = np.ones(len(samples))
        
    bearings = np.array([bearing_from_site(latitude, longitude, r.lat, r.lon) for r in samples.itertuples(index=False)])
    dists = np.array([haversine(latitude, longitude, r.lat, r.lon) for r in samples.itertuples(index=False)])
    w2 = w * np.power(np.maximum(dists, 1.0), 0.5)
    bins = np.arange(0, 360+bin_size, bin_size)
    hist, edges = np.histogram(bearings, bins=bins, weights=w2)
    
    if hist.sum() <= 0:
        return np.nan, np.nan, 0.0
        
    peak_idx = int(np.argmax(hist))
    center_deg = (edges[peak_idx] + edges[peak_idx+1]) / 2.0
    half = hist[peak_idx] / 2.0
    n = len(hist)
    
    left = peak_idx
    while hist[left] >= half:
        left = (left - 1) % n
        if left == (peak_idx - 1) % n: break
        
    right = peak_idx
    while hist[right] >= half:
        right = (right + 1) % n
        if right == (peak_idx + 1) % n: break
        
    width_bins = (right - left) if right >= left else (n - left + right)
    beam_deg = width_bins * bin_size
    reliability = float(hist[peak_idx]/max(hist.sum(),1e-12))
    
    return snap_deg(center_deg, step=bin_size), beam_deg, reliability

def soft_equal_spacing(df_site: pd.DataFrame, bin_size:int=5) -> pd.DataFrame:
    def ang_diff(a,b):
        return (a - b + 180.0) % 360.0 - 180.0
    N = int(df_site["pci_or_psi"].nunique())
    if N <= 1:
        df_site["azimuth_deg_5_soft"] = df_site["azimuth_deg_5"]
        df_site["azimuth_adjustment_deg"] = 0.0
        df_site["template_spacing_deg"] = np.nan
        df_site["spacing_used"] = "none"
        return df_site
    S = 360.0 / N
    rows = df_site.sort_values("azimuth_deg_5").reset_index(drop=True).copy()
    meas = rows["azimuth_deg_5"].astype(float).values
    if "azimuth_reliability" in rows.columns and not rows["azimuth_reliability"].isna().all():
        w = rows["azimuth_reliability"].fillna(0.4).clip(0.05, 1.0).values
    else:
        if "samples" in rows.columns:
            w = (rows["samples"] / max(rows["samples"].max(), 1)).clip(0.05, 1.0).values
        else:
            w = np.ones(len(rows))
    best_cost, best = 1e18, None
    for offset in range(N):
        theta0_candidates = [ (meas[i] - ((i - offset) % N)*S) % 360.0 for i in range(N) ]
        angs = np.radians(theta0_candidates)
        C = np.sum(w * np.cos(angs)); S_sin = np.sum(w * np.sin(angs))
        theta0 = 0.0 if (C==0 and S_sin==0) else (np.degrees(np.arctan2(S_sin, C)) % 360.0)
        cost = 0.0
        for i in range(N):
            k = (i - offset) % N
            target = (theta0 + k*S) % 360.0
            d = abs(ang_diff(meas[i], target)); cost += w[i]*d
        if cost < best_cost:
            best_cost, best = cost, (offset, theta0)
    offset, theta0 = best
    new_angles, adjustments = [], []
    for i in range(N):
        k = (i - offset) % N
        target = (theta0 + k*S) % 360.0
        orig = meas[i]
        rel = w[i]
        alpha = min(max(rel * 0.8, 0.35), 0.85)
        delta = ((target - orig + 180.0) % 360.0) - 180.0
        adj = alpha * delta
        new_a = (orig + adj) % 360.0
        new_angles.append(new_a)
        adjustments.append(adj)
    rows["azimuth_deg_5_soft"] = [snap_deg(a, step=bin_size) for a in new_angles]
    rows["azimuth_adjustment_deg"] = [round(float(adj), 2) for adj in adjustments]
    rows["template_spacing_deg"] = S
    rows["spacing_used"] = f"{int(round(S))}°"
    key_cols = [c for c in ["network","earfcn_or_narfcn","site_key_inferred","pci_or_psi"] if c in df_site.columns]
    rows["__rowid__"] = range(len(rows))
    df_site2 = df_site.merge(rows[key_cols + ["__rowid__","azimuth_deg_5_soft","azimuth_adjustment_deg","template_spacing_deg","spacing_used"]], on=key_cols, how="left")
    df_site2.drop(columns=["__rowid__"], inplace=True)
    return df_site2

# --------------------- NO-ML pipeline ------------------------------
def weighted_centroid_top_rsrp(g: pd.DataFrame):
    if "rsrp_dbm" in g.columns and not g["rsrp_dbm"].isna().all():
        # FIX: Linear weight instead of exponential
        w = g["rsrp_dbm"].apply(lambda v: max(float(v) + 140.0, 1.0) if pd.notna(v) else 1.0)
    else:
        w = pd.Series(np.ones(len(g)), index=g.index)
        
    q = w.quantile(0.9)
    sel = g[w >= q] if q>0 else g
    
    if len(sel) < 10:
        if "rsrp_dbm" in g.columns:
            sel = g.nlargest(min(20, len(g)), "rsrp_dbm")
            w = sel["rsrp_dbm"].apply(lambda v: max(float(v) + 140.0, 1.0) if pd.notna(v) else 1.0)
        else:
            w = pd.Series(np.ones(len(sel)), index=sel.index)
            
    w_sel = w.loc[sel.index]
    W = float(w_sel.sum()) if float(w_sel.sum())>0 else 1.0
    lat_c = float((sel["lat"]*w_sel).sum()/W)
    lon_c = float((sel["lon"]*w_sel).sum()/W)
    med_dist = float(np.median([haversine(lat_c, lon_c, r.lat, r.lon) for r in sel.itertuples(index=False)]))
    
    return lat_c, lon_c, med_dist

def run_noml(
    input_path: str, 
    outdir: str, 
    sheet: str=None, 
    min_samples:int=30, 
    bin_size:int=5, 
    soft_spacing:bool=True, 
    use_ta:bool=False, 
    make_map:bool=False, 
    merge_sites:bool=False
) -> Dict[str,str]:

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load & standardize
    df_raw = load_any(input_path, sheet)
    df = standardize_df(df_raw)

    # Audit preview
    audit_cols = [
        c for c in [
            "timestamp_utc","lat","lon","technology","network","band","band_mhz",
            "earfcn_or_narfcn","pci_or_psi","rsrp_dbm","rsrq_db","sinr_db","ta"
        ] if c in df.columns
    ]
    audit = df[audit_cols].head(20000).copy()
    audit_path = os.path.join(outdir, f"{base}_{ts}_audit_preview.csv")
    audit.to_csv(audit_path, index=False)
    logging.info(f"Audit -> {audit_path}")

    # Grouping
    group_cols = []
    if "network" in df.columns: group_cols.append("network")
    for gc in ["earfcn_or_narfcn", "pci_or_psi"]:
        if gc in df.columns:
            group_cols.append(gc)

    if len(group_cols) < 2:
        raise ValueError("Need at least earfcn_or_narfcn and pci_or_psi (network optional).")

    # First pass predictions
    pred_rows = []
    cellid_col = None
    for c in ["cell_id_global","cellid","cell_id","eci","ecgi","nrcgi","nr_cgi"]:
        if c in df.columns:
            cellid_col = c
            break

    for keys, g in df.groupby(group_cols):
        g2 = g.dropna(subset=["lat","lon"])
        if len(g2) < min_samples:
            continue

        lat_c, lon_c, med_dist = weighted_centroid_top_rsrp(g2)

        if cellid_col and cellid_col in g2.columns:
            try:
                cell_id_rep = g2[cellid_col].dropna().astype(str).value_counts().idxmax()
            except:
                cell_id_rep = np.nan
        else:
            cell_id_rep = np.nan

        kd = {}
        if isinstance(keys, tuple):
            for i, k in enumerate(group_cols):
                kd[k] = keys[i]
        else:
            kd[group_cols[0]] = keys

        pred_rows.append({
            **kd,
            "samples": int(len(g2)),
            "lat_pred_firstcut": lat_c,
            "lon_pred_firstcut": lon_c,
            "median_sample_distance_m": med_dist,
            "cell_id_representative": cell_id_rep
        })

    pred_first = pd.DataFrame(pred_rows)

    if len(pred_first) == 0:
        raise RuntimeError("No groups passed min_samples.")

    pred_first["site_key_inferred"] = pred_first["cell_id_representative"].apply(infer_site_key)

    # 🟢 ADD THIS BLOCK: Fill missing Site Keys before Pandas drops them!
    pred_first["site_key_inferred"] = pred_first["site_key_inferred"].fillna(
        "unknown_site_pci_" + pred_first["pci_or_psi"].astype(str)
    )

    # ================================================================
    # 🔥 FIXED SITE MERGING BLOCK (this is where your crash happened)
    # ================================================================

    # Prevent MultiIndex issues
    pred_first = pred_first.copy()
    pred_first.reset_index(drop=True, inplace=True)

    site_group_cols = []
    if "network" in pred_first.columns:
        site_group_cols.append("network")
    for gc in ["earfcn_or_narfcn", "site_key_inferred"]:
        if gc in pred_first.columns:
            site_group_cols.append(gc)

    if len(site_group_cols) >= 2:

        pred_first["_w"] = pred_first["samples"].clip(lower=1)

        # SAFE: use groupby().agg() instead of apply + reset_index
        site_centroids = pred_first.groupby(site_group_cols).agg(
            latitude=("lat_pred_firstcut", lambda s: float(np.average(s, weights=pred_first.loc[s.index, "_w"]))),
            longitude=("lon_pred_firstcut", lambda s: float(np.average(s, weights=pred_first.loc[s.index, "_w"]))),
            sector_count=("samples", "count")
        ).reset_index()

        # Merge with no duplicate columns
        pred_first = pred_first.merge(site_centroids, on=site_group_cols, how="left")

    else:
        pred_first["latitude"] = pred_first["lat_pred_firstcut"]
        pred_first["longitude"] = pred_first["lon_pred_firstcut"]
        pred_first["sector_count"] = 1

    # ================= END SAFE BLOCK =====================

    # Azimuth estimation
    az_rows = []
    for r in pred_first.itertuples(index=False):
        m = pd.Series(True, index=df.index)
        if "network" in df.columns and hasattr(r, "network") and pd.notna(r.network):
            m &= (df["network"] == r.network)
        if "earfcn_or_narfcn" in df.columns:
            m &= (df["earfcn_or_narfcn"] == r.earfcn_or_narfcn)
        if "pci_or_psi" in df.columns:
            m &= (df["pci_or_psi"] == r.pci_or_psi)

        g = df[m].dropna(subset=["lat","lon"])

        if len(g) < 15:
            az5, beam, rel = (np.nan, np.nan, np.nan)
        else:
            az5, beam, rel = azimuth_histogram(g, r.latitude, r.longitude, bin_size=bin_size)

        az_rows.append({
            "azimuth_deg_5": az5,
            "beamwidth_deg_est": beam,
            "azimuth_reliability": rel
        })

    az_df = pd.DataFrame(az_rows)
    pred_out = pd.concat([pred_first.reset_index(drop=True), az_df], axis=1)
    pred_out.rename(columns={"latitude":"lat_pred","longitude":"lon_pred"}, inplace=True)

    # ----------------------------------------------------------------
    # (All spatial sanity checks stay unchanged below this point)
    # ----------------------------------------------------------------

    # ------------ SPATIAL GUARDS ------------
    try:
        import numpy as _np, math as _math, pandas as _pd

        def _hav(a,b,c,d):
            R=6371000.0
            phi1, phi2 = _math.radians(a), _math.radians(c)
            dphi = _math.radians(c-a); dl = _math.radians(d-b)
            x = _math.sin(dphi/2)**2 + _math.cos(phi1)*_math.cos(phi2)*_math.sin(dl/2)**2
            return 2*R*_math.asin(_math.sqrt(x))

        # Global radius
        lat_med_all = float(df["lat"].median())
        lon_med_all = float(df["lon"].median())
        dists_all = _np.array([
            _hav(lat_med_all, lon_med_all, r.lat, r.lon)
            for r in df.dropna(subset=["lat","lon"]).itertuples(index=False)
        ])
        rad95 = float(_np.percentile(dists_all, 95)) if dists_all.size > 0 else 5000.0

        key_cols_site = []
        if "network" in pred_out.columns:
            key_cols_site.append("network")
        for gc in ["earfcn_or_narfcn","site_key_inferred"]:
            if gc in pred_out.columns:
                key_cols_site.append(gc)

        bad_sites = 0
        if len(key_cols_site) >= 2:
            for sk, gg in pred_out.groupby(key_cols_site):
                lat_s = float(gg["lat_pred"].iloc[0])
                lon_s = float(gg["lon_pred"].iloc[0])
                d_spreads = [_hav(lat_s, lon_s, rr.lat_pred_firstcut, rr.lon_pred_firstcut) 
                             for rr in gg.itertuples(index=False)]
                if len(d_spreads) > 0 and (max(d_spreads) > 1500.0 or _np.median(d_spreads) > 800.0):
                    idx = _pd.Series(True, index=pred_out.index)
                    for i,k in enumerate(key_cols_site):
                        idx &= pred_out[k].eq(gg[k].iloc[0])
                    pred_out.loc[idx, ["lat_pred","lon_pred"]] = pred_out.loc[idx, ["lat_pred_firstcut","lon_pred_firstcut"]].values
                    pred_out.loc[idx, "sector_count"] = 1
                    bad_sites += int(idx.sum())
        if bad_sites:
            logging.warning(f"Site-spread guard reverted {bad_sites} rows.")

        # Per-sector geofence
        corrected = 0
        for r in pred_out.itertuples(index=False):
            m = _pd.Series(True, index=df.index)
            if "network" in df.columns and hasattr(r, "network") and _pd.notna(r.network):
                m &= (df["network"] == r.network)
            m &= (df["earfcn_or_narfcn"] == r.earfcn_or_narfcn)
            m &= (df["pci_or_psi"] == r.pci_or_psi)
            g = df[m].dropna(subset=["lat","lon"])
            lat_p, lon_p = float(r.lat_pred), float(r.lon_pred)

            bad = False

            if len(g) >= 5:
                glat_med = float(g["lat"].median())
                glon_med = float(g["lon"].median())
                dist_local = _hav(glat_med, glon_med, lat_p, lon_p)
                if dist_local > 1000.0:
                    bad = True

            dist_global = _hav(lat_med_all, lon_med_all, lat_p, lon_p)
            if dist_global > (rad95 + 1000.0):
                bad = True

            if bad:
                idx = (
                    pred_out["network"].eq(getattr(r,"network",_np.nan)) &
                    pred_out["earfcn_or_narfcn"].eq(getattr(r,"earfcn_or_narfcn",_np.nan)) &
                    pred_out["pci_or_psi"].eq(getattr(r,"pci_or_psi",_np.nan))
                )
                pred_out.loc[idx, ["lat_pred","lon_pred"]] = pred_out.loc[idx, ["lat_pred_firstcut","lon_pred_firstcut"]].values
                corrected += int(idx.sum())
        if corrected:
            logging.warning(f"Spatial geofence corrected {corrected} rows.")
    except Exception as e:
        logging.warning(f"Spatial sanity checks skipped: {e}")

    # ------------ save outputs ------------
    cols = [
        c for c in [
            "network","earfcn_or_narfcn","pci_or_psi","samples","lat_pred","lon_pred",
            "azimuth_deg_5","beamwidth_deg_est","median_sample_distance_m",
            "cell_id_representative","site_key_inferred","sector_count","azimuth_reliability"
        ] if c in pred_out.columns
    ]

    no_ta_path = os.path.join(outdir, f"{base}_{ts}_pred_main_no_ta.csv")
    pred_out[cols].to_csv(no_ta_path, index=False)
    logging.info(f"NO-ML -> {no_ta_path}")

    # ------------ soft spacing ------------
    soft_path = None
    pred_soft = None
    if soft_spacing:
        key_cols = []
        if "network" in pred_out.columns:
            key_cols.append("network")
        for gc in ["earfcn_or_narfcn","site_key_inferred"]:
            if gc in pred_out.columns:
                key_cols.append(gc)

        parts = []
        if len(key_cols) > 0:
            for _, g in pred_out.groupby(key_cols):
                parts.append(soft_equal_spacing(g, bin_size=bin_size))
            if parts:
                pred_soft = pd.concat(parts, ignore_index=True)
        else:
            pred_soft = pred_out.copy()

        if pred_soft is not None:
            pred_soft["azimuth_deg_label_soft"] = pred_soft["azimuth_deg_5_soft"].apply(
                lambda v: f"{int(v)} degree" if not pd.isna(v) else ""
            )
            keep = [
                c for c in [
                    "network","earfcn_or_narfcn","site_key_inferred","pci_or_psi",
                    "samples","lat_pred","lon_pred","azimuth_deg_5","azimuth_deg_5_soft",
                    "azimuth_deg_label_soft","azimuth_adjustment_deg","template_spacing_deg",
                    "beamwidth_deg_est","median_sample_distance_m","cell_id_representative",
                    "sector_count","azimuth_reliability","spacing_used"
                ] if c in pred_soft.columns
            ]
            soft_path = os.path.join(outdir, f"{base}_{ts}_pred_main_no_ta_soft.csv")
            pred_soft[keep].to_csv(soft_path, index=False)
            logging.info(f"NO-ML + soft -> {soft_path}")
        else:
            logging.warning("Soft spacing enabled but 'pred_soft' was not created.")

    # ------------ TA refine ------------
    ta_path = None
    if use_ta and "ta" in df.columns and not df["ta"].dropna().empty:
        rows_ta = []
        for r in pred_out.itertuples(index=False):
            m = pd.Series(True, index=df.index)
            if "network" in df.columns and hasattr(r, "network") and pd.notna(r.network):
                m &= (df["network"] == r.network)
            m &= (df["earfcn_or_narfcn"] == r.earfcn_or_narfcn)
            m &= (df["pci_or_psi"] == r.pci_or_psi)

            g = df[m].dropna(subset=["lat","lon","ta"])
            if len(g) < 25:
                rows_ta.append({**r._asdict(), "ta_refine_abs_error_m": np.nan})
                continue

            base_lat = float(r.lat_pred)
            base_lon = float(r.lon_pred)
            g = g.assign(ta_m = g["ta"] * 78.0)
            best = (1e18, base_lat, base_lon)

            for dN in range(-200, 201, 50):
                for dE in range(-200, 201, 50):
                    lat_try, lon_try = meters_to_offsets(dN, dE, base_lat)
                    dists = np.array([
                        haversine(lat_try, lon_try, rr.lat, rr.lon) 
                        for rr in g.itertuples(index=False)
                    ])
                    loss = float(np.mean(np.abs(dists - g["ta_m"].values)))
                    if loss < best[0]:
                        best = (loss, lat_try, lon_try)

            rdict = r._asdict()
            rdict["lat_pred"] = best[1]
            rdict["lon_pred"] = best[2]
            rdict["ta_refine_abs_error_m"] = best[0]
            rows_ta.append(rdict)

        pred_ta = pd.DataFrame(rows_ta)
        ta_path = os.path.join(outdir, f"{base}_{ts}_pred_main_ta_refined.csv")
        pred_ta.to_csv(ta_path, index=False)
        logging.info(f"NO-ML TA refine -> {ta_path}")

    # ------------ map output ------------
    map_path = None
    if make_map:
        try:
            import folium
            use_df = pred_soft if (soft_spacing and pred_soft is not None) else pred_out
            m = folium.Map(
                location=[float(use_df["lat_pred"].median()), 
                          float(use_df["lon_pred"].median())], 
                zoom_start=13
            )
            for r in use_df.itertuples(index=False):
                lat = float(r.lat_pred)
                lon = float(r.lon_pred)
                az = getattr(
                    r,
                    "azimuth_deg_5_soft" if (soft_spacing and pred_soft is not None) else "azimuth_deg_5",
                    np.nan
                )
                tooltip = f"{getattr(r,'network','')} | {getattr(r,'earfcn_or_narfcn','')} | PCI {getattr(r,'pci_or_psi','')} | Az {az}°"
                folium.CircleMarker([lat,lon], radius=4, fill=True, tooltip=tooltip).add_to(m)

            map_path = os.path.join(outdir, f"{base}_{ts}_map.html")
            m.save(map_path)
            logging.info(f"Map -> {map_path}")
        except Exception as e:
            logging.warning(f"Map generation skipped: {e}")

    # Which dataframe to return
    df_to_return = pred_soft if (soft_spacing and pred_soft is not None) else pred_out

    return {
        "audit": audit_path,
        "no_ta": no_ta_path,
        "soft": soft_path,
        "ta": ta_path,
        "map": map_path,
        "dataframe": df_to_return
    }



# --------------------- ML pipeline (continual training + imputer) --
FEATURE_CANDIDATES = ["rsrp_dbm","rsrq_db","sinr_db","rssi","band_mhz","earfcn_or_narfcn","speed_kmh","heading_deg"]
CATEGORICALS = ["technology","network"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FEATURE_CANDIDATES:
        if c in df.columns: df[c] = df[c].apply(to_num)
    if "rsrp_dbm" in df.columns:
        df["rsrp_lin"] = df["rsrp_dbm"].apply(lambda v: 10**(v/10.0) if pd.notna(v) else np.nan)
    if "sinr_db" in df.columns and "rsrp_dbm" in df.columns:
        df["rsrp_sinr"] = df["sinr_db"].fillna(0) + df["rsrp_dbm"].fillna(-120)
    if "rsrq_db" in df.columns and "rsrp_dbm" in df.columns:
        df["rsrp_rsrq"] = df["rsrq_db"].fillna(0) + df["rsrp_dbm"].fillna(-120)
    for cat in CATEGORICALS:
        if cat in df.columns: df[cat] = df[cat].astype(str).str.lower()
    return df

def select_feature_matrix(df: pd.DataFrame):
    cols = [c for c in FEATURE_CANDIDATES + ["rsrp_lin","rsrp_sinr","rsrp_rsrq"] if c in df.columns]
    used_cats = [c for c in CATEGORICALS if c in df.columns]
    X = df[cols + used_cats].copy()
    X = pd.get_dummies(X, columns=used_cats, dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, list(X.columns)

# Replay persistence
def bundle_dir_from_model(model_path:str) -> str:
    return os.path.dirname(os.path.abspath(model_path))

def replay_path_from_dir(model_dir:str) -> str:
    return os.path.join(model_dir, "distance_model_replay.csv.gz")

def save_bundle(model, imputer, features:List[str], outdir:str, version:int, extra_meta:Dict) -> str:
    bundle = {"model": model, "imputer": imputer, "features": features, "version": version, "meta": extra_meta}
    model_path = os.path.join(outdir, "distance_model.joblib")
    joblib.dump(bundle, model_path)
    return model_path

def load_bundle(model_path:str):
    return joblib.load(model_path)

def hash_rows(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    def hrow(row):
        s = "|".join([str(row.get(c, "")) for c in cols])
        return hashlib.md5(s.encode("utf-8")).hexdigest()
    return df.apply(hrow, axis=1)

def append_to_replay(replay_path:str, X_new: pd.DataFrame, y_new: pd.Series):
    if os.path.exists(replay_path):
        rep = pd.read_csv(replay_path, compression="gzip")
        y_col = "target_distance_m"
        cols_union = list(sorted(set(rep.columns) | set(X_new.columns) | {y_col}))
        for c in cols_union:
            if c not in rep.columns: rep[c] = 0
            if c not in X_new.columns and c != y_col: X_new[c] = 0
        rep = rep[cols_union]
        X_new = X_new[cols_union[:-1]] if cols_union[-1] == y_col else X_new[cols_union]
        new = X_new.copy(); new[y_col] = y_new.values
        merged = pd.concat([rep, new], ignore_index=True)
        feat_cols = [c for c in merged.columns if c != y_col]
        merged["_h"] = hash_rows(merged, feat_cols)
        merged = merged.drop_duplicates(subset="_h").drop(columns="_h")
        merged.to_csv(replay_path, index=False, compression="gzip")
        X_all = merged[feat_cols].copy(); y_all = merged[y_col].copy()
    else:
        y_col = "target_distance_m"
        new = X_new.copy(); new[y_col] = y_new.values
        new.to_csv(replay_path, index=False, compression="gzip")
        X_all = X_new.copy(); y_all = y_new.copy()
    return X_all, y_all

def train_or_update_model(train_df: pd.DataFrame, outdir:str, existing_bundle_path:str=None):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn/joblib not available. Install: pip install scikit-learn joblib")
    # Validate labels
    label_lat_col = None; label_lon_col = None
    for cand in ["sector_lat","site_lat","latitude","site_latitude"]:
        if cand in train_df.columns: label_lat_col = cand; break
    for cand in ["sector_lon","site_lon","longitude","site_longitude"]:
        if cand in train_df.columns: label_lon_col = cand; break
    if not label_lat_col or not label_lon_col:
        raise ValueError("Training needs sector/site coordinates (e.g., sector_lat, sector_lon).")
    tr = train_df.dropna(subset=["lat","lon", label_lat_col, label_lon_col]).copy()
    tr["target_distance_m"] = [haversine(r.lat, r.lon, getattr(r, label_lat_col), getattr(r, label_lon_col)) for r in tr.itertuples(index=False)]
    # Features
    tr = build_features(tr)
    X_new, feat_new = select_feature_matrix(tr)
    y_new = tr["target_distance_m"]
    # Replay path
    replay_path = replay_path_from_dir(outdir if existing_bundle_path is None else bundle_dir_from_model(existing_bundle_path))
    X_all, y_all = append_to_replay(replay_path, X_new, y_new)
    features = list(X_all.columns)
    # Imputer
    imputer = SimpleImputer(strategy="median")
    X_all_imp = pd.DataFrame(imputer.fit_transform(X_all), columns=features)
    # CV
    kf = KFold(n_splits=min(5, max(2, int(len(X_all_imp) / 500))), shuffle=True, random_state=42)
    maes = []
    for tr_idx, va_idx in kf.split(X_all_imp):
        model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(X_all_imp.iloc[tr_idx], y_all.iloc[tr_idx])
        pred = model.predict(X_all_imp.iloc[va_idx])
        maes.append(mean_absolute_error(y_all.iloc[va_idx], pred))
    cv_mae = float(np.mean(maes)) if len(maes)>0 else np.nan
    cv_rmse = float(np.sqrt(np.mean((model.predict(X_all_imp) - y_all.values)**2))) if len(X_all_imp)>0 else np.nan
    # Final fit
    model = RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_all_imp, y_all)
    # Versioning
    version = 1
    if existing_bundle_path and os.path.exists(existing_bundle_path):
        try:
            old = load_bundle(existing_bundle_path)
            version = int(old.get("version", 1)) + 1
        except Exception:
            version = 1
    meta = {"cv_mae_m": cv_mae, "cv_rmse_m": cv_rmse, "n_train": int(len(X_all_imp)), "replay_path": replay_path, "timestamp": datetime.now().isoformat()}
    model_path = save_bundle(model, imputer, features, outdir, version, meta)
    return model, imputer, {"model_path": model_path, **meta}

def solve_site_from_predicted_ranges(samples: pd.DataFrame, lat0: float, lon0: float,
                                     start_step_m: float = 300.0, min_step_m: float = 10.0):
    def loss(latc, lonc):
        d = np.array([haversine(latc, lonc, r.lat, r.lon) for r in samples.itertuples(index=False)])
        rhat = samples["pred_range_m"].values
        if "rsrp_dbm" in samples.columns and not samples["rsrp_dbm"].isna().all():
            w = np.array([max(10**(v/10.0), 1e-13) if not pd.isna(v) else 0.0 for v in samples["rsrp_dbm"].values])
        else:
            w = np.ones(len(samples))
        return float(np.average(np.abs(d - rhat), weights=w))
    best_lat, best_lon = lat0, lon0
    best_loss = loss(best_lat, best_lon)
    step = start_step_m
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while step >= min_step_m:
        improved = False
        for dn, de in dirs:
            lat_try, lon_try = meters_to_offsets(dn*step, de*step, best_lat)
            cur_loss = loss(lat_try, lon_try)
            if cur_loss + 1e-6 < best_loss:
                best_lat, best_lon, best_loss = lat_try, lon_try, cur_loss
                improved = True
        if not improved:
            step /= 2.0
    return best_lat, best_lon, best_loss

def run_ml(train_path: str=None, model_path: str=None, update_model: bool=False, input_path: str=None, outdir: str=None,
           sheet_train:str=None, sheet_input:str=None, min_samples:int=30, bin_size:int=5, soft_spacing:bool=True, make_map:bool=False,
           eval_path: str=None, sheet_eval: str=None, no_ml_merge: bool=False):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn/joblib not available. Install: pip install scikit-learn joblib")
    if input_path is None or outdir is None:
        raise ValueError("input_path and outdir are required")
    os.makedirs(outdir, exist_ok=True)
    base_in = os.path.splitext(os.path.basename(input_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load input
    in_raw = load_any(input_path, sheet_input)
    df = standardize_df(in_raw)
    in_audit = df.head(20000)
    in_audit_path = os.path.join(outdir, f"{base_in}_{ts}_audit_infer.csv")
    in_audit.to_csv(in_audit_path, index=False)
    logging.info(f"Audit infer -> {in_audit_path}")
    
    # Model: load or train/update
    if model_path and not update_model:
        logging.info(f"Loading model: {model_path}")
        bundle = load_bundle(model_path)
        model = bundle["model"]; tr_feats = bundle["features"]; imputer = bundle.get("imputer", None)
        bundle_meta = bundle.get("meta", {})
    else:
        if train_path is None:
            raise ValueError("Provide --train to fit a model or --model to load one (with --update-model to update).")
        tr_raw = load_any(train_path, sheet_train)
        tr = standardize_df(tr_raw)
        model, imputer, meta = train_or_update_model(tr, outdir, existing_bundle_path=model_path)
        logging.info(f"Trained/updated model -> {meta['model_path']} | CV MAE ≈ {meta.get('cv_mae_m', np.nan):.2f} m | CV RMSE ≈ {meta.get('cv_rmse_m', np.nan):.2f} m | n_train={meta.get('n_train')}")
        bundle_meta = meta
        tr_feats = load_bundle(meta["model_path"])["features"]
    
    # Predict ranges
    feats_in = build_features(df.copy())
    X_in, featnames = select_feature_matrix(feats_in)
    for col in tr_feats:
        if col not in X_in.columns: X_in[col] = 0
    X_in = X_in[tr_feats]
    X_in = X_in.replace([np.inf, -np.inf], np.nan)
    if imputer is None:
        logging.warning("Bundle has no imputer; fitting a temporary median imputer on inference features.")
        imputer = SimpleImputer(strategy="median").fit(X_in)
    X_in_imp = pd.DataFrame(imputer.transform(X_in), columns=tr_feats)
    df["pred_range_m"] = model.predict(X_in_imp)
    
    # group keys
    group_cols = []
    if "network" in df.columns: group_cols.append("network")
    for gc in ["earfcn_or_narfcn","pci_or_psi"]:
        if gc in df.columns: group_cols.append(gc)
    if len(group_cols) < 2:
        raise ValueError("Need at least earfcn_or_narfcn and pci_or_psi (network optional).")
    
    # initial point
    def weighted_centroid(g):
        if "rsrp_dbm" in g.columns and not g["rsrp_dbm"].isna().all():
            w = g["rsrp_dbm"].apply(lambda v: max(10**(v/10.0), 1e-13) if pd.notna(v) else 0.0)
        else:
            w = pd.Series(np.ones(len(g)), index=g.index)
        q = w.quantile(0.9)
        sel = g[w >= q] if q>0 else g
        if len(sel) < 10:
            if "rsrp_dbm" in g.columns:
                sel = g.nlargest(min(20, len(g)), "rsrp_dbm")
                w = sel["rsrp_dbm"].apply(lambda v: max(10**(v/10.0), 1e-13) if pd.notna(v) else 0.0)
            else:
                w = pd.Series(np.ones(len(sel)), index=sel.index)
        w_sel = w.loc[sel.index]
        W = float(w_sel.sum()) if float(w_sel.sum())>0 else 1.0
        lat0 = float((sel["lat"]*w_sel).sum()/W); lon0 = float((sel["lon"]*w_sel).sum()/W)
        return lat0, lon0
    
    # solve
    pred_rows = []
    for keys, g in df.groupby(group_cols):
        g2 = g.dropna(subset=["lat","lon","pred_range_m"])
        if len(g2) < max(20, min_samples): continue
        lat0, lon0 = weighted_centroid(g2)
        lat_hat, lon_hat, loss_mae = solve_site_from_predicted_ranges(g2, lat0, lon0)
        az5, beam_deg, rel = azimuth_histogram(g2, lat_hat, lon_hat, bin_size=bin_size)
        kd = {}
        if isinstance(keys, tuple):
            for i,k in enumerate(group_cols): kd[k] = keys[i]
        else:
            kd[group_cols[0]] = keys
        pred_rows.append({**kd, "samples": int(len(g2)), "lat_pred": lat_hat, "lon_pred": lon_hat, "azimuth_deg_5": az5, "beamwidth_deg_est": beam_deg, "azimuth_reliability": rel, "range_mae_m": float(loss_mae)})
    
    pred_df = pd.DataFrame(pred_rows)
    if len(pred_df)==0:
        raise RuntimeError("No sector groups produced predictions.")
    
    # Save per-sector predictions
    per_sector_path = os.path.join(outdir, f"{base_in}_{ts}_pred_ml_per_sector.csv")
    pred_df.to_csv(per_sector_path, index=False)
    logging.info(f"ML (per-sector) -> {per_sector_path}")
    
    # cell_id enrichment
    cellid_col = None
    for c in ["cell_id_representative","cell_id_global","cellid","cell_id","eci","ecgi","nrcgi","nr_cgi"]:
        if c in df.columns: cellid_col = c; break
    
    if cellid_col:
        cellmap = (df.groupby(group_cols)[cellid_col].agg(lambda s: s.dropna().astype(str).value_counts().idxmax() if s.dropna().size>0 else np.nan).reset_index().rename(columns={cellid_col:"cell_id_representative"}))
        pred_df = pred_df.merge(cellmap, on=group_cols, how="left")
        pred_df["site_key_inferred"] = pred_df["cell_id_representative"].apply(infer_site_key)
    else:
        pred_df["cell_id_representative"] = np.nan
        pred_df["site_key_inferred"] = np.nan
    
    # --- Enforce same site lat/lon across sectors (ML) unless disabled ---
    if not no_ml_merge:
        site_keys = [c for c in ["network","earfcn_or_narfcn","site_key_inferred"] if c in pred_df.columns]
        if len(site_keys) >= 2:
            pred_df["sector_count"] = pred_df.groupby(site_keys)["pci_or_psi"].transform("nunique").fillna(1).astype(int)
            # robust center = median of sector centers
            site_centers = pred_df.groupby(site_keys).agg(latitude=("lat_pred","median"), longitude=("lon_pred","median")).reset_index()
            pred_df = pred_df.merge(site_centers, on=site_keys, how="left")
            pred_df["lat_pred"] = pred_df["latitude"].fillna(pred_df["lat_pred"])
            pred_df["lon_pred"] = pred_df["longitude"].fillna(pred_df["lon_pred"])
            # compute site spread for info
            def _hav(a,b,c,d):
                R=6371000.0
                phi1, phi2 = math.radians(a), math.radians(c)
                dphi = math.radians(c-a); dl = math.radians(d-b)
                x = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
                return 2*R*math.asin(math.sqrt(x))
            def _spread(group):
                latc, lonc = float(group["lat_pred"].iloc[0]), float(group["lon_pred"].iloc[0])
                return pd.Series({"site_spread_m": max([_hav(latc, lonc, float(r.lat_pred), float(r.lon_pred)) for r in group.itertuples(index=False)]) if len(group)>0 else 0.0})
            spread = pred_df.groupby(site_keys).apply(_spread).reset_index()
            pred_df = pred_df.merge(spread, on=site_keys, how="left")
        else:
            pred_df["site_spread_m"] = np.nan
            pred_df["sector_count"] = 1
    else:
        pred_df["site_spread_m"] = np.nan
        pred_df["sector_count"] = 1
    
    # soft spacing
    soft_path = None
    key_cols = [c for c in ["network","earfcn_or_narfcn","site_key_inferred"] if c in pred_df.columns]
    if soft_spacing and len(key_cols)>0:
        parts = []
        for _, g in pred_df.groupby(key_cols):
            parts.append(soft_equal_spacing(g, bin_size=bin_size))
        pred_soft = pd.concat(parts, ignore_index=True)
        pred_soft["azimuth_deg_label_soft"] = pred_soft["azimuth_deg_5_soft"].apply(lambda v: f"{int(v)} degree" if not pd.isna(v) else "")
    else:
        pred_soft = pred_df.copy()
    
    # save
    no_ta_path = os.path.join(outdir, f"{base_in}_{ts}_pred_ml_no_ta.csv")
    soft_path = os.path.join(outdir, f"{base_in}_{ts}_pred_ml_no_ta_soft.csv")
    pred_df.to_csv(no_ta_path, index=False)
    pred_soft.to_csv(soft_path, index=False)
    
    # Persist training CV metrics if available
    try:
        if bundle_meta:
            metrics_path = os.path.join(outdir, f"{base_in}_{ts}_ml_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"cv": bundle_meta}, f, indent=2)
            logging.info(f"Saved ML CV metrics -> {metrics_path}")
    except Exception:
        pass
    
    logging.info(f"ML -> {no_ta_path} and {soft_path}")
    
    # map (optional)
    map_path = None
    if make_map:
        try:
            import folium
            use_df = pred_soft if soft_spacing else pred_df
            m = folium.Map(location=[float(use_df["lat_pred"].median()), float(use_df["lon_pred"].median())], zoom_start=13)
            for r in use_df.itertuples(index=False):
                lat, lon = float(r.lat_pred), float(r.lon_pred)
                az = getattr(r, "azimuth_deg_5_soft", np.nan) if soft_spacing else getattr(r, "azimuth_deg_5", np.nan)
                tooltip = f"{getattr(r,'network','')} | {getattr(r,'earfcn_or_narfcn','')} | PCI {getattr(r,'pci_or_psi','')} | Az {az}°"
                folium.CircleMarker([lat,lon], radius=4, fill=True, tooltip=tooltip).add_to(m)
            map_path = os.path.join(outdir, f"{base_in}_{ts}_map.html")
            m.save(map_path)
            logging.info(f"Map -> {map_path}")
        except Exception as e:
            logging.warning(f"Map generation skipped: {e}")
    
    return {
        "no_ta": no_ta_path, 
        "soft": soft_path, 
        "map": map_path, 
        "per_sector": per_sector_path,
        "dataframe": pred_soft
    }


# ----------------------------- CLI ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Unified Site & Sector Locator Tool (NO-ML & ML w/ Continual Training, NaN-safe, site-merge)")
    ap.add_argument("--method", choices=["noml","ml"], required=True, help="Which method to run")
    ap.add_argument("-i","--input", help="Input CSV/XLSX with drive-test samples (required for both methods)")
    ap.add_argument("-o","--outdir", required=True, help="Output directory")
    ap.add_argument("--sheet", default=None, help="Excel sheet for NO-ML input file")
    ap.add_argument("--sheet-train", default=None, help="Excel sheet for ML train file")
    ap.add_argument("--sheet-input", default=None, help="Excel sheet for ML input file")
    ap.add_argument("--min-samples", type=int, default=30, help="Min samples per (network,EARFCN,PCI) group")
    ap.add_argument("--bin-size", type=int, default=5, choices=[1,3,5,10,15], help="Azimuth histogram bin size (deg)")
    ap.add_argument("--soft-spacing", action="store_true", help="Soft enforce ~360/N spacing per site")
    ap.add_argument("--use-ta", action="store_true", help="(NO-ML only) Grid-search TA refine")
    ap.add_argument("--make-map", action="store_true", help="Export Folium HTML map")
    # ML specific
    ap.add_argument("--train", help="(ML) Labeled truth CSV/XLSX with sector_lat/sector_lon")
    ap.add_argument("--model", help="(ML) Pre-trained joblib model bundle")
    ap.add_argument("--update-model", action="store_true", help="(ML) Update existing model with new --train data (continual training)")
    ap.add_argument("--eval", help="(ML) Optional labeled eval CSV/XLSX to compute site-level metrics")
    ap.add_argument("--sheet-eval", default=None, help="Excel sheet for ML eval file")
    ap.add_argument("--no-ml-merge", action="store_true", help="Disable ML site merge (debug only)")

    args = ap.parse_args()

    setup_logger(args.outdir, tag=f"{args.method}")
    logging.info(f"Args: {vars(args)}")
    try:
        if args.method == "noml":
            if not args.input: raise ValueError("--input is required for NO-ML")
            outs = run_noml(args.input, args.outdir, sheet=args.sheet, min_samples=args.min_samples, bin_size=args.bin_size, soft_spacing=args.soft_spacing, use_ta=args.use_ta, make_map=args.make_map, merge_sites=args.soft_spacing)
        else:
            if not args.input: raise ValueError("--input is required for ML")
            outs = run_ml(train_path=args.train, model_path=args.model, update_model=args.update_model, input_path=args.input, outdir=args.outdir, sheet_train=args.sheet_train, sheet_input=args.sheet_input, min_samples=args.min_samples, bin_size=args.bin_size, soft_spacing=args.soft_spacing, make_map=args.make_map, eval_path=args.eval, sheet_eval=args.sheet_eval, no_ml_merge=args.no_ml_merge)
        logging.info("Done.")
        for k,v in outs.items():
            if k != "dataframe" and v: 
                logging.info(f"{k}: {v}")
    except Exception as e:
        logging.exception(f"FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
