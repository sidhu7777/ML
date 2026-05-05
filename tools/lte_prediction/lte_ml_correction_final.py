import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree


def _metric_bundle(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def run_ml_from_api(pred_df, dt_df):

    print("🧠 ML Correction Started...")

    # ==========================
    # CLEAN
    # ==========================
    def clean(df):
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return df

    pred = clean(pred_df.copy())
    dt   = clean(dt_df.copy())

    pred_original = pred.copy()

    # ==========================
    # RENAME
    # ==========================
    pred.rename(columns={
        'pred_rsrp': 'predicted_rsrp',
        'pred_rsrq': 'predicted_rsrq',
        'pred_sinr': 'predicted_sinr'
    }, inplace=True)

    dt = dt.rename(columns={
        'grid_lat': 'lat',
        'grid_lon': 'lon',
        'long': 'lon',
        'avg_rsrp': 'rsrp'
    })

    pred = pred.dropna(subset=['lat','lon','predicted_rsrp','predicted_rsrq','predicted_sinr'])
    dt   = dt.dropna(subset=['lat','lon','rsrp','rsrq','sinr'])
    print(f"[LTE][ML] pred_rows_after_clean={len(pred)} dt_rows_after_clean={len(dt)}")

    # ==========================
    # 🚀 KD TREE (OPTIMIZED)
    # ==========================
    print("🔗 KDTree mapping...")

    pred_coords = pred[['lat','lon']].values
    dt_coords   = dt[['lat','lon']].values

    tree = KDTree(pred_coords, leaf_size=40)

    _, ind = tree.query(dt_coords, k=1)

    dt['predicted_rsrp'] = pred.iloc[ind.flatten()]['predicted_rsrp'].values
    dt['predicted_rsrq'] = pred.iloc[ind.flatten()]['predicted_rsrq'].values
    dt['predicted_sinr'] = pred.iloc[ind.flatten()]['predicted_sinr'].values
    print(f"[LTE][ML] kd_tree_reference_points={len(pred_coords)} mapped_dt_points={len(dt_coords)}")

    # ==========================
    # FEATURE ENGINEERING
    # ==========================
    center_lat = pred['lat'].mean()
    center_lon = pred['lon'].mean()

    dt['distance'] = np.sqrt((dt['lat'] - center_lat)**2 + (dt['lon'] - center_lon)**2)
    pred['distance'] = np.sqrt((pred['lat'] - center_lat)**2 + (pred['lon'] - center_lon)**2)

    # ==========================
    # TRAIN MODELS
    # ==========================
    kpis = ['rsrp', 'rsrq', 'sinr']

    for kpi in kpis:

        print(f"⚙ Processing {kpi.upper()}")

        pred_col = f'predicted_{kpi}'
        error_col = f'error_{kpi}'

        dt[error_col] = dt[kpi] - dt[pred_col]

        X = dt[['lat','lon',pred_col,'distance']]
        y = dt[error_col]
        print(
            f"[LTE][ML][{kpi.upper()}] train_rows_total={len(X)} "
            f"feature_columns={list(X.columns)}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(
            f"[LTE][ML][{kpi.upper()}] train_rows={len(X_train)} test_rows={len(X_test)} "
            f"error_range={float(y.min()):.4f}..{float(y.max()):.4f}"
        )

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)
        test_error_pred = model.predict(X_test)
        error_metrics = _metric_bundle(y_test, test_error_pred)
        baseline_test = X_test[pred_col].to_numpy()
        actual_test = baseline_test + y_test.to_numpy()
        corrected_test = baseline_test + test_error_pred
        baseline_metrics = _metric_bundle(actual_test, baseline_test)
        corrected_metrics = _metric_bundle(actual_test, corrected_test)
        print(f"[LTE][ML][{kpi.upper()}] error_model_metrics={error_metrics}")
        print(f"[LTE][ML][{kpi.upper()}] holdout_baseline_metrics={baseline_metrics}")
        print(f"[LTE][ML][{kpi.upper()}] holdout_corrected_metrics={corrected_metrics}")

        features = pred[['lat','lon',pred_col,'distance']]
        corrected = pred[pred_col] + model.predict(features)

        # CLIP
        if kpi == 'rsrp':
            corrected = np.clip(corrected, -140, -44)
        elif kpi == 'rsrq':
            corrected = np.clip(corrected, -20, -3)
        elif kpi == 'sinr':
            corrected = np.clip(corrected, -10, 30)

        pred_original[f'ML_Corrected_{kpi.upper()}'] = corrected
        print(
            f"[LTE][ML][{kpi.upper()}] full_prediction_rows={len(features)} "
            f"corrected_range={float(np.min(corrected)):.4f}..{float(np.max(corrected)):.4f}"
        )

    print("✅ ML Correction Done")

    return pred_original
