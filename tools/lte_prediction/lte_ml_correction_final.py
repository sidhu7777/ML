import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree


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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)

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

    print("✅ ML Correction Done")

    return pred_original