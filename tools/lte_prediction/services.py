import uuid
import threading
import pandas as pd
import os
import numpy as np
from datetime import datetime

from sqlalchemy import create_engine
from dotenv import load_dotenv

from .ml_engine import (
    run_rf_prediction_fast,
    run_ml_fast,
    fetch_site_data,
    fetch_drive_data,
    fetch_building_data
)

from extensions import db

load_dotenv()
engine_dict = {
    "taiwan": create_engine(
        os.getenv("DATABASE_URL_Taiwan"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}

JOBS = {}


def _metric_range(df, col):
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.4f}..{series.max():.4f}"


def _job_df_summary(stage, df):
    print(f"[LTE][{stage}] shape={df.shape}")
    print(f"[LTE][{stage}] columns={list(df.columns)}")
    for col in ["cell_id", "nodeb_id", "Node_Cell_ID", "node_b_id", "operator", "site_id"]:
        if col in df.columns:
            print(f"[LTE][{stage}] distinct_{col}={int(df[col].nunique(dropna=True))}")
    for col in [
        "pred_rsrp",
        "pred_rsrq",
        "pred_sinr",
        "pred_rsrp_geo",
        "pred_rsrq_geo",
        "pred_sinr_geo",
        "pred_rsrp_demo",
        "pred_rsrq_demo",
        "pred_sinr_demo",
    ]:
        if col in df.columns:
            print(f"[LTE][{stage}] range_{col}={_metric_range(df, col)}")


def _clean_text_series(series):
    cleaned = series.astype(str).str.strip()
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})


def _pick_first_present(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coalesce_columns(df, target, candidates, default=None):
    out = pd.Series([default] * len(df), index=df.index, dtype="object")
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            series = _clean_text_series(series)
        out = out.where(out.notna(), series)
    df[target] = out
    return df


class LTEPredictionService:

    def submit(self, app, cfg):
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"status": "queued"}

        threading.Thread(
            target=self._run_with_app_context,
            args=(app, job_id, cfg),
            daemon=True
        ).start()

        return {"job_id": job_id}

    def get(self, job_id):
        return JOBS.get(job_id)

    def _run_with_app_context(self, app, job_id, cfg):
        with app.app_context():
            self._run(job_id, cfg)

    def _run(self, job_id, cfg):
        try:
            region = str(cfg.get("region", "india")).lower()
            print(
                f"[LTE][JOB_START] job_id={job_id} project_id={cfg['project_id']} "
                f"region={region} session_ids={cfg['session_ids']} radius={cfg['radius_m']} "
                f"grid_resolution={cfg['grid_resolution']} n_workers={cfg['n_workers']} "
                f"max_interference_sites={cfg.get('max_interference_sites', 50)}"
            )

            self._update(job_id, "running", f"Fetching site data from {region.upper()} database")

            site_df, operator = fetch_site_data(cfg["project_id"], region=region)
            _job_df_summary("SITE_DF", site_df)

            self._update(job_id, "running", f"Operator: {operator}")

            self._update(job_id, "running", "Fetching drive data")
            drive_df = fetch_drive_data(
                cfg["session_ids"], operator, cfg["project_id"], region=region
            )
            _job_df_summary("DRIVE_DF", drive_df)

            self._update(job_id, "running", "Fetching building data")
            building_df = fetch_building_data(cfg["project_id"], region=region)
            _job_df_summary("BUILDING_DF", building_df)

            self._update(job_id, "running", "RF Prediction")
            pred_df = run_rf_prediction_fast(
                site_df,
                drive_df,
                building_df,
                {
                    "project_id": cfg["project_id"],
                    "region": region,
                    "radius": cfg["radius_m"],
                    "grid": cfg["grid_resolution"],
                    "workers": cfg["n_workers"],
                    "max_interference_sites": cfg.get("max_interference_sites", 50)
                }
            )

            _job_df_summary("RF_PRED_DF", pred_df)
            self._update(job_id, "running", "Geo correction and smoothing")
            final_df = run_ml_fast(
                pred_df,
                drive_df,
                site_df=site_df,
                building_df=building_df,
                params={
                    "project_id": cfg["project_id"],
                    "region": region,
                    "grid": cfg["grid_resolution"],
                    "tile_size_m": cfg.get("tile_size_m", 100),
                    "cluster_count": cfg.get("cluster_count", 5),
                    "dem_raster_path": cfg.get("dem_raster_path"),
                    "optimizer_weights_path": cfg.get("optimizer_weights_path"),
                    "dt_replace_radius_m": cfg.get("dt_replace_radius_m", 20),
                    "dt_blend_sigma_m": cfg.get("dt_blend_sigma_m", 60),
                    "dt_blend_radius_m": cfg.get("dt_blend_radius_m", 140),
                },
            )
            _job_df_summary("DISPLAY_OUTPUT_DF", final_df)
            production_summary = dict(final_df.attrs.get("production_summary") or {})
            if production_summary:
                geo_metrics = production_summary.get("geo_validation_metrics")
                weights_summary = production_summary.get("weights_summary")
                if weights_summary:
                    print(f"[LTE][GEO_WEIGHTS] {weights_summary}")
                if geo_metrics:
                    print(f"[LTE][GEO_VALIDATION] {geo_metrics}")
                JOBS[job_id]["metrics"] = {
                    "baseline": production_summary.get("baseline_validation_metrics"),
                    "geo": geo_metrics,
                }
                JOBS[job_id]["weights"] = weights_summary

            self._update(job_id, "running", "Saving results to database")
            self._save_baseline_results(
                final_df,
                cfg["project_id"],
                job_id,
                site_df=site_df,
                operator=operator,
                region=region
            )

            output = f"temp/final_{job_id}.csv"
            os.makedirs("temp", exist_ok=True)
            final_df.to_csv(output, index=False)

            JOBS[job_id]["output"] = output
            JOBS[job_id]["rows"] = len(final_df)

            self._update(job_id, "done", "Completed")

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            print(f"Error in Job {job_id}: {str(e)}")

    def _update(self, job_id, status, msg):
        JOBS[job_id]["status"] = status
        JOBS[job_id]["progress"] = msg
        print(f"[{job_id[:6]}] {msg}")

    def _save_baseline_results(self, df, project_id, job_id, site_df=None, operator=None, region="india"):
        print(f"Saving baseline results to {region.upper()} DB...")

        if region.lower() == "taiwan" and engine_dict.get("taiwan"):
            save_engine = engine_dict["taiwan"]
        else:
            save_engine = db.engine

        out = df.copy()

        if "Node_Cell_ID" in out.columns and "node_cell_id" not in out.columns:
            out["node_cell_id"] = out["Node_Cell_ID"]
        if "node_cell_id" in out.columns:
            out["node_cell_id"] = _clean_text_series(out["node_cell_id"])

        # Save final display KPI values into the existing baseline prediction columns.
        out = _coalesce_columns(out, "pred_rsrp", ["pred_rsrp"])
        out = _coalesce_columns(out, "pred_rsrq", ["pred_rsrq"])
        out = _coalesce_columns(out, "pred_sinr", ["pred_sinr"])

        if site_df is not None and not site_df.empty:
            site_meta = site_df.copy()
            if "Node_Cell_ID" in site_meta.columns and "node_cell_id" not in site_meta.columns:
                site_meta["node_cell_id"] = site_meta["Node_Cell_ID"]
            elif "cell_id" in site_meta.columns:
                site_meta["node_cell_id"] = site_meta["cell_id"]

            if "node_cell_id" in site_meta.columns:
                site_meta["node_cell_id"] = _clean_text_series(site_meta["node_cell_id"])
                site_id_col = _pick_first_present(site_meta, ["site_id", "Site ID", "site"])
                operator_col = _pick_first_present(site_meta, ["operator", "network", "cluster", "Technology"])

                rename_map = {}
                if "nodeb_id" in site_meta.columns:
                    rename_map["nodeb_id"] = "site_nodeb_id"
                if site_id_col:
                    rename_map[site_id_col] = "site_site_id"
                if operator_col:
                    rename_map[operator_col] = "site_operator"

                site_meta = site_meta.rename(columns=rename_map)
                keep_cols = ["node_cell_id"] + [
                    col for col in ["site_nodeb_id", "site_site_id", "site_operator"] if col in site_meta.columns
                ]
                site_meta = site_meta[keep_cols].drop_duplicates(subset=["node_cell_id"], keep="first")
                out = out.merge(site_meta, on="node_cell_id", how="left")

        if "node_cell_id" in out.columns:
            split_cols = out["node_cell_id"].astype(str).str.split("_", n=1, expand=True)
            if split_cols.shape[1] >= 2:
                out["derived_nodeb_id"] = _clean_text_series(split_cols[0])
                out["derived_cell_id"] = _clean_text_series(split_cols[1])
            else:
                out["derived_nodeb_id"] = pd.NA
                out["derived_cell_id"] = pd.NA
        else:
            out["derived_nodeb_id"] = pd.NA
            out["derived_cell_id"] = pd.NA

        out["project_id"] = project_id
        out["job_id"] = job_id
        out["created_at"] = datetime.now()

        out = _coalesce_columns(out, "node_b_id", ["node_b_id", "nodeb_id", "site_nodeb_id", "derived_nodeb_id"])
        out = _coalesce_columns(out, "cell_id", ["derived_cell_id", "cell_id"])
        out = _coalesce_columns(out, "operator", ["operator", "site_operator"], default=operator)
        out = _coalesce_columns(out, "site_id", ["site_id", "site_site_id", "node_b_id"])

        for col in ["node_b_id", "cell_id", "operator", "site_id"]:
            out[col] = _clean_text_series(out[col])

        out["nodeb_id_cell_id"] = np.where(
            out["node_b_id"].notna() & out["cell_id"].notna(),
            out["node_b_id"].astype(str) + "_" + out["cell_id"].astype(str),
            out.get("node_cell_id")
        )

        final_cols = [
            "project_id",
            "job_id",
            "lat",
            "lon",
            "pred_rsrp",
            "pred_rsrq",
            "pred_sinr",
            "node_b_id",
            "cell_id",
            "operator",
            "created_at",
            "site_id",
            "nodeb_id_cell_id"
        ]

        out = out[final_cols]
        _job_df_summary("BASELINE_DB_PAYLOAD", out)
        print(
            f"[LTE][BASELINE_DB_WRITE] table=lte_prediction_baseline_results "
            f"mode=append rows={len(out)} project_id={project_id} job_id={job_id}"
        )

        out.to_sql(
            "lte_prediction_baseline_results",
            con=save_engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000
        )

        print(f"{len(out)} rows inserted into lte_prediction_baseline_results")
