import uuid
import threading
import pandas as pd
import os
import datetime
import traceback
from sqlalchemy import create_engine

# Import your ML engine functions
from .ml_engine import (
    fetch_baseline,
    fetch_site_data,
    fetch_optimized_sites,
    compute_k1k2,
    run_prediction_only_optimized,
    replace_cells
)

# Global dictionary to track job status
JOBS = {}


def _metric_range(df, col):
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.4f}..{series.max():.4f}"


def _df_summary(stage, df):
    print(f"[LTE_OPT][{stage}] shape={df.shape}")
    print(f"[LTE_OPT][{stage}] columns={list(df.columns)}")
    for col in ["Node_Cell_ID", "cell_id", "node_b_id", "site_id", "Operator"]:
        if col in df.columns:
            print(f"[LTE_OPT][{stage}] distinct_{col}={int(df[col].nunique(dropna=True))}")
    for col in ["pred_rsrp", "pred_rsrq", "pred_sinr"]:
        if col in df.columns:
            print(f"[LTE_OPT][{stage}] range_{col}={_metric_range(df, col)}")

# Database connection
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


class LTEPredictionService_optimised:

    def submit(self, cfg):
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"status": "queued"}

        threading.Thread(
            target=self._run,
            args=(job_id, cfg),
            daemon=True
        ).start()

        return {"job_id": job_id}

    def get(self, job_id):
        return JOBS.get(job_id)

    def _run(self, job_id, cfg):
        try:
            print(
                f"[LTE_OPT][JOB_START] job_id={job_id} project_id={cfg['project_id']} "
                f"region={str(cfg.get('region', 'india')).lower()} operator={cfg.get('operator', 'Airtel')} "
                f"radius={cfg.get('radius', 500)} grid_resolution={cfg.get('grid_resolution', 10)} "
                f"n_workers={cfg.get('n_workers')}"
            )
            self._update(job_id, "running", "Loading baseline")

            project_id = cfg["project_id"]
            
            # ✅ Extract region and operator from the frontend
            region = str(cfg.get("region", "india")).lower()
            operator = cfg.get("operator", "Airtel")

            # ✅ Pass the region into your fetch functions!
            baseline_df = fetch_baseline(project_id, region=region)
            _df_summary("BASELINE_DF", baseline_df)

            self._update(job_id, "running", "Loading site data")
            site_df = fetch_site_data(project_id, region=region)
            _df_summary("SITE_DF", site_df)

            self._update(job_id, "running", "Calculating K1/K2")
            k1k2_map = compute_k1k2(baseline_df, site_df)

            self._update(job_id, "running", f"Loading optimized sites for {operator}")
            opt_sites = fetch_optimized_sites(project_id, operator, region=region)
            _df_summary("OPTIMIZED_SITE_DF", opt_sites)

            params = {
                "radius": cfg.get("radius", 500),
                "grid_resolution": cfg.get("grid_resolution", 10),
                "n_workers": cfg.get("n_workers"),
                "antenna_gain": 18,
                "cable_loss": 2,
                "ue_height": 1.5,
                "frequency_mhz": 1800,
                "bandwidth_mhz": 10
            }

            self._update(job_id, "running", "Running prediction")
            optimized_df = run_prediction_only_optimized(
                opt_sites,
                k1k2_map,
                params
            )
            _df_summary("OPTIMIZED_RF_OUTPUT_DF", optimized_df)

            self._update(job_id, "running", "Saving CSV")

            # Save the CSV
            file_path = self._save_csv(optimized_df, project_id, operator)

            # Format for DB 
            db_df = self._format_for_db(optimized_df, project_id, job_id, operator)
            _df_summary("OPTIMIZED_DB_PAYLOAD", db_df)

            # ✅ Save to DB (Passing the region down!)
            self._save_to_db(db_df, region=region)

            JOBS[job_id]["output"] = file_path
            JOBS[job_id]["rows"] = len(optimized_df)

            self._update(job_id, "done", "Completed")

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            print("❌ ERROR:", traceback.format_exc())

    def _save_csv(self, df, project_id, operator):
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        file_path = os.path.join(
            output_dir,
            f"optimized_{operator}_{project_id}_{timestamp}.csv"
        )

        df.to_csv(file_path, index=False)
        print(f"✅ CSV saved: {file_path}")

        return file_path
    
    def _save_to_db(self, df, region="india"):
        # ✅ FIXED: Removed 'self.' because engine is a global dictionary
        current_engine = engine.get(region.lower(), engine["india"])
        print(
            f"[LTE_OPT][DB_WRITE] table=lte_prediction_optimised_results "
            f"mode=append rows={len(df)} region={region}"
        )
        
        df.to_sql(
            "lte_prediction_optimised_results",
            con=current_engine,
            if_exists="append",
            index=False,
            chunksize=15000,
            method="multi"
        )
        print("✅ Data saved to DB")
    
    def _format_for_db(self, df, project_id, job_id, operator):
        import datetime

        df = df.copy()

        split_cols = df["Node_Cell_ID"].str.split("_", expand=True)

        if split_cols.shape[1] < 2:
            raise ValueError("❌ Invalid Node_Cell_ID format")

        df["node_b_id"] = split_cols[0].astype(str)
        df["cell_id"] = split_cols[1].astype(str)
        df["nodeb_id_cell_id"] = df["node_b_id"] + "_" + df["cell_id"]

        df["project_id"] = project_id
        df["job_id"] = job_id
        
        df["Technology"] = "4G"
        df["Operator"] = operator  
        df["created_at"] = datetime.datetime.now()
        df["site_id"] = df["node_b_id"]

        final_df = df[[
            "project_id",
            "job_id",
            "lat",
            "lon",
            "pred_rsrp",
            "pred_rsrq",
            "pred_sinr",
            "node_b_id",
            "cell_id",
            "Technology",
            "created_at",
            "site_id",
            "nodeb_id_cell_id",
            "Operator" 
        ]]

        return final_df

    def _update(self, job_id, status, msg):
        JOBS[job_id]["status"] = status
        JOBS[job_id]["progress"] = msg
        print(f"[{job_id[:6]}] {msg}")
