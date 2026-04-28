import uuid
import threading
import pandas as pd
import os
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

# 🔥 Restored your required extension import!
from extensions import db

# We only need the dictionary for the Taiwan connection now
load_dotenv()
engine_dict = {
    "taiwan": create_engine(
        os.getenv("DATABASE_URL_Taiwan"), 
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}

JOBS = {}

class LTEPredictionService:

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
            # Grab the region from your routes (defaults to india)
            region = str(cfg.get("region", "india")).lower()

            self._update(job_id, "running", f"Fetching site data from {region.upper()} database")

            # STEP 1: SITE + OPERATOR
            site_df, operator = fetch_site_data(cfg["project_id"], region=region)

            self._update(job_id, "running", f"Operator: {operator}")

            # STEP 2: DRIVE DATA
            self._update(job_id, "running", "Fetching drive data")
            drive_df = fetch_drive_data(cfg["session_ids"], operator, region=region)

            # STEP 3: BUILDING DATA
            self._update(job_id, "running", "Fetching building data")
            building_df = fetch_building_data(cfg["project_id"], region=region)

            # 🚀 RF PREDICTION
            self._update(job_id, "running", "RF Prediction")
            pred_df = run_rf_prediction_fast(
                site_df,
                drive_df,
                building_df,
                {
                    "radius": cfg["radius_m"],
                    "grid": cfg["grid_resolution"],
                    "workers": cfg["n_workers"]
                }
            )

            # 🧠 ML CORRECTION
            self._update(job_id, "running", "ML Correction")
            final_df = run_ml_fast(pred_df, drive_df)

            # 💾 SAVE OUTPUT TO DB
            self._update(job_id, "running", "Saving results to database")
            
            # Call the save function and pass the region!
            self._save_baseline_results(final_df, cfg["project_id"], job_id, region=region)

            # 💾 SAVE OUTPUT (TEMP CSV)
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

    # Added 'region' parameter
    def _save_baseline_results(self, df, project_id, job_id, region="india"):
        print(f"💾 Saving baseline results to {region.upper()} DB...")

        # ✅ THE MAGIC FIX: Respect the db extension!
        # If region is Taiwan, use the custom engine. Otherwise, use your db.engine.
        if region.lower() == "taiwan" and engine_dict.get("taiwan"):
            save_engine = engine_dict["taiwan"]
        else:
            save_engine = db.engine 

        # COPY DATA
        out = df.copy()

        # REQUIRED COLUMN MAPPING
        out["project_id"] = project_id
        out["job_id"] = job_id
        out["created_at"] = datetime.now()

        # HANDLE MISSING COLUMNS SAFELY
        if "nodeb_id" in out.columns:
            out["node_b_id"] = out["nodeb_id"].astype(str)
        else:
            out["node_b_id"] = None

        if "cell_id" not in out.columns:
            out["cell_id"] = None

        if "operator" not in out.columns:
            out["operator"] = None

        if "site_id" not in out.columns:
            out["site_id"] = None

        # CREATE nodeb_id_cell_id
        out["nodeb_id_cell_id"] = (
            out["node_b_id"].astype(str) + "_" + out["cell_id"].astype(str)
        )

        # FINAL COLUMN ORDER
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

        # 🚀 FAST INSERT USING THE DYNAMIC ENGINE
        out.to_sql(
            "lte_prediction_baseline_results",
            con=save_engine,  # Uses db.engine for India, engine_dict["taiwan"] for Taiwan
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000
        )

        print(f"✅ {len(out)} rows inserted into lte_prediction_baseline_results")