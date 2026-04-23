import uuid
import threading
import pandas as pd
import os

from .ml_engine import (
    run_rf_prediction_fast,
    run_ml_fast,
    fetch_site_data,
    fetch_drive_data,
    fetch_building_data
)
from datetime import datetime
from extensions import db

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
            self._update(job_id, "running", "Fetching site data")

            # ✅ STEP 1: SITE + OPERATOR
            site_df, operator = fetch_site_data(cfg["project_id"])

            self._update(job_id, "running", f"Operator: {operator}")

            # ✅ STEP 2: DRIVE DATA (FILTERED + CACHE)
            self._update(job_id, "running", "Fetching drive data")

            drive_df = fetch_drive_data(cfg["session_ids"], operator)

            # ✅ STEP 3: BUILDING DATA
            self._update(job_id, "running", "Fetching building data")

            building_df = fetch_building_data(cfg["project_id"])

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

            # 💾 SAVE OUTPUT (TEMP)
            output = f"temp/final_{job_id}.csv"
            final_df.to_csv(output, index=False)

            JOBS[job_id]["output"] = output
            JOBS[job_id]["rows"] = len(final_df)

            self._update(job_id, "done", "Completed")

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)

    def _update(self, job_id, status, msg):
        JOBS[job_id]["status"] = status
        JOBS[job_id]["progress"] = msg
        print(f"[{job_id[:6]}] {msg}")
    def _save_baseline_results(self, df, project_id, job_id):

        print("💾 Saving baseline results to DB...")

        # ✅ COPY DATA
        out = df.copy()

        # ✅ REQUIRED COLUMN MAPPING
        out["project_id"] = project_id
        out["job_id"] = job_id
        out["created_at"] = datetime.now()

        # ⚠️ HANDLE MISSING COLUMNS SAFELY
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

        # 🔥 CREATE nodeb_id_cell_id
        out["nodeb_id_cell_id"] = (
            out["node_b_id"].astype(str) + "_" + out["cell_id"].astype(str)
        )

        # ✅ FINAL COLUMN ORDER
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

        # 🚀 FAST INSERT
        out.to_sql(
            "lte_prediction_baseline_results",
            db.engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000
        )

        print(f"✅ {len(out)} rows inserted into lte_prediction_baseline_results")