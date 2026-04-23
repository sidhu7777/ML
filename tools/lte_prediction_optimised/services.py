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

# Database connection
engine = create_engine(os.getenv("DATABASE_URL"))

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
            self._update(job_id, "running", "Loading baseline")

            project_id = cfg["project_id"]
            
            # ✅ Extract the operator sent from the frontend (Defaults to Airtel if missing)
            operator = cfg.get("operator", "Airtel")

            baseline_df = fetch_baseline(project_id)

            self._update(job_id, "running", "Loading site data")
            site_df = fetch_site_data(project_id)

            self._update(job_id, "running", "Calculating K1/K2")
            k1k2_map = compute_k1k2(baseline_df, site_df)

            self._update(job_id, "running", f"Loading optimized sites for {operator}")
            
            # ✅ Pass the operator to fetch only that specific operator's sites
            opt_sites = fetch_optimized_sites(project_id, operator)

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

            self._update(job_id, "running", "Saving CSV")

            # Save the CSV
            file_path = self._save_csv(optimized_df, project_id, operator)

            # ✅ Format for DB (Passing the operator down)
            db_df = self._format_for_db(optimized_df, project_id, job_id, operator)

            # Save to DB
            self._save_to_db(db_df)

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
    
    def _save_to_db(self, df):
        df.to_sql(
            "lte_prediction_optimised_results",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=15000,
            method="multi"
        )
        print("✅ Data saved to DB")
    
    # ✅ Includes 'operator' parameter
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
        df["Operator"] = operator  # ✅ Assign the dynamic operator value
        df["created_at"] = datetime.datetime.now()
        df["site_id"] = df["node_b_id"]

        # ✅ Final column ordering (Operator moved to the very end as requested)
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