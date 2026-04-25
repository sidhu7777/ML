import uuid
import threading
import pandas as pd
import os
import subprocess
import datetime
from sqlalchemy import create_engine, text

# Global dictionary to track job status
JOBS = {}

# 1. ENGINE FIX: Aggressive Connection Recycling
# This prevents SQLAlchemy from using "dead" connections
engine = create_engine(
    os.getenv("DATABASE_URL"),
    pool_size=10,
    max_overflow=20,
    pool_recycle=300,        # Drops connections older than 5 minutes
    pool_pre_ping=True       # Pings the DB to check if it's alive before querying
)

class RFOptimizationService:

    def submit(self, cfg):
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"status": "queued"}
        threading.Thread(target=self._run, args=(job_id, cfg), daemon=True).start()
        return {"job_id": job_id}

    def get(self, job_id):
        return JOBS.get(job_id)

    def _get_next_scenario_id(self, project_id):
        query = text("SELECT COALESCE(MAX(scenario_id), 0) + 1 FROM rf_optimization_results WHERE project_id = :pid")
        try:
            with engine.connect() as conn:
                result = conn.execute(query, {"pid": project_id}).scalar()
                return int(result) if result else 1
        except Exception:
            return 1

    def _run(self, job_id, cfg):
        try:
            self._update(job_id, "running", "Fetching antenna records...")
            project_id = cfg["project_id"]
            
            operator_input = cfg.get("operator")
            is_all_operators = not operator_input or str(operator_input).lower() in ["all", "none", ""]

            r_thresh = str(cfg.get("rsrp", -105))
            q_thresh = str(cfg.get("rsrq", -15))
            s_thresh = str(cfg.get("sinr", 0))

            # ==========================================
            # STEP 1: Fetch Small Table First (Prevents Timeout)
            # ==========================================
            ant_query = text("SELECT * FROM site_prediction WHERE tbl_project_id = :pid")
            with engine.connect() as conn:
                antenna_df = pd.read_sql(ant_query, conn, params={"pid": project_id})

            # ==========================================
            # STEP 2: Fetch Massive Table in Chunks
            # ==========================================
            self._update(job_id, "running", "Fetching a data")
            
            # Select only needed columns to speed up the network transfer without DB changes
            log_cols = "node_b_id, cell_id, operator, pred_rsrp, pred_rsrq, pred_sinr, lat, lon"
            
            if not is_all_operators:
                log_query = text(f"SELECT {log_cols} FROM lte_prediction_baseline_results WHERE project_id = :pid AND operator = :op")
                log_params = {"pid": project_id, "op": operator_input}
            else:
                log_query = text(f"SELECT {log_cols} FROM lte_prediction_baseline_results WHERE project_id = :pid")
                log_params = {"pid": project_id}

            log_dfs = []
            with engine.connect() as conn:
                # Reading in chunks keeps the connection active and prevents memory crashes
                for chunk in pd.read_sql(log_query, conn, params=log_params, chunksize=50000):
                    log_dfs.append(chunk)

            if not log_dfs:
                raise ValueError(f"No log data found for project {project_id}")
            
            # Combine all chunks into one DataFrame
            log_df = pd.concat(log_dfs, ignore_index=True)
            del log_dfs # Free up memory

            # ==========================================
            # STEP 3: Robust Operator Mapping
            # ==========================================
            self._update(job_id, "running", "Processing optimization script...")

            def clean_id(val):
                s = str(val).strip()
                return s[:-2] if s.endswith(".0") else s

            log_df["clean_key"] = (
                log_df["node_b_id"].apply(clean_id) + "_" + 
                log_df["cell_id"].apply(clean_id)
            )
            operator_map = log_df.drop_duplicates("clean_key").set_index("clean_key")["operator"].to_dict()

            # Prepare Paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.normpath(os.path.join(current_dir, "..", ".."))
            temp_dir = os.path.normpath(os.path.join(root_dir, "outputs", f"temp_{job_id}"))
            os.makedirs(temp_dir, exist_ok=True)

            log_csv = os.path.join(temp_dir, "input_log.csv")
            ant_csv = os.path.join(temp_dir, "input_ant.csv")
            
            log_df_script = log_df.rename(columns={
                "pred_rsrp": "rsrp", "pred_rsrq": "rsrq", 
                "pred_sinr": "sinr", "node_b_id": "nodeb_id"
            })
            
            # Fast disk writing
            log_df_script.to_csv(log_csv, index=False, chunksize=50000)
            antenna_df.to_csv(ant_csv, index=False)

            # Trigger Script
            scenario_id = self._get_next_scenario_id(project_id)
            script_path = os.path.normpath(os.path.join(current_dir, "etilt_optimizer_cd2.py"))
            
            process = subprocess.run(
                ["python", script_path, log_csv, ant_csv, r_thresh, q_thresh, s_thresh],
                capture_output=True, text=True
            )

            if process.returncode != 0:
                raise Exception(f"Script Error: {process.stderr}")

            # ==========================================
            # STEP 4: Save Results mapping Operator
            # ==========================================
            self._update(job_id, "running", "Saving recommendations to database...")
            
            output_file = os.path.join(temp_dir, "RF_Optimization_Report.xlsx")
            reco_df = pd.read_excel(output_file, sheet_name="Recommendations")
            
            reco_df["final_operator"] = reco_df["Cell ID"].astype(str).map(operator_map).fillna(
                operator_input if (operator_input and not is_all_operators) else "Unknown"
            )

            db_save_df = pd.DataFrame({
                "project_id": project_id,
                "scenario_id": scenario_id,
                "operator": reco_df["final_operator"],
                "cell_id": reco_df["Cell ID"],
                "technology": reco_df["Technology"],
                "parameter": reco_df["Parameter"],
                "current_value": reco_df["Current Value"],
                "recommended_value": reco_df["Recommended Value"],
                "reason": reco_df["Reason"],
                "swap_sector_detected": reco_df["Swap Sector Detected"],
                "rsrp_threshold": float(r_thresh),
                "rsrq_threshold": float(q_thresh),
                "sinr_threshold": float(s_thresh),
                "created_at": datetime.datetime.now()
            })

            # Fast DB saving
            with engine.begin() as conn:
                db_save_df.to_sql("rf_optimization_results", conn, if_exists="append", index=False, method="multi", chunksize=1000)

            JOBS[job_id].update({"status": "done", "output": output_file, "scenario": scenario_id})

        except Exception as e:
            print(f"Error in RF Service: {str(e)}")
            JOBS[job_id].update({"status": "failed", "error": str(e)})

    def _update(self, job_id, status, msg):
        JOBS[job_id].update({"status": status, "progress": msg})