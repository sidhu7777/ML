import uuid
import threading
import pandas as pd
import os
import subprocess
import datetime
from sqlalchemy import create_engine, text

# Global dictionary to track job status
JOBS = {}


def _safe_nunique(df, col):
    return int(df[col].nunique(dropna=True)) if col in df.columns else "n/a"


def _safe_minmax(df, col):
    if col not in df.columns:
        return "n/a"
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return "n/a"
    return f"{series.min():.4f}..{series.max():.4f}"


def _log_df(stage, df):
    print(f"[TILT][{stage}] shape={df.shape}")
    print(f"[TILT][{stage}] columns={list(df.columns)}")
    for col in ["cell_id", "nodeb_id", "node_b_id", "operator", "final_operator", "Technology"]:
        if col in df.columns:
            print(f"[TILT][{stage}] distinct_{col}={_safe_nunique(df, col)}")
    for col in ["rsrp", "rsrq", "sinr", "pred_rsrp", "pred_rsrq", "pred_sinr"]:
        if col in df.columns:
            print(f"[TILT][{stage}] range_{col}={_safe_minmax(df, col)}")

# ==========================================================
# MULTI-REGION DATABASE ENGINES
# Aggressive Connection Recycling prevents SQLAlchemy from using "dead" connections
# ==========================================================
engine = {
    # India Database (uses your original DATABASE_URL)
    "india": create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL") else None,
    
    # Taiwan Database (Matches your exact .env variable name)
    "taiwan": create_engine(
        os.getenv("DATABASE_URL_Taiwan"), 
        pool_size=10, max_overflow=20, pool_recycle=300, pool_pre_ping=True
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}

class RFOptimizationService:

    def submit(self, cfg):
        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"status": "queued"}
        threading.Thread(target=self._run, args=(job_id, cfg), daemon=True).start()
        return {"job_id": job_id}

    def get(self, job_id):
        return JOBS.get(job_id)

    def _get_next_scenario_id(self, project_id, current_engine):
        """Finds the next scenario number for a given project using the correct regional DB."""
        query = text("SELECT COALESCE(MAX(scenario_id), 0) + 1 FROM rf_optimization_results WHERE project_id = :pid")
        try:
            with current_engine.connect() as conn:
                result = conn.execute(query, {"pid": project_id}).scalar()
                return int(result) if result else 1
        except Exception as e:
            print(f"Error fetching scenario ID: {e}")
            return 1

    def _run(self, job_id, cfg):
        try:
            # ==========================================
            # 1. REGION & ENGINE SELECTION
            # ==========================================
            region = str(cfg.get("region", "india")).lower()
            current_engine = engine.get(region)
            
            if current_engine is None:
                raise ValueError(f"Database config for region '{region}' not found or .env variable is missing.")

            self._update(job_id, "running", f"Initializing {region.upper()} optimization job...")
            project_id = cfg["project_id"]
            print(
                f"[TILT][JOB_START] job_id={job_id} project_id={project_id} region={region} "
                f"operator={cfg.get('operator')} rsrp={cfg.get('rsrp', -105)} "
                f"rsrq={cfg.get('rsrq', -15)} sinr={cfg.get('sinr', 0)}"
            )
            
            operator_input = cfg.get("operator")
            is_all_operators = not operator_input or str(operator_input).lower() in ["all", "none", ""]

            r_thresh = str(cfg.get("rsrp", -105))
            q_thresh = str(cfg.get("rsrq", -15))
            s_thresh = str(cfg.get("sinr", 0))

            # ==========================================
            # 2. FETCH ANTENNA DATA FIRST (Prevents Timeout)
            # ==========================================
            self._update(job_id, "running", "Fetching antenna records...")
            ant_query = text("SELECT * FROM site_prediction WHERE tbl_project_id = :pid")
            with current_engine.connect() as conn:
                antenna_df = pd.read_sql(ant_query, conn, params={"pid": project_id})
            _log_df("ANTENNA_FETCH", antenna_df)

            # ==========================================
            # 3. FETCH LARGE LOG DATA IN CHUNKS
            # ==========================================
            self._update(job_id, "running", "Downloading large log data in chunks...")
            
            # Select only needed columns to speed up the network transfer without DB indexing
            log_cols = "node_b_id, cell_id, operator, pred_rsrp, pred_rsrq, pred_sinr, lat, lon"
            
            if not is_all_operators:
                log_query = text(f"SELECT {log_cols} FROM lte_prediction_baseline_results WHERE project_id = :pid AND operator = :op")
                log_params = {"pid": project_id, "op": operator_input}
            else:
                log_query = text(f"SELECT {log_cols} FROM lte_prediction_baseline_results WHERE project_id = :pid")
                log_params = {"pid": project_id}

            log_dfs = []
            with current_engine.connect() as conn:
                for chunk in pd.read_sql(log_query, conn, params=log_params, chunksize=50000):
                    print(f"[TILT][BASELINE_FETCH] chunk_rows={len(chunk)}")
                    log_dfs.append(chunk)

            if not log_dfs:
                raise ValueError(f"No log data found for project {project_id}")
            
            log_df = pd.concat(log_dfs, ignore_index=True)
            del log_dfs # Free up memory
            _log_df("BASELINE_FETCH_COMBINED", log_df)

            # ==========================================
            # 4. ROBUST OPERATOR MAPPING
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
            print(f"[TILT][OPERATOR_MAP] mapped_keys={len(operator_map)}")

            # ==========================================
            # 5. PREPARE PATHS & TRIGGER SCRIPT
            # ==========================================
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

            scenario_id = self._get_next_scenario_id(project_id, current_engine)
            script_path = os.path.normpath(os.path.join(current_dir, "etilt_optimizer_cd2.py"))
            
            process = subprocess.run(
                ["python", script_path, log_csv, ant_csv, r_thresh, q_thresh, s_thresh],
                capture_output=True, text=True
            )
            if process.stdout:
                print("[TILT][SCRIPT_STDOUT_BEGIN]")
                print(process.stdout)
                print("[TILT][SCRIPT_STDOUT_END]")

            if process.returncode != 0:
                raise Exception(f"Script Error: {process.stderr}")

            # ==========================================
            # 6. SAVE RESULTS MAPPING CORRECT OPERATOR
            # ==========================================
            self._update(job_id, "running", "Saving recommendations to database...")
            
            output_file = os.path.join(temp_dir, "RF_Optimization_Report.xlsx")
            reco_df = pd.read_excel(output_file, sheet_name="Recommendations")
            _log_df("RECOMMENDATIONS_FETCH", reco_df)
            
            # Ensure "ALL" is replaced by the actual cell's operator
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
            _log_df("DB_PAYLOAD", db_save_df)
            print(
                f"[TILT][DB_WRITE] table=rf_optimization_results mode=append "
                f"rows={len(db_save_df)} project_id={project_id} scenario_id={scenario_id}"
            )

            # Fast DB saving with chunks using the correct regional engine
            with current_engine.begin() as conn:
                db_save_df.to_sql("rf_optimization_results", conn, if_exists="append", index=False, method="multi", chunksize=1000)

            JOBS[job_id].update({"status": "done", "output": output_file, "scenario": scenario_id})

        except Exception as e:
            print(f"Error in RF Service: {str(e)}")
            JOBS[job_id].update({"status": "failed", "error": str(e)})

    def _update(self, job_id, status, msg):
        JOBS[job_id].update({"status": status, "progress": msg})
