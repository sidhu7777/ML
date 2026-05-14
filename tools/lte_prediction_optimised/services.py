import uuid
import threading
import pandas as pd
import os
import datetime
import traceback
from sqlalchemy import create_engine, text

# Import your ML engine functions
from .ml_engine import (
    fetch_baseline,
    fetch_site_data,
    fetch_optimized_sites,
    compute_k1k2_for_cells,
    _compute_affected_cells,
    run_prediction_only_optimized,
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


def _resolve_engine(region="india"):
    return engine.get(str(region).lower(), engine["india"])


def _latest_baseline_job_id(project_id, region="india"):
    current_engine = _resolve_engine(region)
    query = text("""
        SELECT job_id
        FROM lte_prediction_baseline_results
        WHERE project_id = :project_id
        ORDER BY created_at DESC
        LIMIT 1
    """)
    with current_engine.connect() as conn:
        row = conn.execute(query, {"project_id": int(project_id)}).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def _build_scenario_name(cfg):
    provided = str(cfg.get("scenario_name", "")).strip()
    if provided:
        return provided
    target_type = str(cfg.get("target_type", "target")).strip() or "target"
    target_id = str(cfg.get("target_id", "")).strip()
    change_labels = []
    if float(cfg.get("delta_lat", 0) or 0) or float(cfg.get("delta_lon", 0) or 0):
        change_labels.append("Site Move")
    if float(cfg.get("delta_azimuth", 0) or 0):
        change_labels.append("Azimuth Change")
    if float(cfg.get("delta_electrical_tilt", 0) or 0):
        change_labels.append("Electrical Tilt Change")
    if float(cfg.get("delta_mechanical_tilt", 0) or 0):
        change_labels.append("Mechanical Tilt Change")
    if float(cfg.get("delta_tx_power", 0) or 0):
        change_labels.append("Tx Power Change")
    if float(cfg.get("delta_antenna_height", 0) or 0):
        change_labels.append("Antenna Height Change")
    if not change_labels:
        change_labels.append("Optimization Change")
    if target_id:
        return f"{' + '.join(change_labels)} - {target_type} {target_id}"
    return " + ".join(change_labels)


def _build_scenario_description(cfg):
    provided = str(cfg.get("scenario_description", "")).strip()
    if provided:
        return provided

    target_type = str(cfg.get("target_type", "")).strip() or "target"
    target_id = str(cfg.get("target_id", "")).strip() or "unknown"
    parts = [f"Target {target_type} {target_id}"]
    if float(cfg.get("delta_lat", 0) or 0) or float(cfg.get("delta_lon", 0) or 0):
        parts.append(
            f"move(lat={float(cfg.get('delta_lat', 0) or 0):.6f}, lon={float(cfg.get('delta_lon', 0) or 0):.6f})"
        )
    if float(cfg.get("delta_azimuth", 0) or 0):
        parts.append(f"delta_azimuth={float(cfg.get('delta_azimuth', 0) or 0):.2f}")
    if float(cfg.get("delta_electrical_tilt", 0) or 0):
        parts.append(f"delta_electrical_tilt={float(cfg.get('delta_electrical_tilt', 0) or 0):.2f}")
    if float(cfg.get("delta_mechanical_tilt", 0) or 0):
        parts.append(f"delta_mechanical_tilt={float(cfg.get('delta_mechanical_tilt', 0) or 0):.2f}")
    if float(cfg.get("delta_tx_power", 0) or 0):
        parts.append(f"delta_tx_power={float(cfg.get('delta_tx_power', 0) or 0):.2f}")
    if float(cfg.get("delta_antenna_height", 0) or 0):
        parts.append(f"delta_antenna_height={float(cfg.get('delta_antenna_height', 0) or 0):.2f}")
    parts.append(
        f"impact_radius_m={float(cfg.get('impact_radius_m', cfg.get('radius', 500)) or 500):.1f}"
    )
    parts.append(f"neighbor_site_count={int(cfg.get('neighbor_site_count', 2) or 2)}")
    parts.append(f"max_interference_sites={int(cfg.get('max_interference_sites', 10) or 10)}")
    return "; ".join(parts)


class LTEPredictionService_optimised:

    def submit(self, cfg):
        job_id = str(uuid.uuid4())
        region = str(cfg.get("region", "india")).lower()
        scenario_id = cfg.get("scenario_id")
        scenario_row_id = cfg.get("scenario_row_id")
        if scenario_id and scenario_row_id:
            scenario_id = int(scenario_id)
            scenario_row_id = int(scenario_row_id)
        else:
            scenario_row_id, scenario_id = self._create_scenario(cfg, job_id, region)
            cfg["scenario_row_id"] = scenario_row_id
            cfg["scenario_id"] = scenario_id

        JOBS[job_id] = {
            "status": "queued",
            "scenario_row_id": scenario_row_id,
            "scenario_id": scenario_id,
            "project_id": int(cfg["project_id"]),
        }

        threading.Thread(
            target=self._run,
            args=(job_id, cfg),
            daemon=True
        ).start()

        return {"job_id": job_id, "scenario_id": scenario_id, "scenario_row_id": scenario_row_id}

    def get(self, job_id):
        return JOBS.get(job_id)

    def _run(self, job_id, cfg):
        scenario_id = cfg.get("scenario_id")
        scenario_row_id = cfg.get("scenario_row_id")
        region = str(cfg.get("region", "india")).lower()
        try:
            print(
                f"[LTE_OPT][JOB_START] job_id={job_id} project_id={cfg['project_id']} "
                f"region={str(cfg.get('region', 'india')).lower()} operator={cfg.get('operator', 'Airtel')} "
                f"radius={cfg.get('radius', 500)} grid_resolution={cfg.get('grid_resolution', 10)} "
                f"n_workers={cfg.get('n_workers')}"
            )
            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "running", region=region, job_id=job_id)
            self._update(job_id, "running", "Loading baseline")

            project_id = cfg["project_id"]
            operator = cfg.get("operator", "Airtel")

            baseline_df = fetch_baseline(project_id, region=region)
            _df_summary("BASELINE_DF", baseline_df)
            baseline_job_id = None
            if "job_id" in baseline_df.columns and not baseline_df["job_id"].dropna().empty:
                baseline_job_id = str(baseline_df["job_id"].dropna().iloc[0]).strip()

            self._update(job_id, "running", "Loading site data")
            site_df = fetch_site_data(project_id, region=region, operator=operator)
            _df_summary("SITE_DF", site_df)

            self._update(job_id, "running", f"Loading optimized sites for {operator}")
            opt_sites = fetch_optimized_sites(project_id, operator, region=region)
            if opt_sites.empty:
                raise ValueError(
                    f"No rows found in site_prediction_optimized for project_id={project_id} operator={operator}"
                )
            _df_summary("OPTIMIZED_SITE_DF", opt_sites)

            self._update(job_id, "running", "Calculating local K1/K2 from optimized DB changes")
            affected_cells, _, changed_rows = _compute_affected_cells(
                opt_sites,
                float(cfg.get("impact_radius_m", cfg.get("radius", 500)) or cfg.get("radius", 500) or 500),
                int(cfg.get("neighbor_site_count", 2) or 2),
            )
            calibration_cells = sorted(changed_rows["Node_Cell_ID"].astype(str).unique().tolist())
            print(
                f"[LTE_OPT][K1K2_LOCAL_SCOPE] changed_cells={len(calibration_cells)} "
                f"affected_cells={len(affected_cells)} calibration_cells={calibration_cells}"
            )
            k1k2_map = compute_k1k2_for_cells(baseline_df, opt_sites, calibration_cells)
            if not k1k2_map:
                raise ValueError("No calibrated cells found from DB-driven optimized site changes")

            params = {
                "radius": cfg.get("radius", 500),
                "grid_resolution": cfg.get("grid_resolution", 10),
                "n_workers": cfg.get("n_workers"),
                "antenna_gain": 18,
                "cable_loss": 2,
                "ue_height": 1.5,
                "frequency_mhz": 1800,
                "bandwidth_mhz": 10,
                "project_id": project_id,
                "region": region,
                "baseline_job_id": baseline_job_id,
                "impact_radius_m": cfg.get("impact_radius_m", cfg.get("radius", 500) or 500),
                "neighbor_site_count": cfg.get("neighbor_site_count", 2) or 2,
                "max_interference_sites": cfg.get("max_interference_sites", 10) or 10,
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

            db_df = self._format_for_db(optimized_df, project_id, job_id, operator, scenario_id=scenario_id)
            _df_summary("OPTIMIZED_DB_PAYLOAD", db_df)

            self._save_to_db(db_df, region=region)

            JOBS[job_id]["output"] = file_path
            JOBS[job_id]["rows"] = len(optimized_df)

            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "done", region=region, job_id=job_id)
            self._update(job_id, "done", "Completed")

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            if scenario_row_id:
                self._update_scenario_status(int(scenario_row_id), "failed", region=region, job_id=job_id)
            print(" ERROR:", traceback.format_exc())

    def _save_csv(self, df, project_id, operator):
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        file_path = os.path.join(
            output_dir,
            f"optimized_{operator}_{project_id}_{timestamp}.csv"
        )

        df.to_csv(file_path, index=False)
        print(f" CSV saved: {file_path}")

        return file_path
    
    def _save_to_db(self, df, region="india"):
        current_engine = _resolve_engine(region)
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
        print(" Data saved to DB")
    
    def _format_for_db(self, df, project_id, job_id, operator, scenario_id=None):
        import datetime

        df = df.copy()

        split_cols = df["Node_Cell_ID"].str.split("_", expand=True)

        if split_cols.shape[1] < 2:
            raise ValueError(" Invalid Node_Cell_ID format")

        df["node_b_id"] = split_cols[0].astype(str)
        df["cell_id"] = split_cols[1].astype(str)
        df["nodeb_id_cell_id"] = df["node_b_id"] + "_" + df["cell_id"]

        df["project_id"] = project_id
        df["job_id"] = job_id
        
        df["Technology"] = "4G"
        df["Operator"] = operator  
        df["created_at"] = datetime.datetime.now()
        df["site_id"] = df["node_b_id"]
        df["scenario_id"] = scenario_id

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
            "Operator",
            "scenario_id",
        ]]

        return final_df

    def _update(self, job_id, status, msg):
        JOBS[job_id]["status"] = status
        JOBS[job_id]["progress"] = msg
        print(f"[{job_id[:6]}] {msg}")

    def _get_next_project_scenario_id(self, project_id, current_engine):
        query = text("""
            SELECT COALESCE(MAX(scenario_id), 0) + 1
            FROM lte_optimization_scenarios
            WHERE project_id = :project_id
        """)
        with current_engine.connect() as conn:
            next_id = conn.execute(query, {"project_id": int(project_id)}).scalar()
        return int(next_id or 1)

    def _create_scenario(self, cfg, job_id, region):
        current_engine = _resolve_engine(region)
        baseline_job_id = cfg.get("baseline_job_id") or _latest_baseline_job_id(cfg["project_id"], region=region)
        public_scenario_id = self._get_next_project_scenario_id(cfg["project_id"], current_engine)
        if public_scenario_id > 6:
            raise ValueError(
                f"Maximum scenario limit reached for project_id={int(cfg['project_id'])}. "
                f"Only 6 scenarios are allowed per project."
            )
        payload = {
            "project_id": int(cfg["project_id"]),
            "scenario_id": public_scenario_id,
            "baseline_job_id": baseline_job_id,
            "scenario_name": _build_scenario_name(cfg),
            "scenario_description": _build_scenario_description(cfg),
            "region": region,
            "operator": str(cfg.get("operator", "Airtel")),
            "target_type": str(cfg.get("target_type", "")).strip() or None,
            "target_id": str(cfg.get("target_id", "")).strip() or None,
            "impact_radius_m": float(cfg.get("impact_radius_m", cfg.get("radius", 500)) or 500),
            "neighbor_site_count": int(cfg.get("neighbor_site_count", 2) or 2),
            "max_interference_sites": int(cfg.get("max_interference_sites", 10) or 10),
            "delta_lat": float(cfg.get("delta_lat", 0) or 0),
            "delta_lon": float(cfg.get("delta_lon", 0) or 0),
            "delta_azimuth": float(cfg.get("delta_azimuth", 0) or 0),
            "delta_electrical_tilt": float(cfg.get("delta_electrical_tilt", 0) or 0),
            "delta_mechanical_tilt": float(cfg.get("delta_mechanical_tilt", 0) or 0),
            "delta_tx_power": float(cfg.get("delta_tx_power", 0) or 0),
            "delta_antenna_height": float(cfg.get("delta_antenna_height", 0) or 0),
            "status": "created",
            "created_by": str(cfg.get("created_by", "backend")),
        }
        insert_sql = text("""
            INSERT INTO lte_optimization_scenarios (
                project_id, scenario_id, baseline_job_id, scenario_name, scenario_description,
                region, operator, target_type, target_id, impact_radius_m,
                neighbor_site_count, max_interference_sites, delta_lat, delta_lon,
                delta_azimuth, delta_electrical_tilt, delta_mechanical_tilt,
                delta_tx_power, delta_antenna_height, status, created_by
            ) VALUES (
                :project_id, :scenario_id, :baseline_job_id, :scenario_name, :scenario_description,
                :region, :operator, :target_type, :target_id, :impact_radius_m,
                :neighbor_site_count, :max_interference_sites, :delta_lat, :delta_lon,
                :delta_azimuth, :delta_electrical_tilt, :delta_mechanical_tilt,
                :delta_tx_power, :delta_antenna_height, :status, :created_by
            )
        """)
        with current_engine.begin() as conn:
            result = conn.execute(insert_sql, payload)
            scenario_row_id = int(result.lastrowid)
        print(
            f"[LTE_OPT][SCENARIO_CREATE] row_id={scenario_row_id} scenario_id={public_scenario_id} job_id={job_id} "
            f"name={payload['scenario_name']!r} description={payload['scenario_description']!r}"
        )
        return scenario_row_id, public_scenario_id

    def _update_scenario_status(self, scenario_row_id, status, region="india", job_id=None):
        current_engine = _resolve_engine(region)
        update_sql = text("""
            UPDATE lte_optimization_scenarios
            SET status = :status,
                baseline_job_id = COALESCE(baseline_job_id, :baseline_job_id),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :scenario_row_id
        """)
        baseline_job_id = None
        try:
            baseline_job_id = _latest_baseline_job_id(JOBS.get(job_id, {}).get("project_id"), region) if job_id and JOBS.get(job_id, {}).get("project_id") else None
        except Exception:
            baseline_job_id = None
        with current_engine.begin() as conn:
            conn.execute(
                update_sql,
                {
                    "status": status,
                    "baseline_job_id": baseline_job_id,
                    "scenario_row_id": int(scenario_row_id),
                },
            )
        print(f"[LTE_OPT][SCENARIO_STATUS] row_id={scenario_row_id} status={status}")
