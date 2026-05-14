# Python ML Backend

Technical documentation for the LTE prediction, optimization, and recommendation services implemented in this repository.

This document describes the system architecture, processing workflows, database interactions, API behavior, and implementation structure of the following modules:

- `LTE Prediction`
- `LTE Prediction Optimisation`
- `Tilt Recommendation`

It is intended to serve as the primary reference for engineering onboarding, design review, operational understanding, and ongoing maintenance.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [End-to-End Data Flow](#end-to-end-data-flow)
4. [Module 1: LTE Prediction](#module-1-lte-prediction)
5. [Module 2: LTE Prediction Optimisation](#module-2-lte-prediction-optimisation)
6. [Module 3: Tilt Recommendation](#module-3-tilt-recommendation)
7. [Database Layer](#database-layer)
8. [API Layer](#api-layer)
9. [Configuration and Environment](#configuration-and-environment)
10. [Deployment and Runtime Notes](#deployment-and-runtime-notes)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Code Navigation Guide](#code-navigation-guide)

---

# 1. Project Overview

## 1.1 What this system does

This backend exposes production APIs for three LTE-focused workflows:

1. `LTE Prediction`
   - Generates baseline LTE coverage predictions for all sectors in a project.
   - Produces `RSRP`, `RSRQ`, and `SINR` prediction grids.
   - Applies geo-aware correction logic and drive-test visual blending.
   - Saves the resulting baseline prediction surface and derived geo features into database tables.

2. `LTE Prediction Optimisation`
   - Re-runs prediction only for the cells/sites affected by edited site parameters.
   - Uses `site_prediction` as the baseline site configuration.
   - Uses `site_prediction_optimized` as the DB-managed override/change set.
   - Detects changed cells, computes affected scope, calibrates local path-loss parameters, and predicts only impacted cells.
   - Stores project-specific optimization scenarios and optimized output rows.

3. `Tilt Recommendation`
   - Analyzes baseline LTE prediction data and geo features to identify poor-performing cells.
   - Detects bad sample concentrations, swap-sector suspicion, directional mismatch, blockage/NLOS patterns, and overshoot/coverage imbalance.
   - Generates parameter recommendations such as:
     - `ETilt`
     - `Azimuth`
     - `TX Power`
   - Exports an Excel report and persists recommendation rows into the database.

## 1.2 Business purpose of each module

| Module | Primary Business Goal | Output Type |
|---|---|---|
| LTE Prediction | Produce project-wide baseline RF coverage used by dashboards, validation, and downstream optimization | Prediction grid + geo feature records |
| LTE Prediction Optimisation | Simulate impact of saved site edits and generate optimized affected-area predictions | Optimized prediction grid + scenario history |
| Tilt Recommendation | Convert poor-coverage evidence into actionable RF tuning recommendations | Excel report + recommendation rows |

## 1.3 How the modules are interconnected

The modules are coupled by data, not by direct synchronous calls.

```text
site_prediction
    |
    v
LTE Prediction
    | writes
    +--> lte_prediction_baseline_results
    +--> lte_prediction_geo_features
                  |
                  +---------------------------+
                  |                           |
                  v                           v
Tilt Recommendation                 LTE Prediction Optimisation
reads baseline + geo + sites        reads baseline + geo + sites + optimized sites
```

### Interdependency summary

- `LTE Prediction` is the foundational workflow.
- `LTE Prediction Optimisation` depends on baseline outputs produced by `LTE Prediction`.
- `Tilt Recommendation` also depends on baseline outputs and geo features produced by `LTE Prediction`.

## 1.4 Expected end-to-end inputs and outputs

### Inputs

- Project metadata:
  - `project_id`
  - `region`
  - operator / cluster
- Site inventory:
  - antenna coordinates
  - azimuth
  - tilt
  - power
  - height
- Drive-test or measurement data
- Building polygons
- Project polygon geometry
- Optional optimized site edits in `site_prediction_optimized`

### Outputs

- Baseline LTE predictions:
  - `pred_rsrp`
  - `pred_rsrq`
  - `pred_sinr`
- Geo feature records for each predicted point
- Optimized prediction records for affected cells only
- Tilt recommendation rows and downloadable Excel report

---

# 2. Architecture Flow

## 2.1 High-level service architecture

```text
Flask App
  |
  +--> /api/lte-prediction/run
  |       -> tools/lte_prediction/routes.py
  |       -> tools/lte_prediction/services.py
  |       -> tools/lte_prediction/ml_engine.py
  |       -> tools/lte_prediction/geo_correction_pipeline.py
  |       -> tools/lte_prediction/dem_utils.py
  |
  +--> /api/lte-prediction-optimised/optimized
  |       -> tools/lte_prediction_optimised/routes.py
  |       -> tools/lte_prediction_optimised/services.py
  |       -> tools/lte_prediction_optimised/ml_engine.py
  |
  +--> /api/lte-tilt-recommandation/optimize
          -> tools/lte_tilt_recommandation/routes.py
          -> tools/lte_tilt_recommandation/services.py
          -> tools/lte_tilt_recommandation/etilt_optimizer_cd2.py
          -> tools/lte_tilt_recommandation/geo_logic.py
```

## 2.2 Execution sequence: LTE Prediction

```text
Client -> POST /api/lte-prediction/run
  -> route validates payload and builds config
  -> service starts background thread
  -> fetch site_prediction rows
  -> resolve operator
  -> ensure DEM exists for project polygon
  -> fetch drive data from network log tables
  -> fetch building polygons
  -> run source RF prediction grid
  -> filter prediction output to project polygon
  -> apply geo correction pipeline
  -> apply drive-test visual overlay
  -> save baseline prediction rows
  -> save geo feature rows
  -> persist temp CSV for download/debug
```

## 2.3 Execution sequence: LTE Prediction Optimisation

```text
Client -> POST /api/lte-prediction-optimised/optimized
  -> route validates minimal payload
  -> service creates scenario row
  -> fetch baseline prediction state
  -> fetch site_prediction baseline sites
  -> fetch site_prediction_optimized edited rows
  -> overlay optimized rows onto baseline site inventory
  -> detect changed cells by before/after comparison
  -> compute affected sites and cells
  -> calibrate local K1/K2 for changed cells only
  -> run prediction only for affected cells
  -> apply saved geo correction
  -> save CSV
  -> append optimized prediction results
  -> update scenario status
```

## 2.4 Execution sequence: Tilt Recommendation

```text
Client -> POST /api/lte-tilt-recommandation/optimize
  -> service starts background thread
  -> fetch site_prediction antenna data
  -> fetch baseline prediction rows in chunks
  -> normalize and enrich log rows with antenna context
  -> fetch geo feature rows
  -> write input CSVs to temp output folder
  -> run etilt_optimizer_cd2.py as subprocess
  -> script filters bad samples
  -> script detects swap sectors
  -> script computes dominant bearing summary
  -> script aggregates geo context
  -> script generates recommendations + forecast
  -> script exports RF_Optimization_Report.xlsx
  -> service loads Recommendations sheet
  -> service maps per-cell operator
  -> append rf_optimization_results
```

## 2.5 Database interactions overview

| Workflow | Reads | Writes |
|---|---|---|
| LTE Prediction | `site_prediction`, `tbl_network_log`, `tbl_network_log_neighbour`, `tbl_savepolygon`, `map_regions` | `lte_prediction_baseline_results`, `lte_prediction_geo_features` |
| LTE Prediction Optimisation | `lte_prediction_baseline_results`, `site_prediction`, `site_prediction_optimized`, `lte_prediction_geo_features`, `lte_optimization_scenarios` | `lte_optimization_scenarios`, `lte_prediction_optimised_results` |
| Tilt Recommendation | `site_prediction`, `lte_prediction_baseline_results`, `lte_prediction_geo_features`, `rf_optimization_results` | `rf_optimization_results` |

---

# 3. Module-wise Detailed Explanation

# 3.1 LTE Prediction

## Purpose of the Module

This module creates the baseline RF prediction state for a project.

It exists because downstream optimization and recommendation workflows need:

- a stable project-wide prediction grid,
- consistent serving-cell-level KPI estimates,
- geo-context features at each prediction point,
- and a visualization-friendly prediction layer that blends engineering prediction with nearby field evidence.

From a telecom perspective, it solves the problem of generating a reproducible LTE coverage surface from:

- site configuration,
- project polygon,
- building footprint context,
- drive-test samples,
- and terrain/elevation data.

## Input Data

### API payload

Endpoint: `POST /api/lte-prediction/run`

| Field | Required | Type | Meaning |
|---|---|---|---|
| `project_id` | Yes | int | Project identifier |
| `session_ids` | Yes | list[int] | Drive-test session IDs used for data extraction |
| `region` | No | string | DB region; defaults to `india` |
| `radius` or `radius_m` | No | float | Prediction radius in meters; default `500` |
| `grid_resolution` | No | float | Grid spacing in meters; default `25` |
| `n_workers` | No | int | Parallel worker count; defaults to `CPU - 1` |
| `max_interference_sites` | No | int | Max nearby interferers used by source RF engine; default `50` |
| `dem_raster_path` | No | string | Explicit DEM path; otherwise auto-generated |

### Source tables

#### `site_prediction`

Used to fetch baseline LTE site and antenna configuration.

Important columns used after normalization:

| Source Column | Normalized Column |
|---|---|
| `latitude` | `lat` |
| `longitude` | `lon` |
| `e_tilt` | `Etilt` |
| `m_tilt` | `Mtilt` |
| `height` | `Height` |
| `pci` | `PCI` |
| `site` | `Site ID` |
| `cluster` | `network` |

#### `tbl_network_log` and `tbl_network_log_neighbour`

Used for drive-test extraction.

Required output columns:

- `lat`
- `lon`
- `rsrp`
- `rsrq`
- `sinr`
- `cell_id`
- `nodeb_id`
- `pci`
- `earfcn`

#### `tbl_savepolygon`

Used for building geometry extraction.

#### `map_regions`

Used for project polygon selection and polygon clipping.

### Validation rules

- Site data must exist for the project.
- Required site columns after normalization:
  - `lat`
  - `lon`
  - `Etilt`
  - `Mtilt`
  - `Height`
  - `tx_power`
- Drive data is filtered by:
  - requested session IDs
  - operator
  - primary-serving flag
- If no project polygon is found, polygon clipping falls back or fails depending on stage.

## Data Preprocessing

### Site data preprocessing

Implemented mainly in `fetch_site_data()` in `tools/lte_prediction/ml_engine.py`.

Steps:

1. Read raw `site_prediction`.
2. Normalize column names and aliases.
3. Convert RF columns to numeric.
4. Fill missing defaults:
   - `Etilt = 3`
   - `Mtilt = 0`
   - `Height = 30`
   - `tx_power = 46`
   - `frequency_mhz = 1800` if unavailable
5. Build `Node_Cell_ID` from `cell_id` if not already present.
6. Drop rows missing `lat/lon`.
7. Resolve operator from `network`, `cluster`, `operator`, or `Technology`.

### Drive data preprocessing

Implemented in `fetch_drive_data()`.

Steps:

1. Pull matching rows from both main and neighbour tables.
2. Keep only matching operator.
3. Keep only primary-serving rows for the main run dataset.
4. Normalize identifier fields to strings.
5. Cache result in `cache/drive_<project>_<operator>_<sessions>.parquet`.
6. Clip rows to project polygon.
7. Retry polygon containment using swapped XY coordinates if direct containment gives no rows.

### Building and polygon preprocessing

Implemented in `geo_correction_pipeline.py`.

Key operations:

- Parse WKT/WKB geometry from multiple possible columns.
- Align project polygons to points if lat/lon orientation is inverted.
- Align building geometry orientation to the project polygon.
- Convert building polygons to GeoDataFrames for downstream feature extraction.

### DEM preprocessing

Implemented in `dem_utils.py`.

Steps:

1. Read project polygon from `map_regions`.
2. Derive bounding box.
3. Download required SRTM/Skadi tiles.
4. Merge tiles into GeoTIFF.
5. Validate bounds and centroid coverage.
6. Cache output under:
   - `data/dem/project_<project_id>_dem.tif`

## Core Logic

### Phase 1: source RF prediction

`run_rf_prediction_fast()` prepares `site.csv` and `building.csv` and delegates core RF computation to:

- `run_prediction_from_api()` from `Sector_wise_prediction_code_copy.py`

This source RF engine:

- generates prediction grids around sectors,
- computes per-point RSRP/RSRQ/SINR,
- models interference from nearby sites,
- and writes `prediction_ALL_SITES.csv`.

### Phase 2: geo correction and display correction

`run_ml_fast()` calls:

- `apply_full_display_correction()`

This layer adds business-specific realism:

1. Normalize site inventory for geo features.
2. Load best geo weights if optimizer output exists, otherwise use default weights.
3. Create analysis grid over the project polygon.
4. Attach:
   - building density metrics
   - road/green/water ratios
   - LOS/NLOS/path blockage features
   - terrain features from DEM
   - serving distance / nearest-site metrics
5. Compute weighted geo offsets.
6. Blend physical prediction with fixed-serving proxy metrics.
7. Blend final display KPIs toward nearby drive-test points.

### Why the geo layer exists

Pure propagation calculation is often too smooth for production visualization. This layer deliberately introduces context-aware adjustments so the display surface better reflects:

- dense urban clutter loss,
- NLOS/blockage,
- strong off-axis degradation,
- interference asymmetry,
- and drive-test evidence near known measurement points.

## Model Explanation

This module is not a single train-once ML model in the standard sense.

It is a hybrid prediction stack:

1. Source RF computation:
   - path-loss / antenna geometry / interference logic from `Sector_wise_prediction_code_copy.py`
2. Feature engineering:
   - polygon/building/road/terrain/site density context
3. Heuristic weighted geo correction:
   - `DEFAULT_GEO_WEIGHTS` and optional `best_weights.csv`
4. Display-time drive-test overlay:
   - replaces or blends prediction near measured points

### Algorithmic components used

| Component | Purpose |
|---|---|
| COST/path-loss style calibration | Estimate signal decay behavior |
| 3GPP antenna gain calculations | Sector directional modeling |
| `KMeans` | Morphology clustering in geo pipeline |
| `BallTree` | Nearest-neighbor lookups |
| Weighted correction rules | Adjust predicted KPIs using geo features |

### Evaluation metrics

Defined in `_metric_bundle()`:

- `MAE`
- `RMSE`
- `R2`
- `Bias`
- `P50 absolute error`
- `P90 absolute error`
- threshold hit-rates such as `within_3`, `within_6`, etc. for KPI-specific bands

### Confidence logic

There is no single confidence score output for baseline prediction, but validation summaries are attached to `final_df.attrs["production_summary"]` and logged:

- baseline validation metrics
- geo-corrected validation metrics
- weights summary used

## Functions Breakdown

### Function: `fetch_site_data(project_id, region="india")`

Purpose:

- Load and normalize baseline site data.

Parameters:

- `project_id`: project identifier
- `region`: database region

Returns:

- normalized site DataFrame
- resolved operator string

Internal logic:

1. Query `site_prediction`.
2. Rename LTE-specific columns.
3. Coerce numeric RF fields.
4. Fill safe defaults.
5. Drop invalid coordinates.
6. Resolve operator.

Dependencies:

- SQLAlchemy engine
- `_resolve_operator()`

Edge cases:

- missing site rows
- missing required columns
- no identifiable operator

### Function: `fetch_drive_data(session_ids, operator, project_id, region="india")`

Purpose:

- Load drive-test samples for the requested sessions and operator.

Returns:

- filtered, polygon-clipped drive DataFrame

Important rules:

- only rows with `primary = yes` are used in the main prediction run
- cache is reused if it contains the required columns

### Function: `run_rf_prediction_fast(site_df, drive_df, building_df, params)`

Purpose:

- Execute the source RF prediction engine.

Returns:

- raw predicted point grid

Important rules:

- building geometry is pre-aligned before CSV export
- output is clipped back to the project polygon

### Function: `apply_full_display_correction(...)`

Purpose:

- Convert raw RF prediction into production display prediction.

Returns:

- corrected prediction DataFrame
- summary dict

Important logic:

- loads weights from optimizer output if present
- attaches geo context
- attaches DEM features
- computes geo offset
- applies DT anchor / DT blend

### Function: `_save_baseline_results(...)`

Purpose:

- Persist final project-wide prediction state.

Writes:

- `lte_prediction_baseline_results`
- `lte_prediction_geo_features`

Important rules:

- baseline table uses delta-upsert semantics
- geo feature table uses delete/insert delta semantics for changed keys

## Database Layer

### Table: `lte_prediction_baseline_results`

Stores final display prediction KPIs.

Important columns:

- `project_id`
- `job_id`
- `lat`, `lon`
- `lat_6dp`, `lon_6dp`
- `pred_rsrp`
- `pred_rsrq`
- `pred_sinr`
- `node_b_id`
- `cell_id`
- `operator`
- `site_id`
- `nodeb_id_cell_id`
- `Technology`

Behavior:

- not simple append
- implemented as delta-upsert through staging table + `ON DUPLICATE KEY UPDATE`

### Table: `lte_prediction_geo_features`

Stores geo and propagation context aligned to baseline prediction points.

Important columns:

- clutter/morphology features
- building density metrics
- LOS/NLOS metrics
- DEM-based terrain metrics
- serving and neighbour distance metrics
- `azimuth_delta_deg`
- `baseline_job_id`

Behavior:

- changed rows are reinserted
- stale rows can be deleted

## API Layer

### `POST /api/lte-prediction/run`

Example request:

```json
{
  "project_id": 196,
  "session_ids": [101, 102],
  "region": "india",
  "radius": 500,
  "grid_resolution": 25,
  "n_workers": 4,
  "max_interference_sites": 50
}
```

Example response:

```json
{
  "job_id": "..."
}
```

### `GET /api/lte-prediction/status/<job_id>`

Returns queued/running/done/failed state.

## Execution Flow

Exact order:

1. Route validates payload.
2. Service spawns background thread.
3. Site data fetched.
4. DEM resolved.
5. Drive data fetched and clipped.
6. Building data fetched.
7. Source prediction executed.
8. Geo/display correction executed.
9. Baseline and geo features persisted.

## Error Handling

- Job failures are stored in in-memory `JOBS`.
- DEM auto-resolution falls back to requested path if generation fails.
- Polygon XY swap is attempted when containment is empty.
- Cache refresh happens if cached drive parquet lacks required columns.

## Performance Considerations

- Drive-test data is cached in parquet.
- DB writes use staging tables and chunked `to_sql`.
- Geo delta logic avoids rewriting unchanged rows.
- Worker count is configurable.

## Configuration

Important parameters:

- `DATABASE_URL`
- `DATABASE_URL_Taiwan`
- `PORT`
- `HOST`
- `LOG_LEVEL`
- `MAX_CONTENT_LENGTH`
- `OUTPUT_FOLDER`

## Deployment

Minimum setup:

1. Create virtual environment.
2. Install Python dependencies.
3. Provide `.env` with DB URLs.
4. Ensure MySQL-compatible database and tables exist.
5. Start Flask app with `python app.py`.

## Example Walkthrough

1. A project has site inventory and drive sessions.
2. Client calls `/api/lte-prediction/run`.
3. Backend builds prediction grid around all cells.
4. Geo layer adjusts predictions for clutter, blockage, terrain, and interference context.
5. Final prediction points are saved.
6. These outputs become the source for optimization and tilt workflows.

---

# 3.2 LTE Prediction Optimisation

## Purpose of the Module

This module exists to answer:

> “If the frontend user edits site parameters and saves them, what is the predicted RF impact in the affected area?”

It does not recompute the entire project blindly. Instead, it detects changed cells from the database and predicts only the impacted scope.

This solves the telecom workflow where planners:

- move a site,
- rotate azimuth,
- change electrical/mechanical tilt,
- change Tx power,
- change antenna height,

and then want a focused prediction run that is cheaper and faster than full baseline regeneration.

## Input Data

### API payload

Endpoint: `POST /api/lte-prediction-optimised/optimized`

Current minimal payload:

```json
{
  "project_id": 196,
  "region": "india",
  "operator": "Airtel",
  "radius": 500,
  "grid_resolution": 10
}
```

Optional runtime parameters:

| Field | Required | Default |
|---|---|---|
| `n_workers` | No | backend supplied |
| `impact_radius_m` | No | `radius` |
| `neighbor_site_count` | No | `2` |
| `max_interference_sites` | No | `10` |
| `scenario_name` | No | auto-generated |
| `scenario_description` | No | auto-generated |

### Source tables

#### `site_prediction`

Baseline site state for the operator/project.

#### `site_prediction_optimized`

Edited rows saved by frontend or upstream workflow.

Important implementation detail:

- this table is treated as an override table, not a full copy of all site rows
- for project `196`, it contained fewer rows than `site_prediction`
- duplicates are deduplicated by sorting on:
  - `version`
  - `updated_at`
  - `created_at`
  - `id`
  - `Node_Cell_ID`

#### `lte_prediction_baseline_results`

Used as the local calibration/serving reference.

#### `lte_prediction_geo_features`

Used to reapply saved geo correction to optimized predictions.

#### `lte_optimization_scenarios`

Tracks optimization scenario metadata and status.

## Data Preprocessing

### Site normalization

Implemented in `_normalize_site_df()`.

Normalizes:

- `latitude` -> `lat`
- `longitude` -> `lon`
- `e_tilt` -> `electrical_tilt`
- `m_tilt` -> `mechanical_tilt`
- `height` -> `antenna_height`

Builds:

- `Node_Cell_ID`
- `dashboard_site_id`

Fills defaults:

- `electrical_tilt = 0`
- `mechanical_tilt = 0`
- `antenna_height = 30`
- `azimuth = 0`
- `tx_power = 46`
- `frequency_mhz = 1800`

### Optimized overlay preprocessing

Implemented in `fetch_optimized_sites()`.

Steps:

1. Load baseline `site_prediction`.
2. Load raw `site_prediction_optimized`.
3. Normalize both.
4. Deduplicate optimized rows by `Node_Cell_ID`.
5. Preserve baseline values in:
   - `orig_lat`
   - `orig_lon`
   - `orig_azimuth`
   - `orig_electrical_tilt`
   - `orig_mechanical_tilt`
   - `orig_tx_power`
   - `orig_antenna_height`
6. Overlay optimized values onto matching baseline rows.
7. Compute changed mask by comparing before/after values only.

### Why DB comparison replaced payload deltas

The current design assumes frontend persists edited values into `site_prediction_optimized`.

That means the backend should trust the DB state rather than temporary request deltas.

This is more robust because:

- the database becomes the source of truth,
- the run is reproducible,
- and multiple edits can be detected together without reconstructing payload deltas.

## Core Logic

### Scenario creation

Before running prediction, the service creates a scenario row.

Important behavior:

- `lte_optimization_scenarios.id` remains the table row primary key
- `lte_optimization_scenarios.scenario_id` is now project-specific
- next scenario number is:

```sql
SELECT COALESCE(MAX(scenario_id), 0) + 1
FROM lte_optimization_scenarios
WHERE project_id = :project_id
```

- max allowed scenarios per project = `6`

### Changed-cell detection

Implemented in `_build_change_mask()`.

Compared fields:

- `lat`
- `lon`
- `azimuth`
- `electrical_tilt`
- `mechanical_tilt`
- `tx_power`
- `antenna_height`

### Affected-area expansion

Implemented in `_compute_affected_cells()`.

Key rules:

1. Directly changed cells are the seed set.
2. Their site centers are computed for both:
   - old center
   - new center
3. Neighbor sites are considered if they fall within `impact_radius_m`.
4. Neighbor inclusion is direction-aware using azimuth-to-center difference.
5. Up to `neighbor_site_count` sites are added per center.

This is why runtime logs show values such as:

- `changed_cell_count=12`
- `affected_cell_count=39`

### Local K1/K2 calibration

Implemented in `compute_k1k2_for_cells()`.

Only changed cells are calibrated, not the entire project.

Each target cell uses baseline points from `lte_prediction_baseline_results`.

If fewer than 10 baseline rows exist, that cell is skipped for calibration.

### Optimized prediction execution

Implemented in `run_prediction_only_optimized()`.

For each affected cell:

1. Build local interference site subset.
2. Generate grid around the cell.
3. If geo feature rows exist for that cell, mask the generated grid to those known geo points.
4. Compute predictions with calibrated or fallback `K1/K2`.
5. Merge saved geo features.
6. Reapply geo correction.
7. Clip KPI ranges.

## Model Explanation

This module reuses the baseline physical prediction engine and geo-correction logic rather than a separate ML model.

### Main algorithmic components

| Component | Role |
|---|---|
| `calibrate_site()` | Calibrate local path-loss parameters from baseline results |
| `compute_predictions_parallel()` | Parallel per-point RF prediction |
| `generate_grid()` | Build point grid around target cell |
| `apply_experimental_geo_adjustments()` | Reapply saved geo correction logic |

### K1/K2 behavior

The calibration function clips fitted values into safe ranges, for example:

- `K1` clipped to physical bounds
- `K2` clipped to physical bounds

If no local calibration is available for a processed cell:

- `run_prediction_only_optimized()` may fall back to `(0, 0)` for non-changed affected neighbors

## Functions Breakdown

### Function: `fetch_optimized_sites(project_id, operator, region="india")`

Purpose:

- Merge optimized DB edits into baseline site state.

Returns:

- merged site DataFrame with `orig_*` columns and `optimization_applied`

Important rules:

- duplicates in `site_prediction_optimized` are deduplicated
- changed rows are detected by actual field comparison

### Function: `_compute_affected_cells(site_df, impact_radius_m, neighbor_site_count)`

Purpose:

- Expand direct changes into impacted cells/sites.

Returns:

- affected cell IDs
- affected site IDs
- changed rows DataFrame

Important rules:

- old and new site centers are both considered
- direction-aware neighbor filtering is applied

### Function: `compute_k1k2_for_cells(baseline_df, site_df, target_cells)`

Purpose:

- Calibrate path-loss only for changed cells.

Edge cases:

- skips cells with insufficient baseline rows

### Function: `run_prediction_only_optimized(opt_sites, k1k2_map, params)`

Purpose:

- Execute affected-area-only optimized prediction.

Return value:

- optimized prediction DataFrame

Important rules:

- affected cells only
- local interference subset only
- geo mask can reduce generated grid points to precomputed geo points

## Database Layer

### Table: `lte_optimization_scenarios`

Used to track optimization scenario lifecycle.

Important columns:

- `id`:
  - row primary key
- `scenario_id`:
  - project-specific visible scenario number
- `project_id`
- `baseline_job_id`
- `scenario_name`
- `scenario_description`
- `region`
- `operator`
- `status`
- delta columns for audit/history

Status is updated using `id`, not `scenario_id`.

### Table: `lte_prediction_optimised_results`

Stores optimized prediction points.

Important columns:

- `project_id`
- `job_id`
- `lat`
- `lon`
- `pred_rsrp`
- `pred_rsrq`
- `pred_sinr`
- `node_b_id`
- `cell_id`
- `nodeb_id_cell_id`
- `Operator`
- `Technology`
- `site_id`
- `scenario_id`

Important detail:

- results table stores the project-specific `scenario_id`
- it does **not** store the `lte_optimization_scenarios.id` primary key

Write behavior:

- `append`

## API Layer

### `POST /api/lte-prediction-optimised/optimized`

Example request:

```json
{
  "project_id": 196,
  "region": "india",
  "operator": "Airtel",
  "radius": 500,
  "grid_resolution": 10
}
```

Example response:

```json
{
  "job_id": "...",
  "scenario_id": 3,
  "scenario_row_id": 15
}
```

### `GET /api/lte-prediction-optimised/status/<job_id>`

Returns in-memory job state.

### `GET /api/lte-prediction-optimised/download?file=...`

Returns generated CSV.

## Execution Flow

1. Validate minimal payload.
2. Create scenario metadata row.
3. Fetch baseline predictions.
4. Fetch baseline sites.
5. Fetch optimized site overrides.
6. Detect changed cells.
7. Expand affected scope.
8. Calibrate local K1/K2.
9. Run optimized prediction for affected cells.
10. Save CSV and DB rows.
11. Mark scenario `done` or `failed`.

## Error Handling

- Missing required request fields -> `400`
- Missing `site_prediction_optimized` rows -> explicit failure
- More than 6 scenarios per project -> explicit failure
- No calibrated cells from changed rows -> explicit failure

## Performance Considerations

- Does not recompute the whole project
- Uses affected-cell-only processing
- Limits interference rows
- Uses geo masks to reduce point count where geo rows exist
- Worker count is configurable

## Configuration

Backend defaults if frontend does not send them:

- `impact_radius_m = radius`
- `neighbor_site_count = 2`
- `max_interference_sites = 10`
- `n_workers` can be backend controlled

## Deployment

No additional deployment surface beyond the baseline service, but this module requires these tables to be present and populated:

- `lte_prediction_baseline_results`
- `site_prediction`
- `site_prediction_optimized`
- `lte_optimization_scenarios`
- `lte_prediction_geo_features`

## Example Walkthrough

1. Frontend edits 12 cells and saves them into `site_prediction_optimized`.
2. Backend receives a minimal run request.
3. It overlays optimized values onto 703 baseline site rows.
4. It detects 12 directly changed cells.
5. It expands scope to 39 affected cells using neighbor logic.
6. It calibrates `K1/K2` only for changed cells.
7. It predicts only those 39 cells and appends results using the current project-specific `scenario_id`.

---

# 3.3 Tilt Recommendation

## Purpose of the Module

This module generates RF tuning recommendations from the current baseline prediction state.

It is intended for post-prediction optimization analysis, especially where planners need structured advice such as:

- reduce ETilt to recover edge coverage,
- increase ETilt to reduce overlap/interference,
- rotate azimuth toward dominant user bearings,
- or hold changes because geometry indicates blockage or swap-sector suspicion.

It solves the planning question:

> “Which cells are underperforming, why are they underperforming, and what bounded parameter changes are justified?”

## Input Data

### API payload

Endpoint: `POST /api/lte-tilt-recommandation/optimize`

Minimum:

```json
{
  "project_id": 196
}
```

Common fields:

| Field | Required | Default |
|---|---|---|
| `project_id` | Yes | - |
| `region` | No | `india` |
| `operator` | No | all operators if omitted |
| `rsrp` | No | `-105` |
| `rsrq` | No | `-15` |
| `sinr` | No | `0` |

### Source tables

#### `site_prediction`

Used as antenna/site configuration source.

Service-level normalization adds:

- `Node_Cell_ID`
- `lat`
- `lon`
- `electrical_tilt`
- `mechanical_tilt`
- `antenna_height`
- `dashboard_site_id`

#### `lte_prediction_baseline_results`

Used as the log-like KPI source for bad-sample analysis.

Fetched columns:

- `node_b_id`
- `cell_id`
- `operator`
- `pred_rsrp`
- `pred_rsrq`
- `pred_sinr`
- `lat`
- `lon`

#### `lte_prediction_geo_features`

Used to attach blockage, clutter, terrain, and site-density context to bad samples.

#### `rf_optimization_results`

Used only for scenario numbering in current implementation.

## Data Preprocessing

### Antenna preprocessing

Implemented in `_prepare_tilt_antenna_df()`.

Key transformations:

- build `Node_Cell_ID` from `nodeb_id` + `cell_id`
- map `latitude/longitude` to `lat/lon`
- map tilt and height aliases
- derive `dashboard_site_id`

### Baseline/log preprocessing

Implemented in `_prepare_tilt_log_df()`.

Key steps:

1. Normalize `node_b_id` / `nodeb_id`
2. Normalize `cell_id`
3. Build `Node_Cell_ID`
4. Merge antenna context onto baseline rows:
   - `Technology`
   - `lat/lon`
   - `azimuth`
   - `electrical_tilt`
   - `mechanical_tilt`
   - `tx_power`
   - `antenna_height`

This preprocessing was important because test and production previously diverged here.

### Bad-sample filtering

Implemented in `filter_bad_samples()`.

Rules:

- sample is bad if any of:
  - `RSRP_eval < RSRP_THRESH`
  - `RSRQ_eval < RSRQ_THRESH`
  - `SINR_eval < SINR_THRESH`
- technology can be filtered dynamically

### Geo aggregation preprocessing

Implemented in `geo_logic.py`:

- `attach_geo_to_bad_samples()`
- `aggregate_bad_geo_context()`

Geo features are joined using:

- `Node_Cell_ID`
- rounded `lat/lon` (`lat_6dp`, `lon_6dp`)

## Core Logic

### Step 1: identify underperforming cells

The script counts bad samples by cell and KPI type.

This yields a cell-level summary:

- bad RSRP count
- bad RSRQ count
- bad SINR count

### Step 2: detect swap-sector suspicion

Implemented in `detect_swap_sector()`.

This is a protective step. If sector mapping looks wrong, recommendation logic avoids unsafe directional changes.

### Step 3: compute dominant bearing summary

Implemented in `compute_dominant_bearing_summary()` in `geo_logic.py`.

It uses:

- sample bearings from site to bad points
- severity weighting
- binned directional peak analysis

Derived features include:

- `peak_bearing_deg`
- `peak_spread_deg`
- `peak_share`
- `directional_contrast`

These determine whether azimuth steering is trustworthy.

### Step 4: attach blockage and geometry context

Geo features influence whether poor KPI is due to:

- simple coverage shortage,
- overlap/interference,
- dense urban blockage,
- terrain relief,
- or NLOS conditions.

### Step 5: generate geo-aware recommendations

Implemented in `build_geo_aware_recommendations()`.

This is the heart of the module.

It scores candidate actions:

- coverage ETilt opening
- overlap ETilt closing
- azimuth steering
- Tx power support

Then it applies business gates such as:

- minimum bad sample count for action
- NLOS/blocked-geometry suppression
- directional confidence thresholds
- safe ETilt bounds
- azimuth step limits

### Key threshold families

Examples from `geo_logic.py`:

- `MIN_BAD_SAMPLE_COUNT_FOR_ACTION = 25`
- `MIN_BEARING_SAMPLE_COUNT = 30`
- `MAX_AZIMUTH_STEP_DEG = 10`
- `MIN_SAFE_ETILT_DEG = 2`
- `MAX_SAFE_ETILT_DEG = 12`
- `MAX_ETILT_INCREASE_PER_RUN_DEG = 2`
- `MAX_ETILT_DECREASE_PER_RUN_DEG = 2`

### Why these bounds exist

The module deliberately avoids aggressive single-run RF changes.

This is important because:

- tilt and azimuth changes affect neighbors,
- geometric evidence can be noisy,
- and planners usually want bounded, first-cycle recommendations.

### Step 6: export filtered recommendations and forecast

The script keeps:

- rows where `Current Value != Recommended Value`
- or protective statuses such as:
  - `blocked_by_blockage`
  - `hold_swap`

Forecast is separate from recommendation rows.

Forecast fields:

- `Pre-Change`
- `Est. Post-Change`
- `Improvement %`

These are heuristic estimates, not freshly recomputed RF predictions.

## Model Explanation

This module is rule-based and score-based, not an ML inference model.

It uses:

- KPI thresholding,
- directionality analysis,
- blockage/NLOS context,
- bounded action scoring,
- and confidence scoring.

### Confidence logic

Implemented with helper functions such as:

- `_clamp_score()`
- `_confidence_label()`

Confidence is driven by:

- sample count
- directionality quality
- candidate score gap
- geometry severity
- action reliability

### Forecast logic

Implemented in `build_forecast()`.

Current heuristic assumptions:

- `ETilt 1° ~= 4%`
- `Azimuth 5° ~= 6%`
- `Power 1 dB ~= 2.5%`

These are capped by cell to avoid unrealistic estimates.

## Functions Breakdown

### Function: `filter_bad_samples(log_df, allowed_techs)`

Purpose:

- Identify bad baseline prediction points.

Returns:

- full bad sample DataFrame
- per-cell summary DataFrame

Important rules:

- dynamic thresholding based on request values
- technology-aware filter support

### Function: `detect_swap_sector(log_df, antenna_df)`

Purpose:

- Flag possible sector mapping anomalies.

Why it matters:

- directional recommendations should be suppressed if the serving mapping itself is suspect

### Function: `compute_dominant_bearing_summary(log_df, antenna_df, rsrp_thresh, rsrq_thresh, sinr_thresh)`

Purpose:

- Compute directional evidence for azimuth recommendations.

Returns:

- per-cell directional summary

### Function: `attach_geo_to_bad_samples(bad_df, geo_df)`

Purpose:

- Attach point-level geo features to bad samples.

### Function: `aggregate_bad_geo_context(bad_geo_df)`

Purpose:

- Aggregate point-level geo features into cell-level context summaries.

### Function: `build_geo_aware_recommendations(...)`

Purpose:

- Produce final recommendation rows with confidence and status.

Important outputs:

- `Parameter`
- `Current Value`
- `Recommended Value`
- `Reason`
- `Recommendation Status`
- `Recommendation Confidence`
- score columns such as:
  - `Coverage ETilt Score`
  - `Overlap ETilt Score`
  - `Azimuth Score`
  - `TX Power Score`

### Function: `build_forecast(bad_summary, recommendations_df)`

Purpose:

- Create estimated KPI improvement summary.

Important note:

- forecast is heuristic, not a new RF simulation

### Function: `prepare_recommendation_exports(recommendations_df)`

Purpose:

- split full recommendation set into:
  - complete internal set
  - filtered export set

## Database Layer

### Table: `rf_optimization_results`

Stores recommendation rows.

Important columns written by service:

- `project_id`
- `scenario_id`
- `operator`
- `cell_id`
- `technology`
- `parameter`
- `current_value`
- `recommended_value`
- `reason`
- `swap_sector_detected`
- `rsrp_threshold`
- `rsrq_threshold`
- `sinr_threshold`
- `created_at`

Write behavior:

- `append`

Important note:

- forecast fields are **not** currently stored in DB
- only recommendation rows are stored

## API Layer

### `POST /api/lte-tilt-recommandation/optimize`

Example request:

```json
{
  "project_id": 196,
  "operator": "Airtel",
  "region": "india",
  "rsrp": -105,
  "rsrq": -15,
  "sinr": 0
}
```

Example response:

```json
{
  "job_id": "..."
}
```

### `GET /api/lte-tilt-recommandation/status/<job_id>`

Returns current job state.

### `GET /api/lte-tilt-recommandation/download?file=...`

Downloads the generated Excel file.

## Execution Flow

1. Fetch antenna data.
2. Fetch baseline rows in chunks.
3. Normalize and enrich data.
4. Fetch geo features.
5. Write CSV inputs.
6. Run `etilt_optimizer_cd2.py`.
7. Read Excel `Recommendations` sheet.
8. Map final operator per cell.
9. Append DB rows.

## Error Handling

- missing DB config for region -> fail
- no baseline rows -> fail
- subprocess non-zero exit -> fail
- operator mapping fallback -> `Unknown` if not derivable

## Performance Considerations

- baseline fetch uses `chunksize=50000`
- CSV writing is chunk-friendly
- recommendation persistence uses `chunksize=1000`
- heavy recommendation logic is isolated in a subprocess to keep service layer simple

## Configuration

Thresholds come from request payload or defaults:

- `RSRP = -105`
- `RSRQ = -15`
- `SINR = 0`

Many scoring and gating constants are hard-coded in:

- `tools/lte_tilt_recommandation/geo_logic.py`

## Deployment

This workflow depends on:

- baseline prediction data already existing
- geo features already existing
- Python environment with:
  - `pandas`
  - `openpyxl`

## Example Walkthrough

1. Baseline project predictions already exist.
2. Planner requests tilt recommendation for project `196`.
3. Service pulls baseline and geo rows.
4. Script identifies bad samples and directional evidence.
5. Script produces rows like:
   - `ETilt: 6 -> 5`
   - `Azimuth: 130 -> 130`
   - `TX Power: 46 -> 47`
6. Service stores recommendation rows in `rf_optimization_results`.
7. Forecast remains in the Excel report only.

---

# 4. Code-Level Documentation Notes

## 4.1 Important design patterns in this codebase

### Background job pattern

All three workflows use in-memory job tracking:

```python
JOBS[job_id] = {"status": "queued"}
threading.Thread(target=self._run, args=(job_id, cfg), daemon=True).start()
```

This means:

- job state is process-local,
- restarting the Flask process loses in-memory job state,
- DB persistence exists for outputs, but not for the job queue itself.

### Multi-region DB selection

Each module keeps a region-to-engine map:

```python
engine = {
    "india": create_engine(os.getenv("DATABASE_URL"), ...),
    "taiwan": create_engine(os.getenv("DATABASE_URL_Taiwan"), ...)
}
```

### Polygon alignment fallback

Several stages explicitly handle lat/lon inversion by comparing direct-hit and swapped-XY hit counts before choosing geometry orientation.

### Delta-write strategy

This codebase is intentionally not uniform:

| Table | Write Mode |
|---|---|
| `lte_prediction_baseline_results` | delta upsert |
| `lte_prediction_geo_features` | delta replace/delete-insert |
| `lte_prediction_optimised_results` | append |
| `rf_optimization_results` | append |

---

# 5. Environment and Configuration

## 5.1 Flask application

Main entry point: [app.py](/C:/ml/python-ml-backend_23042026/app.py)

Important app-level settings:

- `MAX_CONTENT_LENGTH = 100MB`
- CORS enabled
- uploads and outputs folders auto-created
- health endpoint at `/health`

## 5.2 Key environment variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | India database |
| `DATABASE_URL_Taiwan` | Taiwan database |
| `PORT` | Flask port |
| `HOST` | Flask host |
| `LOG_LEVEL` | Runtime logging level |
| `SECRET_KEY` | Flask secret |
| `MAX_CONTENT_LENGTH` | Upload size limit |

## 5.3 Local run example

```powershell
$env:FLASK_ENV="development"
$env:LOG_LEVEL="INFO"
venv\Scripts\python.exe app.py
```

---

# 6. API Summary

| Module | Endpoint | Method | Purpose |
|---|---|---|---|
| LTE Prediction | `/api/lte-prediction/run` | `POST` | Run baseline prediction |
| LTE Prediction | `/api/lte-prediction/status/<job_id>` | `GET` | Check job status |
| LTE Prediction Optimisation | `/api/lte-prediction-optimised/optimized` | `POST` | Run affected-area optimized prediction |
| LTE Prediction Optimisation | `/api/lte-prediction-optimised/status/<job_id>` | `GET` | Check optimization job status |
| LTE Prediction Optimisation | `/api/lte-prediction-optimised/download` | `GET` | Download generated CSV |
| Tilt Recommendation | `/api/lte-tilt-recommandation/optimize` | `POST` | Run recommendation job |
| Tilt Recommendation | `/api/lte-tilt-recommandation/status/<job_id>` | `GET` | Check recommendation status |
| Tilt Recommendation | `/api/lte-tilt-recommandation/download` | `GET` | Download Excel report |

---

# 7. Operational Notes for New Developers

## 7.1 If LTE Optimisation looks wrong

Check in this order:

1. Does `site_prediction_optimized` actually contain changed rows for the project/operator?
2. Are `cell_id` values populated even if `nodeb_id` is null?
3. Did dedup reduce multiple versions to one row per `Node_Cell_ID`?
4. Does log show non-zero:
   - `optimized_rows`
   - `changed_rows`
   - `changed_cells`
5. Did scenario creation use project-specific `scenario_id`?

## 7.2 If Tilt Recommendation returns zero DB rows

Check:

1. Did the subprocess succeed?
2. Is the Excel `Recommendations` sheet empty?
3. Are rows filtered out because:
   - `Current Value == Recommended Value`
   - no `blocked_by_blockage` / `hold_swap` rows exist?
4. Did baseline rows receive the expected antenna enrichment?

## 7.3 If DEM generation fails

Check:

1. `DATABASE_URL` or `DATABASE_URL_Taiwan`
2. project polygon existence in `map_regions`
3. internet access to Skadi tile source
4. `rasterio` installation

---

# 8. Recommended Reading Order in Code

For onboarding, read the code in this order:

1. [app.py](/C:/ml/python-ml-backend_23042026/app.py)
2. [tools/lte_prediction/routes.py](/C:/ml/python-ml-backend_23042026/tools/lte_prediction/routes.py)
3. [tools/lte_prediction/services.py](/C:/ml/python-ml-backend_23042026/tools/lte_prediction/services.py)
4. [tools/lte_prediction/ml_engine.py](/C:/ml/python-ml-backend_23042026/tools/lte_prediction/ml_engine.py)
5. [tools/lte_prediction/geo_correction_pipeline.py](/C:/ml/python-ml-backend_23042026/tools/lte_prediction/geo_correction_pipeline.py)
6. [tools/lte_prediction_optimised/services.py](/C:/ml/python-ml-backend_23042026/tools/lte_prediction_optimised/services.py)
7. [tools/lte_prediction_optimised/ml_engine.py](/C:/ml/python-ml-backend_23042026/tools/lte_prediction_optimised/ml_engine.py)
8. [tools/lte_tilt_recommandation/services.py](/C:/ml/python-ml-backend_23042026/tools/lte_tilt_recommandation/services.py)
9. [tools/lte_tilt_recommandation/geo_logic.py](/C:/ml/python-ml-backend_23042026/tools/lte_tilt_recommandation/geo_logic.py)
10. [tools/lte_tilt_recommandation/etilt_optimizer_cd2.py](/C:/ml/python-ml-backend_23042026/tools/lte_tilt_recommandation/etilt_optimizer_cd2.py)

---

# 9. Final Summary

This codebase is best understood as a three-stage LTE workflow:

1. build a baseline project prediction surface,
2. run localized simulation for saved site edits,
3. generate parameter recommendations from the baseline state.

The most important architectural idea is that the modules share state through database tables:

- baseline prediction and geo features are the foundation,
- optimized runs consume and extend that state,
- recommendation logic diagnoses the same baseline state and converts it into actionable RF tuning guidance.
