from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from shapely.ops import unary_union

from tests.lte_rf_debug_lab import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_GRID_RESOLUTION_M,
    DEFAULT_MAX_INTERFERENCE_SITES,
    DEFAULT_PROJECT_ID,
    DEFAULT_RADIUS_M,
    DEFAULT_REGION,
    DEFAULT_SESSION_IDS,
    DEFAULT_TILE_SIZE_M,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_WORKERS,
    RunConfig,
    run_rf_debug_lab,
)


OUTPUT_ROOT = Path("tests/output")
MAX_SITE_POINTS = 250
MAX_DRIVE_POINTS = 1200
MAX_PRED_POINTS = 3500
MAX_BUILDING_POLYGONS = 250

KPI_LIMITS = {
    "RSRP": (-140, -44),
    "RSRQ": (-20, -3),
    "SINR": (-10, 30),
}


def _list_runs(project_id: int) -> List[Path]:
    root = OUTPUT_ROOT / f"project_{project_id}"
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _load_summary(run_dir: Path) -> Dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _metric_row(title: str, metrics: Dict[str, float]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{title} MAE", metrics.get("mae"))
    c2.metric(f"{title} RMSE", metrics.get("rmse"))
    c3.metric(f"{title} R2", metrics.get("r2"))


def _render_metric_detail_table(summary: Dict, metric_name: str) -> None:
    rows = []
    for series_name in ["baseline", "experimental"]:
        metrics = summary.get("full_metrics", {}).get(series_name, {}).get(metric_name)
        if metrics:
            row = {"series": "Baseline RF" if series_name == "baseline" else "Experimental Geo"}
            row.update(metrics)
            rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _build_map(
    polygon_gdf: gpd.GeoDataFrame,
    site_df: pd.DataFrame,
    drive_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    grid_gdf: Optional[gpd.GeoDataFrame] = None,
    show_geo: bool = False,
    kpi_col: str = "pred_rsrp",
    selected_sector: Optional[str] = None,
    selected_nodeb: Optional[str] = None,
    show_site_markers: bool = True,
) -> folium.Map:
    center_source = site_df if not site_df.empty else pred_df
    center = [float(pd.to_numeric(center_source["lat"], errors="coerce").median()), float(pd.to_numeric(center_source["lon"], errors="coerce").median())]
    fmap = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
        width="100%",
        height=680,
    )

    folium.GeoJson(
        polygon_gdf,
        name="Project Polygon",
        style_function=lambda _: {"color": "#ef4444", "weight": 3, "fillOpacity": 0.0},
    ).add_to(fmap)

    if buildings_gdf is not None and not buildings_gdf.empty:
        step = max(1, len(buildings_gdf) // MAX_BUILDING_POLYGONS)
        b_sample = buildings_gdf.iloc[::step].head(MAX_BUILDING_POLYGONS)
        folium.GeoJson(
            b_sample,
            name="Buildings",
            style_function=lambda _: {"color": "#6b7280", "weight": 1, "fillColor": "#9ca3af", "fillOpacity": 0.2},
        ).add_to(fmap)

    if grid_gdf is not None and not grid_gdf.empty:
        geo_col = "morphology_cluster" if show_geo else "clutter_class"
        if geo_col in grid_gdf.columns:
            color_map = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
            colors = {}
            for idx, key in enumerate(sorted(grid_gdf[geo_col].dropna().astype(str).unique())):
                colors[key] = color_map[idx % len(color_map)]

            def _style(feature):
                key = str(feature["properties"].get(geo_col))
                color = colors.get(key, "#888888")
                return {"color": color, "weight": 1, "fillColor": color, "fillOpacity": 0.18}

            folium.GeoJson(grid_gdf, name=geo_col, style_function=_style).add_to(fmap)

    site_work = _prepare_site_selection_df(site_df)
    for col in ["lat", "lon", "nodeb_id", "cell_id", "Node_Cell_ID", "dashboard_nodeb_id"]:
        if col in site_work.columns:
            site_work[col] = site_work[col].astype(str) if col in {"nodeb_id", "cell_id", "Node_Cell_ID", "dashboard_nodeb_id"} else pd.to_numeric(site_work[col], errors="coerce")
    if selected_sector:
        site_work = site_work[site_work.get("Node_Cell_ID", "").astype(str) == str(selected_sector)].copy()
    elif selected_nodeb and "dashboard_nodeb_id" in site_work.columns:
        site_work = site_work[site_work["dashboard_nodeb_id"].astype(str) == str(selected_nodeb)].copy()

    if show_site_markers and not site_work.empty:
        site_sample = site_work.iloc[::max(1, len(site_work) // MAX_SITE_POINTS)].head(MAX_SITE_POINTS)
        for _, row in site_sample.iterrows():
            tooltip_label = f"Site cell={row.get('Node_Cell_ID', row.get('cell_id'))} nodeb={row.get('dashboard_nodeb_id')}"
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4 if selected_sector or selected_nodeb else 3,
                color="#111827",
                weight=1,
                fill=True,
                fill_color="#111827",
                fill_opacity=0.85,
                tooltip=tooltip_label,
            ).add_to(fmap)

    drive_sample = drive_df.copy()
    rsrp_col = next((c for c in drive_sample.columns if "rsrp" in c.lower()), None)
    if rsrp_col:
        drive_sample["_rsrp"] = pd.to_numeric(drive_sample[rsrp_col], errors="coerce")
        drive_sample = drive_sample.dropna(subset=["_rsrp"])
        drive_sample = drive_sample.iloc[::max(1, len(drive_sample) // MAX_DRIVE_POINTS)].head(MAX_DRIVE_POINTS)
        color_scale = px.colors.sample_colorscale("Turbo", [0.1, 0.4, 0.7, 0.9])
        bins = [-140, -110, -95, -80, -44]
        for _, row in drive_sample.iterrows():
            value = row["_rsrp"]
            color = color_scale[0]
            for idx in range(len(bins) - 1):
                if bins[idx] <= value < bins[idx + 1]:
                    color = color_scale[min(idx, len(color_scale) - 1)]
                    break
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                weight=0,
                tooltip=f"DT RSRP={value:.2f}",
            ).add_to(fmap)

    pred_work = pred_df.copy()
    for col in ["lat", "lon", kpi_col]:
        if col in pred_work.columns:
            pred_work[col] = pd.to_numeric(pred_work[col], errors="coerce")
    if "Node_Cell_ID" in pred_work.columns:
        pred_work["Node_Cell_ID"] = pred_work["Node_Cell_ID"].astype(str)
    if selected_sector and "Node_Cell_ID" in pred_work.columns:
        pred_work = pred_work[pred_work["Node_Cell_ID"] == str(selected_sector)].copy()
    elif selected_nodeb and "Node_Cell_ID" in pred_work.columns and not site_df.empty:
        site_lookup = _prepare_site_selection_df(site_df)
        nodeb_cells = set(
            site_lookup.loc[
                site_lookup["dashboard_nodeb_id"].astype(str) == str(selected_nodeb),
                "Node_Cell_ID" if "Node_Cell_ID" in site_lookup.columns else "cell_id",
            ].astype(str).tolist()
        )
        pred_work = pred_work[pred_work["Node_Cell_ID"].isin(nodeb_cells)].copy()

    pred_sample = pred_work.dropna(subset=["lat", "lon", kpi_col]).copy()
    pred_sample[kpi_col] = pd.to_numeric(pred_sample[kpi_col], errors="coerce")
    pred_sample = pred_sample.dropna(subset=[kpi_col])
    if not polygon_gdf.empty and not pred_sample.empty:
        polygon_union = unary_union(polygon_gdf.geometry)
        pred_points = gpd.GeoDataFrame(
            pred_sample,
            geometry=gpd.points_from_xy(pred_sample["lon"], pred_sample["lat"]),
            crs=polygon_gdf.crs or "EPSG:4326",
        )
        pred_sample = pd.DataFrame(pred_points[pred_points.geometry.within(polygon_union)].drop(columns="geometry"))
    pred_sample = pred_sample.iloc[::max(1, len(pred_sample) // MAX_PRED_POINTS)].head(MAX_PRED_POINTS)
    for _, row in pred_sample.iterrows():
        value = float(row[kpi_col])
        metric_name = "RSRP" if "rsrp" in kpi_col.lower() else "RSRQ" if "rsrq" in kpi_col.lower() else "SINR"
        low, high = KPI_LIMITS[metric_name]
        clipped_value = min(max(value, float(low)), float(high))
        scale_position = (clipped_value - low) / (high - low)
        color = px.colors.sample_colorscale("Viridis", [scale_position])[0]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2 if selected_sector or selected_nodeb else 1,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6 if selected_sector or selected_nodeb else 0.45,
            weight=0,
            tooltip=f"{kpi_col}={value:.2f} cell={row.get('Node_Cell_ID')}",
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def _render_map(fmap: folium.Map, key: str) -> None:
    html = fmap.get_root().render()
    html = html.replace(
        "<style>",
        "<style>html, body {height: 100%; margin: 0;} .folium-map {width: 100% !important; height: 680px !important;}",
        1,
    )
    components.html(html, height=700, scrolling=False)


def _prepare_site_selection_df(site_df: pd.DataFrame) -> pd.DataFrame:
    work = site_df.copy()
    if "Node_Cell_ID" not in work.columns and "cell_id" in work.columns:
        work["Node_Cell_ID"] = work["cell_id"].astype(str)
    for col in ["nodeb_id", "cell_id", "Node_Cell_ID", "Site ID"]:
        if col in work.columns:
            work[col] = work[col].astype(str)
    dashboard_nodeb = pd.Series(index=work.index, dtype=object)
    if "nodeb_id" in work.columns:
        nodeb_series = work["nodeb_id"].astype(str).str.strip()
        dashboard_nodeb = nodeb_series.where(~nodeb_series.isin(["", "nan", "None"]))
    if "Site ID" in work.columns:
        site_id_series = work["Site ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        dashboard_nodeb = dashboard_nodeb.fillna(site_id_series.where(~site_id_series.isin(["", "nan", "None"])))
    if "Node_Cell_ID" in work.columns:
        derived_series = work["Node_Cell_ID"].astype(str).str.split("_").str[0].str.strip()
        dashboard_nodeb = dashboard_nodeb.fillna(derived_series.where(~derived_series.isin(["", "nan", "None"])))
    work["dashboard_nodeb_id"] = dashboard_nodeb.astype(str)
    return work


def _render_metric_compare(summary: Dict, metric_name: str) -> None:
    baseline = summary.get("full_metrics", {}).get("baseline", {}).get(metric_name)
    if baseline:
        st.markdown(f"**{metric_name} Baseline**")
        _metric_row(metric_name.replace("_meas", ""), baseline)
    experimental = summary.get("full_metrics", {}).get("experimental", {}).get(metric_name)
    if experimental:
        st.markdown(f"**{metric_name} Experimental Geo Model**")
        _metric_row(metric_name.replace("_meas", ""), experimental)


def _clip_series(series: pd.Series, metric_name: str) -> pd.Series:
    low, high = KPI_LIMITS[metric_name]
    return pd.to_numeric(series, errors="coerce").clip(lower=low, upper=high)


def _prepare_kpi_eval(dt_eval: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    meas_col = f"{metric_name}_meas"
    pred_col = f"{metric_name}_pred"
    geo_pred_col = f"{metric_name}_pred_geo"
    cols = [c for c in ["lat", "lon", meas_col, pred_col, geo_pred_col, "morphology_cluster"] if c in dt_eval.columns]
    work = dt_eval[cols].copy()
    if meas_col in work.columns:
        work[meas_col] = _clip_series(work[meas_col], metric_name)
    if pred_col in work.columns:
        work[pred_col] = _clip_series(work[pred_col], metric_name)
    if geo_pred_col in work.columns:
        work[geo_pred_col] = _clip_series(work[geo_pred_col], metric_name)
    if meas_col in work.columns and pred_col in work.columns:
        work["baseline_error"] = (work[meas_col] - work[pred_col]).abs()
    if meas_col in work.columns and geo_pred_col in work.columns:
        work["experimental_error"] = (work[meas_col] - work[geo_pred_col]).abs()
    return work.dropna()


def _render_range_summary(dt_eval: pd.DataFrame) -> None:
    rows = []
    for metric_name in ("RSRP", "RSRQ", "SINR"):
        work = _prepare_kpi_eval(dt_eval, metric_name)
        meas_col = f"{metric_name}_meas"
        pred_col = f"{metric_name}_pred"
        geo_col = f"{metric_name}_pred_geo"
        if meas_col in work.columns and not work.empty:
            rows.append({
                "metric": metric_name,
                "series": "DT Measured",
                "min": round(float(work[meas_col].min()), 4),
                "max": round(float(work[meas_col].max()), 4),
                "mean": round(float(work[meas_col].mean()), 4),
            })
        if pred_col in work.columns and not work.empty:
            rows.append({
                "metric": metric_name,
                "series": "Baseline RF",
                "min": round(float(work[pred_col].min()), 4),
                "max": round(float(work[pred_col].max()), 4),
                "mean": round(float(work[pred_col].mean()), 4),
            })
        if geo_col in work.columns and not work.empty:
            rows.append({
                "metric": metric_name,
                "series": "Experimental Geo",
                "min": round(float(work[geo_col].min()), 4),
                "max": round(float(work[geo_col].max()), 4),
                "mean": round(float(work[geo_col].mean()), 4),
            })
    if rows:
        st.markdown("**KPI Range Summary**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_kpi_distribution(dt_eval: pd.DataFrame, metric_name: str) -> None:
    work = _prepare_kpi_eval(dt_eval, metric_name)
    if work.empty:
        st.info(f"No data available for {metric_name} distribution.")
        return
    meas_col = f"{metric_name}_meas"
    pred_col = f"{metric_name}_pred"
    geo_pred_col = f"{metric_name}_pred_geo"

    fig = go.Figure()
    for col, label, color in [
        (meas_col, "DT Measured", "#111827"),
        (pred_col, "Baseline RF", "#2563eb"),
        (geo_pred_col, "Experimental Geo", "#dc2626"),
    ]:
        if col in work.columns:
            fig.add_trace(
                go.Histogram(
                    x=work[col],
                    name=label,
                    opacity=0.55,
                    nbinsx=45,
                    marker_color=color,
                )
            )
    fig.update_layout(
        title=f"{metric_name} Distribution",
        barmode="overlay",
        xaxis_title=metric_name,
        yaxis_title="Count",
        legend_title="Series",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_scatter_validation(dt_eval: pd.DataFrame, metric_name: str) -> None:
    work = _prepare_kpi_eval(dt_eval, metric_name)
    if work.empty:
        st.info(f"No data available for {metric_name} validation scatter.")
        return
    meas_col = f"{metric_name}_meas"
    pred_col = f"{metric_name}_pred"
    geo_pred_col = f"{metric_name}_pred_geo"
    row = st.columns(2)
    for idx, (col, title) in enumerate([
        (pred_col, f"{metric_name}: DT vs Baseline RF"),
        (geo_pred_col, f"{metric_name}: DT vs Experimental Geo"),
    ]):
        if col not in work.columns:
            continue
        scatter = px.scatter(
            work,
            x=meas_col,
            y=col,
            color="morphology_cluster" if "morphology_cluster" in work.columns else None,
            opacity=0.5,
            title=title,
        )
        scatter.add_shape(
            type="line",
            x0=work[meas_col].min(),
            y0=work[meas_col].min(),
            x1=work[meas_col].max(),
            y1=work[meas_col].max(),
            line=dict(color="black", dash="dash"),
        )
        with row[idx]:
            st.plotly_chart(scatter, use_container_width=True)


def _render_error_distribution(dt_eval: pd.DataFrame, metric_name: str) -> None:
    work = _prepare_kpi_eval(dt_eval, metric_name)
    if work.empty or ("baseline_error" not in work.columns and "experimental_error" not in work.columns):
        st.info(f"No data available for {metric_name} error distribution.")
        return

    fig = go.Figure()
    for col, label, color in [
        ("baseline_error", "Baseline Abs Error", "#2563eb"),
        ("experimental_error", "Experimental Abs Error", "#dc2626"),
    ]:
        if col in work.columns:
            fig.add_trace(
                go.Histogram(
                    x=work[col],
                    name=label,
                    opacity=0.6,
                    nbinsx=45,
                    marker_color=color,
                )
            )
    fig.update_layout(
        title=f"{metric_name} Absolute Error Distribution",
        barmode="overlay",
        xaxis_title="Absolute Error",
        yaxis_title="Count",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_error_image(dt_eval: pd.DataFrame, metric_name: str) -> None:
    work = _prepare_kpi_eval(dt_eval, metric_name)
    if work.empty or ("baseline_error" not in work.columns and "experimental_error" not in work.columns):
        st.info(f"No data available for {metric_name} error image.")
        return

    row = st.columns(2)
    for idx, (col, title) in enumerate([
        ("baseline_error", f"{metric_name} Baseline Abs Error"),
        ("experimental_error", f"{metric_name} Experimental Abs Error"),
    ]):
        if col not in work.columns:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5), dpi=140)
        hb = ax.hexbin(
            work["lon"],
            work["lat"],
            C=work[col],
            gridsize=36,
            reduce_C_function=np.mean,
            cmap="turbo",
            mincnt=1,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Mean Abs Error")
        fig.tight_layout()
        with row[idx]:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _render_feature_map(feature_df: pd.DataFrame, feature_name: str) -> None:
    if feature_name not in feature_df.columns or feature_df.empty:
        st.info(f"No data available for feature {feature_name}.")
        return

    work = feature_df.dropna(subset=["lat", "lon", feature_name]).copy()
    if work.empty:
        st.info(f"No plottable data available for feature {feature_name}.")
        return

    if pd.api.types.is_numeric_dtype(work[feature_name]) or pd.to_numeric(work[feature_name], errors="coerce").notna().sum() > 0:
        work[feature_name] = pd.to_numeric(work[feature_name], errors="coerce")
        row = st.columns(2)
        fig_map = px.scatter(
            work,
            x="lon",
            y="lat",
            color=feature_name,
            title=f"{feature_name} Spatial View",
            opacity=0.7,
            color_continuous_scale="Turbo",
        )
        with row[0]:
            st.plotly_chart(fig_map, use_container_width=True)
        fig_hist = px.histogram(work, x=feature_name, nbins=40, title=f"{feature_name} Distribution")
        with row[1]:
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        row = st.columns(2)
        fig_map = px.scatter(
            work,
            x="lon",
            y="lat",
            color=feature_name,
            title=f"{feature_name} Spatial View",
            opacity=0.7,
        )
        with row[0]:
            st.plotly_chart(fig_map, use_container_width=True)
        counts = work[feature_name].astype(str).value_counts().reset_index()
        counts.columns = [feature_name, "count"]
        fig_bar = px.bar(counts, x=feature_name, y="count", title=f"{feature_name} Counts")
        with row[1]:
            st.plotly_chart(fig_bar, use_container_width=True)


def _render_signal_image(
    holdout_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metric_name: str,
) -> None:
    work = _prepare_kpi_eval(holdout_df, metric_name)
    if work.empty and pred_df.empty:
        st.info(f"No data available for {metric_name} signal image.")
        return

    meas_col = f"{metric_name}_meas"
    grid_pred_col = {
        "RSRP": "pred_rsrp",
        "RSRQ": "pred_rsrq",
        "SINR": "pred_sinr",
    }[metric_name]
    vmin, vmax = KPI_LIMITS[metric_name]
    panels = [
        ("holdout_dt", meas_col, f"{metric_name} Holdout DT Measured"),
        ("baseline_grid", grid_pred_col, f"{metric_name} Source RF Full Polygon"),
        ("experimental_grid", f"{grid_pred_col}_geo", f"{metric_name} Experimental Geo Full Polygon"),
    ]

    def _plot_panel(panel_kind: str, col: str, title: str) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), dpi=140)
        hb = None
        if panel_kind == "holdout_dt":
            if col in work.columns:
                hb = ax.hexbin(
                    work["lon"],
                    work["lat"],
                    C=work[col],
                    gridsize=36,
                    reduce_C_function=np.mean,
                    cmap="viridis",
                    mincnt=1,
                    vmin=vmin,
                    vmax=vmax,
                )
        elif panel_kind == "baseline_grid":
            if col in pred_df.columns:
                grid_plot = pred_df.dropna(subset=["lat", "lon", col]).copy()
                if not grid_plot.empty:
                    hb = ax.hexbin(
                        grid_plot["lon"],
                        grid_plot["lat"],
                        C=pd.to_numeric(grid_plot[col], errors="coerce"),
                        gridsize=48,
                        reduce_C_function=np.mean,
                        cmap="viridis",
                        mincnt=1,
                        vmin=vmin,
                        vmax=vmax,
                    )
        elif panel_kind == "experimental_grid":
            if col in pred_df.columns:
                grid_plot = pred_df.dropna(subset=["lat", "lon", col]).copy()
                if not grid_plot.empty:
                    hb = ax.hexbin(
                        grid_plot["lon"],
                        grid_plot["lat"],
                        C=pd.to_numeric(grid_plot[col], errors="coerce"),
                        gridsize=48,
                        reduce_C_function=np.mean,
                        cmap="viridis",
                        mincnt=1,
                        vmin=vmin,
                        vmax=vmax,
                    )
        if hb is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label(metric_name)
        ax.set_title(title)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    row1 = st.columns(3)
    with row1[0]:
        _plot_panel(*panels[0])
    with row1[1]:
        _plot_panel(*panels[1])
    with row1[2]:
        _plot_panel(*panels[2])


def main() -> None:
    st.set_page_config(page_title="LTE RF Debug Dashboard", layout="wide")
    st.title("LTE RF Debug Dashboard")
    st.caption("Test-only RF lab for project 196. Reads from DB, uses DT only for validation, keeps baseline RF non-calibrated, and compares it against a test-only geo-adjusted experiment.")

    st.sidebar.header("Run Controls")
    with st.sidebar.form("rf_debug_run_form"):
        project_id = st.number_input("Project ID", value=DEFAULT_PROJECT_ID, step=1)
        session_input = st.text_input("Session IDs", value=",".join(map(str, DEFAULT_SESSION_IDS)))
        region = st.text_input("Region", value=DEFAULT_REGION)
        radius_m = st.number_input("Radius (m)", value=DEFAULT_RADIUS_M, step=50.0)
        grid_resolution_m = st.number_input("Grid Resolution (m)", value=DEFAULT_GRID_RESOLUTION_M, step=5.0)
        workers = st.number_input("Workers", value=DEFAULT_WORKERS, step=1, min_value=1)
        max_interference_sites = st.number_input(
            "Max Interference Sites",
            value=DEFAULT_MAX_INTERFERENCE_SITES,
            step=5,
            min_value=1,
        )
        tile_size_m = st.number_input("Tile Size (m)", value=DEFAULT_TILE_SIZE_M, step=25.0)
        cluster_count = st.number_input("Morphology Clusters", value=DEFAULT_CLUSTER_COUNT, step=1, min_value=2)
        validation_fraction = st.slider(
            "Validation Fraction",
            min_value=0.1,
            max_value=0.5,
            value=float(DEFAULT_VALIDATION_FRACTION),
            step=0.05,
        )
        enable_osm = st.checkbox("Enable OSM Enrichment", value=False)
        run_triggered = st.form_submit_button("Run RF Debug Lab", type="primary")
    if run_triggered:
        session_ids = tuple(int(part.strip()) for part in session_input.split(",") if part.strip())
        config = RunConfig(
            project_id=int(project_id),
            session_ids=session_ids,
            region=region,
            radius_m=float(radius_m),
            grid_resolution_m=float(grid_resolution_m),
            workers=int(workers),
            max_interference_sites=int(max_interference_sites),
            tile_size_m=float(tile_size_m),
            cluster_count=int(cluster_count),
            validation_fraction=float(validation_fraction),
            enable_osm=enable_osm,
            output_root=OUTPUT_ROOT,
        )
        with st.spinner("Running RF debug lab. This uses DB input only and does not save results back to DB."):
            run_dir = run_rf_debug_lab(config)
        st.success(f"Run completed: {run_dir}")

    runs = _list_runs(int(project_id))
    if not runs:
        st.info("No RF debug runs found yet. Use the sidebar to launch one.")
        return

    run_labels = [run.name for run in runs]
    selected_label = st.selectbox("Available Runs", options=run_labels, index=0)
    run_dir = next(run for run in runs if run.name == selected_label)
    summary = _load_summary(run_dir)

    st.subheader("Run Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runtime (sec)", summary.get("total_runtime_sec"))
    c2.metric("RF Grid Rows", summary.get("rows", {}).get("rf_prediction_grid"))
    c3.metric("Accuracy Points", summary.get("rows", {}).get("rf_accuracy_points"))
    c4.metric("Building Polygons", summary.get("rows", {}).get("building_polygons"))

    st.markdown("**Timing Breakdown**")
    timings_df = pd.DataFrame(
        [{"step": key, "seconds": value} for key, value in summary.get("timings_sec", {}).items()]
    )
    if not timings_df.empty:
        st.dataframe(timings_df, use_container_width=True)

    if summary.get("building_alignment"):
        st.markdown(f"**Building Alignment**: `{summary['building_alignment']}`")
    if summary.get("production_style_prediction"):
        st.markdown("**Prediction Mode**: `production_style_rf_polygon`")
    if summary.get("holdout_strategy"):
        if summary.get("holdout_strategy") == "validation_only_sessions":
            st.markdown(
                f"**Validation Mode**: `dt_validation_only` | "
                f"validation_sessions=`{summary.get('holdout_sessions', [])}`"
            )
        else:
            st.markdown(
                f"**Validation Split**: `{summary['holdout_strategy']}` | "
                f"train_sessions=`{summary.get('train_sessions', [])}` | "
                f"holdout_sessions=`{summary.get('holdout_sessions', [])}`"
            )

    feature_diag = summary.get("feature_diagnostics", {})
    if feature_diag:
        st.markdown("**Feature Diagnostics**")
        feature_diag_df = pd.DataFrame(
            [{"feature": key, **value} for key, value in feature_diag.items()]
        )
        st.dataframe(feature_diag_df, use_container_width=True)

    cluster_counts = summary.get("cluster_counts", {})
    if cluster_counts:
        st.markdown("**Cluster Counts**")
        cluster_df = pd.DataFrame(
            [{"morphology_cluster": key, "count": value} for key, value in cluster_counts.items()]
        )
        st.dataframe(cluster_df, use_container_width=True)

    experimental_model = summary.get("experimental_model", {})
    if experimental_model:
        st.markdown("**Experimental Model**")
        experimental_df = pd.DataFrame(
            [
                {
                    "metric": metric,
                    "train_rows": info.get("train_rows"),
                    "feature_count": info.get("feature_count"),
                    "top_features": ", ".join(
                        f"{name}={value}" for name, value in info.get("top_features", {}).items()
                    ),
                }
                for metric, info in experimental_model.items()
            ]
        )
        st.dataframe(experimental_df, use_container_width=True)

    artifacts = summary["artifacts"]
    polygon_gdf = gpd.read_file(artifacts["project_polygon"])
    grid_gdf = gpd.read_file(artifacts["analysis_grid"])
    buildings_path = Path(artifacts.get("buildings", ""))
    buildings_gdf = gpd.read_file(buildings_path) if buildings_path.exists() else None
    analysis_features_df = pd.read_csv(artifacts["analysis_grid_features"])
    site_df = pd.read_csv(artifacts["site_df"])
    site_df = _prepare_site_selection_df(site_df)
    drive_df = pd.read_csv(artifacts["drive_df"])
    pred_df = pd.read_parquet(artifacts["rf_prediction_grid"])
    pred_map_df = pd.read_csv(artifacts["rf_prediction_grid_sample"])
    dt_eval = pd.read_csv(artifacts["rf_accuracy_points"])
    st.subheader("Metric Comparison")
    tabs = st.tabs(["RSRP", "RSRQ", "SINR"])
    metric_names = ["RSRP_meas", "RSRQ_meas", "SINR_meas"]
    for tab, metric_name in zip(tabs, metric_names):
        with tab:
            _render_metric_compare(summary, metric_name)
            _render_metric_detail_table(summary, metric_name)

    _render_range_summary(dt_eval)

    st.subheader("RF Comparison Images")
    image_tabs = st.tabs(["RSRP Images", "RSRQ Images", "SINR Images"])
    for tab, metric_name in zip(image_tabs, ("RSRP", "RSRQ", "SINR")):
        with tab:
            st.markdown(f"**{metric_name}: Holdout DT vs Source RF Full Polygon**")
            _render_signal_image(dt_eval, pred_df, metric_name)
            _render_error_image(dt_eval, metric_name)

    st.subheader("Validation Charts")
    chart_tabs = st.tabs(["RSRP Charts", "RSRQ Charts", "SINR Charts"])
    for tab, metric_name in zip(chart_tabs, ("RSRP", "RSRQ", "SINR")):
        with tab:
            _render_scatter_validation(dt_eval, metric_name)
            _render_error_distribution(dt_eval, metric_name)

    st.subheader("Maps")
    map_control_cols = st.columns(4)
    kpi_map_choice = map_control_cols[0].selectbox(
        "RF KPI",
        options=[
            ("RSRP", "pred_rsrp"),
            ("RSRQ", "pred_rsrq"),
            ("SINR", "pred_sinr"),
            ("RSRP Geo", "pred_rsrp_geo"),
            ("RSRQ Geo", "pred_rsrq_geo"),
            ("SINR Geo", "pred_sinr_geo"),
        ],
        format_func=lambda item: item[0],
        index=0,
    )[1]
    selection_mode = map_control_cols[1].radio("Coverage Scope", options=["All", "Sector", "NodeB"], horizontal=True)
    available_sectors = sorted(site_df["Node_Cell_ID"].dropna().astype(str).unique().tolist()) if "Node_Cell_ID" in site_df.columns else []
    available_nodebs = (
        sorted(
            [
                value
                for value in site_df["dashboard_nodeb_id"].dropna().astype(str).unique().tolist()
                if value not in {"", "nan", "None"}
            ]
        )
        if "dashboard_nodeb_id" in site_df.columns
        else []
    )
    selected_sector = None
    selected_nodeb = None
    if selection_mode == "Sector" and available_sectors:
        selected_sector = map_control_cols[2].selectbox("Sector", options=available_sectors, index=0)
    elif selection_mode == "NodeB" and available_nodebs:
        selected_nodeb = map_control_cols[2].selectbox("NodeB/Site", options=available_nodebs, index=0)
    show_site_markers = map_control_cols[3].checkbox("Show Site Markers", value=True)

    map_tabs = st.tabs(["RF Full Polygon", "Clutter Tiles", "Morphology Clusters"])
    with map_tabs[0]:
        _render_map(
            _build_map(
                polygon_gdf,
                site_df,
                drive_df,
                pred_df,
                buildings_gdf,
                grid_gdf,
                show_geo=False,
                kpi_col=kpi_map_choice,
                selected_sector=selected_sector,
                selected_nodeb=selected_nodeb,
                show_site_markers=show_site_markers,
            ),
            "baseline_map",
        )
    with map_tabs[1]:
        _render_map(
            _build_map(
                polygon_gdf,
                site_df.iloc[:50],
                drive_df.iloc[:1],
                pred_df.iloc[:1],
                buildings_gdf,
                grid_gdf,
                show_geo=False,
                show_site_markers=show_site_markers,
            ),
            "clutter_map",
        )
    with map_tabs[2]:
        _render_map(
            _build_map(
                polygon_gdf,
                site_df.iloc[:50],
                drive_df.iloc[:1],
                pred_df.iloc[:1],
                buildings_gdf,
                grid_gdf,
                show_geo=True,
                show_site_markers=show_site_markers,
            ),
            "cluster_map",
        )

    if {"RSRP_meas", "RSRP_pred"}.issubset(dt_eval.columns):
        scatter = px.scatter(
            dt_eval,
            x="RSRP_meas",
            y="RSRP_pred",
            color="morphology_cluster" if "morphology_cluster" in dt_eval.columns else None,
            title="Baseline RF: Measured vs Predicted RSRP",
            opacity=0.65,
        )
        st.plotly_chart(scatter, use_container_width=True)

    feature_candidates = [
        "building_count",
        "building_area_ratio",
        "avg_building_area_m2",
        "road_length_m",
        "green_ratio",
        "water_ratio",
        "nearest_site_distance_m",
        "mean_nearest3_site_distance_m",
        "site_count_250m",
        "site_count_500m",
        "serving_distance_m",
        "azimuth_delta_deg",
        "clutter_class",
        "morphology_cluster",
    ]
    available_features = [col for col in feature_candidates if col in analysis_features_df.columns]
    if available_features:
        st.subheader("Feature Visualization")
        selected_feature = st.selectbox("Feature Parameter", available_features, index=0)
        _render_feature_map(analysis_features_df, selected_feature)

    st.subheader("Run Logs")
    run_log_path = Path(artifacts["run_log"])
    if run_log_path.exists():
        st.text_area("Test Lab Log", run_log_path.read_text(encoding="utf-8", errors="ignore"), height=320)
    rf_log_path = summary.get("rf_log_path")
    if rf_log_path and Path(rf_log_path).exists():
        st.text_area("Source RF Log", Path(rf_log_path).read_text(encoding="utf-8", errors="ignore"), height=320)


if __name__ == "__main__":
    main()
