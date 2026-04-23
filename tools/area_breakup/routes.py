# area_breakup/routes.py
from flask import Blueprint, request, jsonify, current_app
from shapely import wkt, ops
import geopandas as gpd
import re
from .services import (
    create_block_grid, 
    fetch_buildings, 
    create_ai_zones, 
    cluster_buildings_to_polygons, 
    save_to_database, 
    export_files,
    get_project_data 
)

area_breakup_bp = Blueprint('area_breakup', __name__)

# ================== HELPER: SMART WKT PARSER ==================
def clean_and_load_wkt(raw_string):
    """
    Returns: (geometry, was_swapped)
    was_swapped = True if we detected Lat/Lon input and flipped it to Lon/Lat.
    """
    if not raw_string:
        raise ValueError("WKT string is empty")

    clean_str = raw_string.strip()
    if not clean_str.upper().startswith("POLYGON"):
        if clean_str.startswith("(("):
            clean_str = f"POLYGON {clean_str}"
        else:
            raise ValueError("Input must be a POLYGON or ((...))")

    poly_geom = wkt.loads(clean_str)
    min_x, min_y, max_x, max_y = poly_geom.bounds
    
    needs_swap = False
    
    # 1. Strict Check: Latitude > 90 is impossible, so it must be Longitude.
    if min_y < -90 or max_y > 90:
        needs_swap = True
    
    # 2. Heuristic for India/Asia users: 
    # If Y (Lat) is > 60 (Arctic) and X (Lon) is small (< 60), 
    # user likely meant Lat~28 (India) and Lon~77. 
    # (Because Lat 77 is freezing ocean, Lat 28 is Delhi).
    elif (min_y > 60 and min_x < 60): 
        needs_swap = True

    if needs_swap:
        print("🔄 Detected Lat/Lon input. Swapping to Lon/Lat for processing...")
        poly_geom = ops.transform(lambda x, y: (y, x), poly_geom)
        return poly_geom, True # True = We swapped it

    return poly_geom, False # False = Input was already correct

# ================== PROCESS ENDPOINT (POST) ==================
@area_breakup_bp.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        name = data.get("Name", "output")
        wkt_input = data.get("WKT")
        project_id = data.get("project_id")
        block_size = float(data.get("grid", 100))
        min_samples = int(data.get("min_samples", 10))

        if not wkt_input:
            return jsonify({"status": "error", "message": "WKT is required"}), 400

        # --- CAPTURE THE SWAP FLAG ---
        try:
            poly_geom, was_input_swapped = clean_and_load_wkt(wkt_input)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid WKT format: {str(e)}"}), 400
        
        # If input was Lat/Lon, we want output to be Lat/Lon too.
        # So we pass 'swap_output=True' to save_to_database.
        should_swap_output = was_input_swapped 

        mask_polygon = gpd.GeoDataFrame(
            {"Name": [name], "project_id": [project_id]}, 
            geometry=[poly_geom], 
            crs="EPSG:4326"
        )

        results_summary = []

        # 1. PROCESS GRID
        g_blocks = create_block_grid(mask_polygon, block_size)
        if not g_blocks.empty:
            save_to_database(g_blocks, "output_grid_blocks", project_id, name, swap_output=should_swap_output)
            files = export_files(g_blocks, f"{name}_blocks")
            results_summary.append(f"Grid: {len(g_blocks)} blocks saved.")

        # 2. FETCH BUILDINGS
        g_buildings = fetch_buildings(mask_polygon)
        if not g_buildings.empty:
            
            # 3. PROCESS AI ZONES
            g_ai_zones = create_ai_zones(mask_polygon, g_buildings)
            if not g_ai_zones.empty:
                save_to_database(g_ai_zones, "output_ai_zones", project_id, name, swap_output=should_swap_output)
                files = export_files(g_ai_zones, f"{name}_ai_zones")
                results_summary.append(f"AI Zones: {len(g_ai_zones)} zones saved.")

            # 4. PROCESS CLUSTERS
            g_clusters = cluster_buildings_to_polygons(g_buildings, mask_polygon, min_samples=min_samples)
            if not g_clusters.empty:
                save_to_database(g_clusters, "output_building_clusters", project_id, name, swap_output=should_swap_output)
                files = export_files(g_clusters, f"{name}_clusters")
                results_summary.append(f"Clusters: {len(g_clusters)} clusters saved.")

        return jsonify({
            "status": "success",
            "project_id": project_id,
            "message": "Processing complete.",
            "details": results_summary,
            "input_format_detected": "Lat/Lon" if was_input_swapped else "Lon/Lat"
        })

    except Exception as e:
        current_app.logger.error(f"Area Breakup Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ================== FETCH ENDPOINT (GET) ==================
@area_breakup_bp.route('/fetch/<project_id>', methods=['GET'])
def fetch_data(project_id):
    try:
        data = get_project_data(project_id)
        if data is None:
             return jsonify({"status": "error", "message": "Database error"}), 500

        if not data.get("grid_blocks") and not data.get("ai_zones"):
            return jsonify({
                "status": "success",
                "message": f"No data found for project_id: {project_id}",
                "data": {"grid_blocks": [], "ai_zones": [], "building_clusters": []}
            }), 200

        return jsonify({"status": "success", "project_id": project_id, "data": data}), 200

    except Exception as e:
        current_app.logger.error(f"Fetch Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
