#area_breakup/services.py
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt, ops # Added ops
from shapely.geometry import shape, Polygon
import osmnx as ox
import folium
from flask import current_app
from sqlalchemy import create_engine, text
from sklearn.cluster import DBSCAN, KMeans
from pyproj import CRS
from geovoronoi import voronoi_regions_from_coords

# ================== CONFIG ==================
BUILDING_CLUSTER_EPS_METERS = 100
DEFAULT_MIN_SAMPLES = 10 

def get_db_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url: return None
    return create_engine(db_url, pool_recycle=3600)

def ensure_outdir(path):
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)

# ================== DB SAVING FUNCTION (UPDATED) ==================
def save_to_database(gdf, table_name, project_id, name, swap_output=False):
    engine = get_db_engine()
    if gdf.empty or engine is None:
        return

    try:
        # 1. Create a clean copy
        df_db = pd.DataFrame(gdf.copy()) 

        # 2. SWAP BACK if requested (Lon/Lat -> Lat/Lon)
        if swap_output:
            print(f"🔄 Swapping output back to Lat/Lon for {table_name}...")
            # Flip X and Y
            df_db['geometry'] = df_db['geometry'].apply(
                lambda geom: ops.transform(lambda x, y: (y, x), geom)
            )

        # 3. Convert Geometry to WKT (Text)
        df_db['geometry'] = df_db['geometry'].apply(lambda x: x.wkt)

        # 4. Add Metadata
        df_db['project_id'] = project_id
        df_db['project_name'] = name
        
        # 5. SELECT ONLY COLUMNS THAT EXIST IN DATABASE
        allowed_cols = ['project_id', 'project_name', 'geometry']
        if 'block_id' in df_db.columns: allowed_cols.append('block_id')
        if 'zone_id' in df_db.columns: allowed_cols.append('zone_id')
        if 'cluster_id' in df_db.columns: allowed_cols.append('cluster_id')

        df_final = df_db[allowed_cols]

        # 6. Write to SQL
        df_final.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"💾 Saved {len(df_final)} rows to table: {table_name}")
        
    except Exception as e:
        print(f"❌ Database Error for {table_name}: {e}")

# ================== DB FETCH FUNCTION ==================
def get_project_data(project_id):
    engine = get_db_engine()
    if engine is None: return None
    results = {}
    try:
        with engine.connect() as conn:
            query_grid = text("SELECT * FROM output_grid_blocks WHERE project_id = :pid")
            df_grid = pd.read_sql(query_grid, conn, params={"pid": project_id})
            results["grid_blocks"] = df_grid.replace({np.nan: None}).to_dict(orient="records")

            query_zones = text("SELECT * FROM output_ai_zones WHERE project_id = :pid")
            df_zones = pd.read_sql(query_zones, conn, params={"pid": project_id})
            results["ai_zones"] = df_zones.replace({np.nan: None}).to_dict(orient="records")

            query_clusters = text("SELECT * FROM output_building_clusters WHERE project_id = :pid")
            df_clusters = pd.read_sql(query_clusters, conn, params={"pid": project_id})
            results["building_clusters"] = df_clusters.replace({np.nan: None}).to_dict(orient="records")
        return results
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

# ================== CORE GIS LOGIC ==================
def choose_utm_crs(gdf_4326):
    centroid = gdf_4326.to_crs("EPSG:4326").geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": south}).to_string()

def clip_and_clean(gdf, mask_gdf):
    gdf = gdf.to_crs(mask_gdf.crs)
    gdf["geometry"] = gdf.geometry.buffer(0)
    clipped = gpd.overlay(gdf, mask_gdf, how="intersection")
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notnull()]
    clipped.reset_index(drop=True, inplace=True)
    return clipped

def osm_features_from_polygon_safe(geom, tags):
    try:
        print(f"🌍 Fetching OSM data for tags: {tags}...")
        result = ox.features.features_from_polygon(geom, tags)
        print(f"✅ Found {len(result)} features from OSM.")
        return result
    except Exception as e:
        print(f"❌ OSM API FAILED: {e}") # <--- This will tell you the exact problem
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

def fetch_buildings(mask_polygon):
    geom = mask_polygon.geometry.union_all()
    buildings = osm_features_from_polygon_safe(geom, {"building": True})
    if not buildings.empty:
        buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if buildings.crs is None:
            buildings.set_crs("EPSG:4326", inplace=True)
        else:
            buildings = buildings.to_crs("EPSG:4326")
    return buildings

def cluster_buildings_to_polygons(buildings_gdf, mask_polygon, min_samples=DEFAULT_MIN_SAMPLES):
    if buildings_gdf.empty: return gpd.GeoDataFrame(columns=["cluster_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    utm_crs = choose_utm_crs(mask_polygon)
    buildings_utm = buildings_gdf.to_crs(utm_crs)
    coords = np.array([[p.x, p.y] for p in buildings_utm.geometry.centroid])
    db = DBSCAN(eps=BUILDING_CLUSTER_EPS_METERS, min_samples=min_samples).fit(coords)
    labels = db.labels_
    cluster_geoms, cluster_ids = [], []
    for label in set(labels):
        if label == -1: continue
        indices = np.where(labels == label)[0]
        merged = buildings_utm.iloc[indices].geometry.union_all().convex_hull
        cluster_geoms.append(merged)
        cluster_ids.append(label)
    if not cluster_geoms: return gpd.GeoDataFrame(columns=["cluster_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    gdf_clusters = gpd.GeoDataFrame({"cluster_id": cluster_ids, "geometry": cluster_geoms}, crs=utm_crs)
    return gdf_clusters.to_crs("EPSG:4326")

def create_block_grid(mask_polygon, cell_size_m):
    if mask_polygon.empty: return gpd.GeoDataFrame()
    utm_crs = choose_utm_crs(mask_polygon)
    mask_utm = mask_polygon.to_crs(utm_crs)
    xmin, ymin, xmax, ymax = mask_utm.total_bounds
    polygons, block_ids = [], []
    block_counter = 1
    y = ymin
    while y < ymax:
        x = xmin
        while x < xmax:
            polygons.append(Polygon([(x, y), (x + cell_size_m, y), (x + cell_size_m, y + cell_size_m), (x, y + cell_size_m)]))
            block_ids.append(block_counter)
            block_counter += 1
            x += cell_size_m
        y += cell_size_m
    grid = gpd.GeoDataFrame({"block_id": block_ids, "geometry": polygons}, crs=utm_crs)
    grid = clip_and_clean(grid, mask_utm)
    grid = grid.to_crs("EPSG:4326")
    grid = grid.reset_index(drop=True)
    grid["block_id"] = grid.index + 1
    return grid

def create_ai_zones(mask_polygon, buildings_gdf, target=200, min_zones=10, max_zones=80):
    if buildings_gdf.empty: return gpd.GeoDataFrame()
    utm_crs = choose_utm_crs(mask_polygon)
    buildings_utm = buildings_gdf.to_crs(utm_crs)
    mask_utm = mask_polygon.to_crs(utm_crs)
    coords = np.array([[p.x, p.y] for p in buildings_utm.geometry.centroid])
    
    # Lowered the limit to 3 buildings
    if len(coords) < 3: return gpd.GeoDataFrame() 
    
    # Dynamically adjust the minimum zones so KMeans doesn't crash 
    # (You can't create 10 zones if you only have 5 buildings)
    actual_min_zones = min(min_zones, len(coords))
    k_est = max(actual_min_zones, min(max_zones, len(coords) // target))
    
    kmeans = KMeans(n_clusters=k_est, random_state=42, n_init="auto").fit(coords)
    region_polys, _ = voronoi_regions_from_coords(kmeans.cluster_centers_, mask_utm.geometry.iloc[0])
    records = [{"zone_id": int(cid)+1, "geometry": poly} for cid, poly in region_polys.items() if not poly.is_empty]
    g_zones = gpd.GeoDataFrame(records, crs=utm_crs)
    g_zones = clip_and_clean(g_zones, mask_utm)
    g_zones['geometry'] = g_zones.geometry.buffer(0)
    g_zones = g_zones.to_crs("EPSG:4326")
    g_zones = g_zones.reset_index(drop=True)
    g_zones["zone_id"] = g_zones.index + 1
    return g_zones

def export_files(gdf, basename):
    out_dir = current_app.config.get('OUTPUT_FOLDER', './outputs')
    ensure_outdir(out_dir)
    geojson = os.path.join(out_dir, f"{basename}.geojson")
    html = os.path.join(out_dir, f"{basename}.html")
    gdf.to_file(geojson, driver="GeoJSON")
    minx, miny, maxx, maxy = gdf.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]
    m = folium.Map(location=center, zoom_start=14)
    folium.GeoJson(gdf).add_to(m)
    m.save(html)
    return [geojson, html]
