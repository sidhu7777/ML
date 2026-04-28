import os
import json
import logging
import traceback
import osmnx as ox
import geopandas as gpd
from shapely import ops 
import sqlalchemy as db
from sqlalchemy.exc import OperationalError
from flask import current_app
from dotenv import load_dotenv

# -------------------------------------
# GLOBAL ENGINE (MULTI-REGION)
# -------------------------------------
load_dotenv()

engine_dict = {
    "india": db.create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=3600
    ) if os.getenv("DATABASE_URL") else None,
    
    "taiwan": db.create_engine(
        os.getenv("DATABASE_URL_Taiwan"),
        pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=3600
    ) if os.getenv("DATABASE_URL_Taiwan") else None
}

def get_regional_engine(region_name):
    """Safely retrieves the requested region engine, falling back to India."""
    region = str(region_name).lower()
    current_engine = engine_dict.get(region, engine_dict.get("india"))
    
    if current_engine is None:
        raise RuntimeError(f"Database engine for region '{region}' is not initialized/configured.")
        
    return current_engine


# -------------------------------------
# BUILDING EXTRACTION + SAVE CLASS
# -------------------------------------
class BuildingService:

    def fetch_buildings(self, polygon):
        """Fetch buildings from OSM"""
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        ox.settings.timeout = 180
        ox.settings.use_cache = True

        try:
            buildings = ox.features_from_polygon(
                polygon,
                tags={"building": True, "residential": True}
            )

            buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]

            if buildings.empty:
                return None, 0

            return buildings, len(buildings)

        except Exception as e:
            if "No matching features" in str(e):
                return None, 0
            raise e

    # -------------------------------------
    # SAVE TO DATABASE
    # -------------------------------------
    # ✅ Added 'region' parameter
    def save_buildings_to_db(self, buildings, area_name, project_id, swap_output=False, region="india"):
        
        # ✅ Fetch regional engine dynamically
        current_engine = get_regional_engine(region)

        # EXPLODE MULTIPOLYGONS
        buildings_exp = buildings.explode(index_parts=True, ignore_index=True)
        
        # --- FIX: Calculate Area FIRST ---
        buildings_exp["calc_area"] = buildings_exp.to_crs(epsg=3857).geometry.area
        
        # Safety net: Convert any mathematically impossible areas to 0
        buildings_exp["calc_area"] = buildings_exp["calc_area"].fillna(0)

        # --- SWAP BACK LOGIC (Lon/Lat -> Lat/Lon) ---
        if swap_output:
            print(f"🔄 Buildings: Swapping output back to Lat/Lon before saving...")
            buildings_exp['geometry'] = buildings_exp['geometry'].apply(
                lambda geom: ops.transform(lambda x, y: (y, x), geom)
            )
        # --------------------------------------------

        # Convert geometry to WKT
        buildings_exp["wkt_4326"] = buildings_exp.geometry.to_wkt()

        values_list = [
            (area_name, row.wkt_4326, project_id, row.calc_area)
            for row in buildings_exp.itertuples()
        ]

        # ✅ Grab the raw connection specifically from the regional engine
        raw = current_engine.raw_connection()
        cur = raw.cursor()

        try:
            cur.execute("SET autocommit=0")
            cur.execute("SET unique_checks=0")
            cur.execute("SET foreign_key_checks=0")

            batch_size = 1000
            total = 0

            for i in range(0, len(values_list), batch_size):
                batch = values_list[i : i + batch_size]

                placeholders = "(%s, ST_GeomFromText(%s, 4326), %s, %s)"
                values_str = ", ".join([placeholders] * len(batch))

                insert_sql = f"""
                    INSERT INTO tbl_savepolygon (name, region, project_id, area)
                    VALUES {values_str}
                """

                flat_values = [item for sub in batch for item in sub]
                cur.execute(insert_sql, flat_values)
                total += len(batch)

            cur.execute("SET unique_checks=1")
            cur.execute("SET foreign_key_checks=1")
            raw.commit()

        finally:
            cur.close()
            raw.close()

        return total

    # -------------------------------------
    # MAIN EXTRACT + SAVE METHOD (UPDATED)
    # -------------------------------------
    # ✅ Added 'region' parameter
    def process_buildings(self, polygon, name, project_id, swap_output=False, region="india"):
        """
        Extract buildings and save them to DB
        """
        buildings, count = self.fetch_buildings(polygon)

        if buildings is None or buildings.empty:
            return None, 0, 0

        # ✅ Pass region down to the save function
        saved_count = self.save_buildings_to_db(buildings, name, project_id, swap_output=swap_output, region=region)

        geojson = json.loads(buildings.to_json())

        return geojson, count, saved_count