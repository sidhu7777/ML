import pandas as pd
from shapely.geometry import Point
from shapely import wkb, wkt
from shapely.wkt import loads as load_wkt

from .db import (
    get_project_by_id,
    get_network_logs_for_sessions,
    get_project_regions,
)



def debug_load_project_data(project_id: int):
    print("\n" + "=" * 80)
    print(f"DEBUG LOAD PROJECT DATA | project_id = {project_id}")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Project
    # --------------------------------------------------
    project = get_project_by_id(project_id)
    if not project:
        print(" Project not found")
        return

    print("\n PROJECT DETAILS")
    print(f"  ID           : {project['id']}")
    print(f"  Name         : {project['project_name']}")
    print(f"  Ref Sessions : {project['ref_session_id']}")

    # --------------------------------------------------
    # 2. Sessions
    # --------------------------------------------------
    session_ids = [
        int(s.strip())
        for s in project["ref_session_id"].split(",")
        if s.strip().isdigit()
    ]

    print("\n PARSED SESSION IDS")
    print(f"  Sessions ({len(session_ids)}): {session_ids}")

    # --------------------------------------------------
    # 3. Raw network logs
    # --------------------------------------------------
    raw_df = get_network_logs_for_sessions(session_ids)

    print("\n RAW NETWORK LOG DATA")
    print(f"  Total raw rows : {len(raw_df)}")

    print("\n  Rows per session (RAW):")
    for sid in session_ids:
        count = len(raw_df[raw_df["session_id"] == sid])
        print(f"    Session {sid}: {count} rows")

    print("\n RAW DATAFRAME COLUMNS")
    print(list(raw_df.columns))

    # --------------------------------------------------
    # 4. Regions / polygons
    # --------------------------------------------------
        # --------------------------------------------------
    # 4. Regions / polygons (FULL DEBUG)
    # --------------------------------------------------
    region_rows = get_project_regions(project_id)

    print("\n PROJECT POLYGONS")
    print(f"  Total polygons : {len(region_rows)}")

    polygons = []

    for r in region_rows:
        poly_id = r.get("id")
        region_wkt = r.get("region_wkt")

        if not region_wkt:
            print(f"⚠ Polygon ID {poly_id} has empty geometry")
            continue

        try:
            polygon = load_wkt(region_wkt)
            polygons.append(polygon)

            print("\n" + "-" * 60)
            print(f"Polygon ID      : {poly_id}")
            print(f"Geometry type   : {polygon.geom_type}")

            # Print bounds (VERY IMPORTANT sanity check)
            minx, miny, maxx, maxy = polygon.bounds
            print("Bounds (lon/lat):")
            print(f"  min_lon={minx}, min_lat={miny}")
            print(f"  max_lon={maxx}, max_lat={maxy}")

            # Print number of points
            coords = list(polygon.exterior.coords)
            print(f"Total points    : {len(coords)}")

            # Print coordinates explicitly
            print("Polygon coordinates (lon, lat):")
            for i, (lon, lat) in enumerate(coords, start=1):
                print(f"  {i:02d}. lon={lon}, lat={lat}")

        except Exception as e:
            print(f"⚠ Polygon ID {poly_id} could not be parsed: {e}")

    # --------------------------------------------------
    # 5. Apply polygon filter
    # --------------------------------------------------
    print("\n APPLYING POLYGON FILTER")

    # If no valid polygons, skip filtering safely
    if not polygons:
        print("⚠ No valid polygons found — skipping spatial filtering")
        filtered_df = raw_df.copy()

    else:
        mask = []
        for _, row in raw_df.iterrows():
            # CORRECT COLUMN NAMES (DB SCHEMA)
            lon = row["lon"]
            lat = row["lat"]

            point = Point(lon, lat)
            inside = any(poly.contains(point) for poly in polygons)
            mask.append(inside)

        filtered_df = raw_df.loc[mask].reset_index(drop=True)

    # --------------------------------------------------
    # 6. Filtered output
    # --------------------------------------------------
    print("\n FILTERED DATA")
    print(f"  Total filtered rows : {len(filtered_df)}")

    print("\n  Rows per session (FILTERED):")
    for sid in session_ids:
        count = len(filtered_df[filtered_df["session_id"] == sid])
        print(f"    Session {sid}: {count} rows")

    # --------------------------------------------------
    # 7. Final summary
    # --------------------------------------------------
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Project ID            : {project_id}")
    print(f"Total sessions        : {len(session_ids)}")
    print(f"Total raw rows        : {len(raw_df)}")
    print(f"Total filtered rows   : {len(filtered_df)}")
    print("-" * 80)

    return raw_df, filtered_df, project
