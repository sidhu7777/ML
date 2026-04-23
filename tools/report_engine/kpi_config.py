# src/kpi_config.py

# --------------------------------------------------
# Color functions (USED ONLY FOR POINT COLORING)
# NOTE: Legend colors come from DB, not from here
# --------------------------------------------------

def rsrq_color_manual(value):
    if value is None:
        return "#d3d3d3"
    if value < -18:
        return "#4400ff"
    elif value < -16:
        return "#00ffee"
    elif value < -14:
        return "#90EE90"
    else:
        return "#006400"


def rsrp_colour_manual(value):
    if value is None:
        return "#d3d3d3"
    if value < -124:
        return "#161816"
    elif value < -105:
        return "#d2b6b8"
    elif value < -95:
        return "#e0ee77"
    elif value < -80:
        return "#4a3fe4"
    elif value < -75:
        return "#56c86c"
    elif value < -70:
        return "#3a9f04"
    elif value < -65:
        return "#1fb21f"
    elif value < -60:
        return "#e67d6b"
    else:
        return "#6f7b19"


def sinr_color_manual(value):
    if value is None:
        return "#d3d3d3"
    if value < 1:
        return "#FF0000"
    elif value < 5:
        return "#b07454"
    elif value < 10:
        return "#0000FF"
    elif value < 15:
        return "#90EE90"
    else:
        return "#d548a2"


def dl_colour_manual(value):
    if value is None:
        return "#808080"
    if value < 5:
        return "#ff0000"
    elif value < 20:
        return "#ffff00"
    else:
        return "#00ff00"


def ul_colour_manual(value):
    if value is None:
        return "#808080"
    if value < 1:
        return "#ff0000"
    elif value < 5:
        return "#ffff00"
    else:
        return "#00ff00"


def mos_colour_manual(value):
    if value is None:
        return "#808080"
    if value < 1:
        return "#FF0000"
    elif value < 3:
        return "#FFFF00"
    else:
        return "#0000FF"


# --------------------------------------------------
# KPI CONFIG (STRUCTURE ONLY — NO RANGES)
# --------------------------------------------------

KPI_CONFIG = {
    "RSRP": {
        "column": "rsrp",
        "map_name": "rsrp_map.png",
        "color_func": rsrp_colour_manual,
        "type": "range",
    },
    "RSRQ": {
        "column": "rsrq",
        "map_name": "rsrq_map.png",
        "color_func": rsrq_color_manual,
        "type": "range",
    },
    "SINR": {
        "column": "sinr",
        "map_name": "sinr_map.png",
        "color_func": sinr_color_manual,
        "type": "range",
    },
    "DL": {
        "column": "dl_tpt",
        "map_name": "dl_map.png",
        "color_func": dl_colour_manual,
        "type": "range",
    },
    "UL": {
        "column": "ul_tpt",
        "map_name": "ul_map.png",
        "color_func": ul_colour_manual,
        "type": "range",
    },
    "MOS": {
        "column": "mos",
        "map_name": "mos_map.png",
        "color_func": mos_colour_manual,
        "type": "range",
    },
    "Band": {
        "column": "band",
        "map_name": "band_map.png",
        "type": "categorical",
    },
    "PCI": {
        "column": "pci",
        "map_name": "pci_map.png",
        "type": "categorical",
    },
}

# --------------------------------------------------
# Fixed Threshold Analysis Configuration
# Thresholds for uniform reporting (independent of DB ranges)
# Based on industry standards and reference documentation
# --------------------------------------------------
FIXED_THRESHOLD_CONFIG = {
    "DL": {
        "poor_threshold": 10,       # DL throughput <= 10 Mbps considered poor
        "excellent_threshold": 15,  # DL throughput > 15 Mbps considered excellent
    },
    "UL": {
        "poor_threshold": 5,        # UL throughput <= 5 Mbps considered poor
        "range_min": 6,             # Lower bound of mid-range (6-8 Mbps)
        "range_max": 8,             # Upper bound of mid-range (6-8 Mbps)
    },
    "MOS": {
        "poor_threshold": 3.0,      # MOS < 3.0 considered poor voice quality
    },
    "SINR": {
        "poor_threshold": 5,        # SINR < 5 considered poor
        "range_min": 5,             # Lower bound of acceptable range (5-15)
        "range_max": 15,            # Upper bound of acceptable range (5-15)
    },
}
