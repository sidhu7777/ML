import pandas as pd

from tests.lte_rf_debug_lab import _derive_clutter_class, _normalize_site_for_rf


def test_derive_clutter_class_assigns_multiple_labels():
    df = pd.DataFrame(
        {
            "building_count": [0, 3, 8, 15],
            "building_area_ratio": [0.0, 0.05, 0.18, 0.35],
            "road_length_m": [0.0, 40.0, 120.0, 260.0],
            "green_ratio": [0.0, 0.1, 0.0, 0.0],
            "water_ratio": [0.0, 0.0, 0.0, 0.0],
        }
    )
    clutter = _derive_clutter_class(df)
    assert len(clutter) == len(df)
    assert clutter.nunique() >= 2


def test_normalize_site_for_rf_adds_required_rf_columns():
    df = pd.DataFrame(
        {
            "cell_id": ["11625_1"],
            "lat": [28.63],
            "lon": [77.35],
            "azimuth": [120],
            "Etilt": [4],
            "Mtilt": [1],
            "Height": [32],
            "tx_power": [46],
            "frequency": [1800],
        }
    )
    out = _normalize_site_for_rf(df)
    assert out.loc[0, "Node_Cell_ID"] == "11625_1"
    assert out.loc[0, "electrical_tilt"] == 4
    assert out.loc[0, "mechanical_tilt"] == 1
    assert out.loc[0, "antenna_height"] == 32
    assert out.loc[0, "frequency_mhz"] == 1800
