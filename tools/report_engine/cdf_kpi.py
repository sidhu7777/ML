"""
CDF (Cumulative Distribution Function) Generation Module
========================================================
Fetches CDF data from API and generates distribution graphs for all KPIs.

API Endpoint: http://192.168.1.67:5224/api/MapView/kpi-distribution

https://api.stracer.vinfocom.co.in/api/MapView/kpi-distribution?sessionIds=2696,2678,2822&kpi=rsrp
"""

import os
import requests
import matplotlib.pyplot as plt
import pandas as pd
from .kpi_config import KPI_CONFIG


# API Configuration
OUTPUT_DIR = "data/images/kpi_analysis"

BASE_API_URL = os.getenv("BASE_API_URL")


def fetch_cdf_data_from_api(session_ids: list, kpi: str) -> dict:
    """
    Fetch CDF distribution data from API
    
    Args:
        session_ids: List of session IDs
        kpi: KPI name (e.g., 'rsrp', 'rsrq', 'sinr')
    
    Returns:
        dict: JSON response from API or None if failed
    """
    # Convert session IDs to comma-separated string
    session_ids_str = ",".join(map(str, session_ids))
    
    # Build API URL
    url = f"{BASE_API_URL}?sessionIds={session_ids_str}&kpi={kpi}"
    
    print(f"   📡 Fetching CDF data for {kpi.upper()}...")
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"    CDF data fetched successfully")
            return data
        else:
            print(f"    API returned status code: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"     Request timed out for {kpi}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"     Request failed for {kpi}: {e}")
        return None
    except Exception as e:
        print(f"     Error fetching {kpi}: {e}")
        return None


def generate_cdf_plot(cdf_data: dict, kpi: str, output_dir: str = OUTPUT_DIR) -> str:
    """
    Generate CDF plot from API data
    
    Args:
        cdf_data: JSON data from API containing CDF distribution
        kpi: KPI name (column name like 'rsrp', 'rsrq')
        output_dir: Directory to save the image
    
    Returns:
        str: Path to saved image file or None if failed
    """
    if not cdf_data:
        return None
    
    try:
        # Parse API response format: {"Status": ..., "KPI": ..., "Data": [...]}
        if "Data" not in cdf_data:
            print(f"    No 'Data' key in API response for {kpi}")
            return None
        
        data_content = cdf_data["Data"]
        
        if not isinstance(data_content, list) or len(data_content) == 0:
            print(f"    Empty or invalid data for {kpi}")
            return None
        
        # Extract values and calculate CDF percentage
        # API format: [{"value": x, "count": y, "percentage": z, "cumulative_count": cc}, ...]
        if not isinstance(data_content[0], dict):
            print(f"    Invalid data format for {kpi}")
            return None
        
        if "value" not in data_content[0] or "cumulative_count" not in data_content[0]:
            print(f"    Missing required keys in data for {kpi}")
            return None
        
        # Calculate CDF percentage from cumulative count
        total_count = max(item["cumulative_count"] for item in data_content)
        values = [item["value"] for item in data_content]
        cdf = [(item["cumulative_count"] / total_count) * 100 for item in data_content]
        
        if not values or not cdf:
            print(f"    No valid data points for {kpi}")
            return None
        
        # Create CDF plot
        plt.figure(figsize=(10, 6))
        plt.plot(values, cdf, linewidth=2, color='#2E86AB', marker='o', 
                 markersize=3, markevery=max(1, len(values) // 50))
        
        plt.xlabel(f'{kpi.upper()} Value', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Probability (%)', fontsize=12, fontweight='bold')
        plt.title(f'CDF Distribution - {kpi.upper()}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0, 100)
        
        # Add percentile markers (P10, P50, P90)
        percentiles = [10, 50, 90]
        for p in percentiles:
            # Find closest CDF value to percentile
            idx = min(range(len(cdf)), key=lambda i: abs(cdf[i] - p))
            if idx < len(values):
                plt.axhline(y=p, color='red', linestyle='--', alpha=0.3, linewidth=1)
                plt.axvline(x=values[idx], color='red', linestyle='--', alpha=0.3, linewidth=1)
                plt.text(values[idx], p + 2, f'P{p}: {values[idx]:.2f}', 
                        fontsize=9, color='red', ha='center')
        
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        output_path = os.path.join(output_dir, f"cdf_{kpi.lower()}.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"    CDF plot saved: cdf_{kpi.lower()}.png")

        # For PCI, also generate a PCI vs CDF table image from API data
        if kpi.lower() == "pci":
            try:
                # Use API data: sort by contribution (count desc), take top 30,
                # and compute cumulative % over total_count.
                rows = [r for r in data_content if isinstance(r, dict)]
                if not rows:
                    return output_path

                # Prefer explicit count; fallback to 0 if missing
                rows_sorted = sorted(
                    rows,
                    key=lambda r: r.get("count", 0),
                    reverse=True
                )[:30]

                running = 0
                table_rows = []
                for item in rows_sorted:
                    cnt = item.get("count", 0) or 0
                    running += cnt
                    pci_val = item.get("value")
                    cdf_pct = (running / total_count) * 100 if total_count else 0
                    table_rows.append({
                        "PCI": pci_val,
                        "CDF (%)": round(cdf_pct, 2)
                    })

                if table_rows:
                    df = pd.DataFrame(table_rows)

                    # Render table image
                    fig, ax = plt.subplots(figsize=(12, max(2.5, len(df) * 0.35)))
                    ax.axis("off")
                    table = ax.table(
                        cellText=df.values,
                        colLabels=df.columns,
                        loc="center",
                        cellLoc="center"
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.4)

                    # Header styling
                    for i in range(len(df.columns)):
                        cell = table[(0, i)]
                        cell.set_facecolor('#4472C4')
                        cell.set_text_props(weight='bold', color='white')

                    plt.tight_layout(pad=1.0)
                    table_path = os.path.join(output_dir, "pci_cdf_table.png")
                    plt.savefig(table_path, dpi=200, bbox_inches='tight')
                    plt.close()
                    print("    PCI CDF table saved: pci_cdf_table.png")
            except Exception as e:
                print(f"    Error generating PCI CDF table: {e}")

        return output_path
        
    except Exception as e:
        print(f"    Error generating CDF plot for {kpi}: {e}")
        return None


def generate_all_cdf_plots(session_ids: list, output_dir: str = OUTPUT_DIR):
    """
    Generate CDF plots for all range-based KPIs and PCI
    
    Args:
        session_ids: List of session IDs for the project
        output_dir: Directory to save the images
    
    Returns:
        dict: Summary of successful and failed KPIs
    """
    if not session_ids:
        print(" No session IDs provided for CDF generation")
        return {"successful": [], "failed": []}
    
    print("\n" + "=" * 80)
    print("GENERATING CDF DISTRIBUTION GRAPHS FROM API")
    print("=" * 80)
    print(f"Session IDs: {session_ids}")
    print(f"Output Dir: {output_dir}")
    
    successful_kpis = []
    failed_kpis = []
    
    # Process each range-based KPI from configuration
    for kpi_name, kpi_config in KPI_CONFIG.items():
        # Process range-based KPIs and PCI
        if kpi_config["type"] not in ["range", "categorical"]:
            continue
        
        # Skip Band (categorical but not PCI)
        if kpi_config["type"] == "categorical" and kpi_name != "PCI":
            continue
        
        # Get the column name (API uses these as KPI parameter)
        kpi_column = kpi_config["column"]  # e.g., 'rsrp', 'rsrq', 'sinr'
        
        print(f"\n Processing: {kpi_name} (API parameter: {kpi_column})")
        
        # Fetch CDF data from API
        cdf_data = fetch_cdf_data_from_api(session_ids, kpi_column)
        
        if cdf_data:
            # Generate CDF plot
            output_file = generate_cdf_plot(cdf_data, kpi_column, output_dir)
            
            if output_file:
                successful_kpis.append(kpi_name)
            else:
                failed_kpis.append(kpi_name)
        else:
            failed_kpis.append(kpi_name)
    
    # Print summary
    print("\n" + "=" * 80)
    print("CDF GENERATION SUMMARY")
    print("=" * 80)
    print(f" Successful: {len(successful_kpis)} KPIs")
    if successful_kpis:
        for kpi in successful_kpis:
            print(f"   - {kpi}")
    
    if failed_kpis:
        print(f"\n Failed: {len(failed_kpis)} KPIs")
        for kpi in failed_kpis:
            print(f"   - {kpi}")
    
    print("=" * 80 + "\n")
    
    return {
        "successful": successful_kpis,
        "failed": failed_kpis
    }
