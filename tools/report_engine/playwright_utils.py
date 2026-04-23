# src/playwright_utils.py

from playwright.sync_api import sync_playwright
import os


def html_to_png(
    html_path,
    png_path,
    width=1920,
    height=1200,
    device_scale_factor=2,  # Reduced from 3 to 2 for smaller file size
    clip_to_map=True,
):
    html_path = os.path.abspath(html_path)
    html_url = "file:///" + html_path.replace("\\", "/")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=device_scale_factor,
        )
        page = context.new_page()

        # Load the page and wait for DOM ready
        page.goto(html_url, wait_until="domcontentloaded")
        
        # Wait for the map container to be present
        page.wait_for_selector(".folium-map, .leaflet-container", timeout=30000)

        # Wait for Folium/Leaflet to initialize the map object
        page.wait_for_function(
            """() => {
                const el = document.querySelector('.folium-map');
                const map = el && el.id && window[el.id];
                return !!(map && map._loaded);
            }""",
            timeout=30000,
        )

        # Invalidate map size to ensure proper rendering
        page.evaluate(
            """() => {
                const el = document.querySelector('.folium-map');
                const map = el && el.id && window[el.id];
                if (map && typeof map.invalidateSize === 'function') {
                    map.invalidateSize(true);
                }
            }"""
        )

        # Wait for tiles to load - flexible approach
        # Check multiple times to ensure tiles are actually loaded
        for attempt in range(3):
            try:
                page.wait_for_function(
                    """() => {
                        const loaded = document.querySelectorAll('.leaflet-tile-loaded').length;
                        const loading = document.querySelectorAll('.leaflet-tile-loading').length;
                        // More flexible: require at least some tiles loaded and no loading tiles
                        return loaded >= 10 && loading === 0;
                    }""",
                    timeout=15000,
                )
                # If successful, break the loop
                break
            except Exception as e:
                if attempt < 2:
                    # Wait a bit and retry
                    page.wait_for_timeout(2000)
                else:
                    # Last attempt failed, but continue anyway
                    print(f"Warning: Tile loading check failed after 3 attempts: {e}")

        # Additional wait for network to be idle (all tile requests complete)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception as e:
            # Network idle not critical, continue anyway
            pass

        # Final settling delay to ensure all rendering is complete
        page.wait_for_timeout(2000)

        # Force one more map refresh
        page.evaluate(
            """() => {
                const el = document.querySelector('.folium-map');
                const map = el && el.id && window[el.id];
                if (map && typeof map.invalidateSize === 'function') {
                    map.invalidateSize(true);
                }
            }"""
        )

        # Small delay after final refresh
        page.wait_for_timeout(500)

        if clip_to_map:
            map_el = page.query_selector(".folium-map") or page.query_selector(".leaflet-container")
            if map_el:
                box = map_el.bounding_box()
                if box:
                    page.screenshot(
                        path=png_path,
                        clip={
                            "x": box["x"],
                            "y": box["y"],
                            "width": box["width"],
                            "height": box["height"],
                        },
                    )
                    context.close()
                    browser.close()
                    return

        page.screenshot(path=png_path, full_page=True)
        context.close()
        browser.close()
