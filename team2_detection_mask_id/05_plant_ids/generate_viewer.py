from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import MultiPoint, Point, shape


SECTOR_COLORS = [
    "#4e9a8f",
    "#e07b39",
    "#6a5acd",
    "#c4a035",
    "#3a7abf",
    "#b85c8a",
    "#5a9e4a",
    "#888",
]
NOISE_COLOR = "#888"


def _load_json(path: Path) -> dict:
    """Load a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_geojson_for_webmap(path: Path) -> dict:
    """Load a GeoJSON file and reproject it to EPSG:4326 for MapLibre display."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return json.loads(gdf.to_json())


def _load_optional_geojson(path: Path) -> dict | None:
    """Load an optional GeoJSON file if it exists."""
    if not path.exists():
        return None
    try:
        return _load_geojson_for_webmap(path)
    except Exception:
        return None


def _to_uint8(tile: np.ndarray) -> np.ndarray:
    """Convert raster values into uint8 for PNG export."""
    if tile.dtype == np.uint8:
        return tile
    if np.issubdtype(tile.dtype, np.integer):
        dtype_max = np.iinfo(tile.dtype).max
        if dtype_max <= 255:
            return tile.astype(np.uint8)
        scaled = np.clip(tile.astype(np.float32) / dtype_max * 255.0, 0, 255)
        return scaled.astype(np.uint8)

    finite_tile = np.nan_to_num(tile.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(finite_tile.min())
    max_value = float(finite_tile.max())
    if max_value <= min_value:
        return np.zeros_like(finite_tile, dtype=np.uint8)
    scaled = (finite_tile - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _prepare_background_overlay(summary_data: dict, viewer_dir: Path) -> dict | None:
    """Copy or generate a viewer-friendly orthomosaic background image with map bounds."""
    run_summary_path = summary_data.get("run_summary_path")
    if not run_summary_path:
        return None

    run_summary_file = Path(run_summary_path).resolve()
    if not run_summary_file.exists():
        return None

    run_summary = _load_json(run_summary_file)
    overlay_raster = run_summary.get("overlay_raster")
    if not overlay_raster:
        return None

    overlay_raster_path = Path(overlay_raster).resolve()
    if not overlay_raster_path.exists():
        return None

    preview_png_path = viewer_dir / "orthomosaic_overlay_preview.png"
    preview_png = run_summary.get("preview_png")
    preview_source_path = Path(preview_png).resolve() if preview_png else None

    if preview_source_path and preview_source_path.exists():
        if preview_source_path != preview_png_path:
            shutil.copyfile(preview_source_path, preview_png_path)
    else:
        with rasterio.open(overlay_raster_path) as src:
            scale = min(4096 / src.width, 4096 / src.height, 1.0)
            preview_width = max(1, int(round(src.width * scale)))
            preview_height = max(1, int(round(src.height * scale)))
            rgb = src.read(
                [1, 2, 3] if src.count >= 3 else [1, 1, 1],
                out_shape=(3, preview_height, preview_width),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            preview_rgb = np.transpose(rgb, (1, 2, 0))
            preview_rgb = _to_uint8(preview_rgb)
            cv2.imwrite(str(preview_png_path), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))

    with rasterio.open(overlay_raster_path) as src:
        left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    return {
        "image_path": preview_png_path.name,
        "coordinates": [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
    }


def _sector_color(sector_id: int) -> str:
    """Return the color for a given sector ID."""
    if sector_id == -1:
        return NOISE_COLOR
    return SECTOR_COLORS[sector_id % 8]


def _build_sector_hulls(plants_geojson: dict) -> dict:
    """Build convex hull polygons for each non-noise sector."""
    grouped_points: dict[int, list[Point]] = {}
    for feature in plants_geojson.get("features", []):
        properties = feature.get("properties", {})
        sector_id = int(properties.get("sector_id", -1))
        if sector_id == -1:
            continue
        grouped_points.setdefault(sector_id, []).append(shape(feature["geometry"]))

    features: list[dict] = []
    for sector_id, points in grouped_points.items():
        if len(points) < 3:
            continue

        hull = MultiPoint(points).convex_hull
        if hull.geom_type != "Polygon":
            hull = hull.buffer(0.1)
        if hull.is_empty:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "sector_id": sector_id,
                    "sector_label": f"S{sector_id:02d}",
                    "color": _sector_color(sector_id),
                },
                "geometry": hull.__geo_interface__,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def _make_html(
    plants_data: dict,
    sectors_data: dict,
    summary_data: dict,
    boundary_data: dict | None,
    exclusion_data: dict | None,
    background_overlay: dict | None,
) -> str:
    """Build the self-contained HTML viewer."""
    plants_json = json.dumps(plants_data, ensure_ascii=False)
    sectors_json = json.dumps(sectors_data, ensure_ascii=False)
    summary_json = json.dumps(summary_data, ensure_ascii=False)
    boundary_json = json.dumps(boundary_data, ensure_ascii=False) if boundary_data else "null"
    exclusion_json = json.dumps(exclusion_data, ensure_ascii=False) if exclusion_data else "null"
    background_json = json.dumps(background_overlay, ensure_ascii=False) if background_overlay else "null"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Plant ID Viewer</title>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0b0b0b;
      color: #fff;
      font: 12px/1.4 sans-serif;
    }}
    #map {{
      position: absolute;
      inset: 0;
      background: #101010;
    }}
    .panel {{
      position: absolute;
      background: rgba(15, 15, 15, 0.85);
      color: #fff;
      padding: 12px;
      border-radius: 10px;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
      z-index: 2;
      backdrop-filter: blur(6px);
    }}
    #controls {{
      top: 12px;
      left: 12px;
      width: 270px;
    }}
    #stats {{
      top: 12px;
      right: 12px;
      width: 280px;
      max-height: calc(100vh - 24px);
      overflow: auto;
    }}
    #details {{
      top: 140px;
      right: 12px;
      width: 280px;
      display: none;
    }}
    .panel h3 {{
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 700;
    }}
    .row {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .row input[type="text"] {{
      flex: 1;
      min-width: 0;
      padding: 7px 8px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.05);
      color: #fff;
      border-radius: 6px;
    }}
    .row button {{
      padding: 7px 10px;
      border: 0;
      border-radius: 6px;
      background: #2e2e2e;
      color: #fff;
      cursor: pointer;
    }}
    label.toggle {{
      display: block;
      margin: 6px 0;
    }}
    .stat-line {{
      margin: 4px 0;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 52px 1fr 38px;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
    }}
    .bar {{
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: 88px 1fr;
      gap: 6px 8px;
    }}
    .hint {{
      opacity: 0.75;
      font-size: 11px;
      margin-top: 8px;
    }}
    .collapse-btn {{
      float: right;
      margin-top: -2px;
      background: transparent;
      border: 1px solid rgba(255,255,255,0.15);
      color: #fff;
      border-radius: 6px;
      cursor: pointer;
      padding: 2px 6px;
    }}
    .tooltip {{
      position: absolute;
      z-index: 3;
      pointer-events: none;
      background: rgba(15,15,15,0.92);
      color: #fff;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      display: none;
      white-space: nowrap;
      box-shadow: 0 4px 18px rgba(0,0,0,0.3);
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="tooltip" class="tooltip"></div>

  <div id="controls" class="panel">
    <h3>Layers & Search</h3>
    <div class="row">
      <input id="searchInput" type="text" placeholder="Search plant_id, sector, row..." />
      <button id="clearSearch">Clear</button>
    </div>
    <label class="toggle"><input id="togglePlants" type="checkbox" checked /> Plants</label>
    <label class="toggle"><input id="toggleSectors" type="checkbox" checked /> Sector boundaries</label>
    <label class="toggle"><input id="toggleBoundary" type="checkbox" checked /> Farm boundary</label>
    <label class="toggle"><input id="toggleExclusion" type="checkbox" checked /> Exclusion zones</label>
    <div class="hint">Search filters visible plants by plant ID substring.</div>
  </div>

  <div id="stats" class="panel">
    <button id="toggleStats" class="collapse-btn">Hide</button>
    <h3>Run Stats</h3>
    <div id="statsBody"></div>
  </div>

  <div id="details" class="panel">
    <h3>Selected Plant</h3>
    <div id="detailsBody" class="detail-grid"></div>
  </div>

  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <script>
    const PLANTS_DATA = {plants_json};
    const SECTORS_DATA = {sectors_json};
    const SUMMARY_DATA = {summary_json};
    const BOUNDARY_DATA = {boundary_json};
    const EXCLUSION_DATA = {exclusion_json};
    const BACKGROUND_OVERLAY = {background_json};
    const SECTOR_COLORS = {json.dumps(SECTOR_COLORS)};
    const NOISE_COLOR = "{NOISE_COLOR}";

    const map = new maplibregl.Map({{
      container: "map",
      style: {{
        version: 8,
        glyphs: "https://demotiles.maplibre.org/font/{{fontstack}}/{{range}}.pbf",
        sources: {{}},
        layers: [
          {{
            id: "background",
            type: "background",
            paint: {{
              "background-color": "#0e1116"
            }}
          }}
        ]
      }},
      center: [0, 0],
      zoom: 2
    }});

    const tooltip = document.getElementById("tooltip");
    const detailsPanel = document.getElementById("details");
    const detailsBody = document.getElementById("detailsBody");
    const statsPanel = document.getElementById("stats");
    const statsBody = document.getElementById("statsBody");
    const searchInput = document.getElementById("searchInput");
    const clearSearch = document.getElementById("clearSearch");

    let allPlantFeatures = PLANTS_DATA.features || [];
    let selectedFeature = null;
    let statsCollapsed = false;

    function getSectorColor(sectorId) {{
      if (sectorId === -1) return NOISE_COLOR;
      return SECTOR_COLORS[((sectorId % 8) + 8) % 8];
    }}

    function featureMatchesSearch(feature, query) {{
      if (!query) return true;
      const q = query.toLowerCase();
      const p = feature.properties || {{}};
      return [
        p.plant_id,
        p.sector_label,
        p.row_index ? `R${{String(p.row_index).padStart(3, "0")}}` : "",
        p.col_index ? `C${{String(p.col_index).padStart(3, "0")}}` : ""
      ].join(" ").toLowerCase().includes(q);
    }}

    function filterPlants(query) {{
      const filteredFeatures = allPlantFeatures.filter(feature => featureMatchesSearch(feature, query));
      const filteredGeoJson = {{
        type: "FeatureCollection",
        features: filteredFeatures
      }};
      map.getSource("plants").setData(filteredGeoJson);

      const visibleSectorIds = new Set(
        filteredFeatures
          .map(feature => Number(feature.properties?.sector_id ?? -1))
          .filter(value => value !== -1)
      );
      const filteredSectors = {{
        type: "FeatureCollection",
        features: (SECTORS_DATA.features || []).filter(feature =>
          visibleSectorIds.has(Number(feature.properties?.sector_id ?? -1))
        )
      }};
      if (map.getSource("sectors")) {{
        map.getSource("sectors").setData(filteredSectors);
      }}

      if (selectedFeature && !featureMatchesSearch(selectedFeature, query)) {{
        selectedFeature = null;
        detailsPanel.style.display = "none";
        map.getSource("selectedPlant").setData({{ type: "FeatureCollection", features: [] }});
      }}
    }}

    function fitToPlants() {{
      const bounds = new maplibregl.LngLatBounds();
      let count = 0;
      for (const feature of allPlantFeatures) {{
        if (feature.geometry?.type === "Point") {{
          bounds.extend(feature.geometry.coordinates);
          count += 1;
        }}
      }}
      if (count > 0) {{
        map.fitBounds(bounds, {{ padding: 40, duration: 0 }});
      }}
    }}

    function showTooltip(event, feature) {{
      const p = feature.properties || {{}};
      tooltip.innerHTML = `
        <div><strong>Plant ID:</strong> ${{p.plant_id ?? "N/A"}}</div>
        <div><strong>Sector:</strong> ${{p.sector_label ?? "N/A"}}</div>
        <div><strong>Row / Col:</strong> R${{String(p.row_index ?? 0).padStart(3, "0")}} / C${{String(p.col_index ?? 0).padStart(3, "0")}}</div>
        <div><strong>Confidence:</strong> ${{p.confidence ?? "N/A"}}</div>
      `;
      tooltip.style.display = "block";
      tooltip.style.left = `${{event.point.x + 14}}px`;
      tooltip.style.top = `${{event.point.y + 14}}px`;
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function showDetails(feature) {{
      const p = feature.properties || {{}};
      const coords = feature.geometry?.coordinates || [null, null];
      detailsBody.innerHTML = `
        <div>Plant ID</div><div>${{p.plant_id ?? "N/A"}}</div>
        <div>Sector</div><div>${{p.sector_label ?? "N/A"}}</div>
        <div>Row / Col</div><div>R${{String(p.row_index ?? 0).padStart(3, "0")}} / C${{String(p.col_index ?? 0).padStart(3, "0")}}</div>
        <div>Confidence</div><div>${{p.confidence ?? "N/A"}}</div>
        <div>Geo X</div><div>${{coords[0] == null ? "N/A" : Number(coords[0]).toFixed(6)}}</div>
        <div>Geo Y</div><div>${{coords[1] == null ? "N/A" : Number(coords[1]).toFixed(6)}}</div>
      `;
      detailsPanel.style.display = "block";
      selectedFeature = feature;
      map.getSource("selectedPlant").setData({{
        type: "FeatureCollection",
        features: [feature]
      }});
    }}

    function renderStats() {{
      const totalPlants = Number(SUMMARY_DATA.total_plants || allPlantFeatures.length || 0);
      const noisePlants = Number(SUMMARY_DATA.noise_plants || 0);
      const plantsBySector = SUMMARY_DATA.plants_by_sector || {{}};
      const maxCount = Math.max(1, ...Object.values(plantsBySector).map(Number));
      const runTimestamp = SUMMARY_DATA.run_timestamp || "N/A";

      const bars = Object.entries(plantsBySector).sort((a, b) => a[0].localeCompare(b[0])).map(([sectorLabel, count]) => {{
        const sectorId = Number((sectorLabel || "S00").replace("S", "")) || 0;
        const width = (Number(count) / maxCount) * 100;
        return `
          <div class="bar-row">
            <div>${{sectorLabel}}</div>
            <div class="bar"><div class="bar-fill" style="width:${{width}}%;background:${{getSectorColor(sectorId)}}"></div></div>
            <div>${{count}}</div>
          </div>
        `;
      }}).join("");

      statsBody.innerHTML = `
        <div class="stat-line"><strong>Total plants detected:</strong> ${{totalPlants}}</div>
        <div class="stat-line"><strong>Noise / unassigned:</strong> ${{noisePlants}}</div>
        <div class="stat-line"><strong>Run timestamp:</strong> ${{runTimestamp}}</div>
        <div style="margin-top:10px;"><strong>Plants per sector</strong></div>
        ${{bars || "<div class='hint'>No clustered sectors available.</div>"}}
      `;
    }}

    function setLayerVisibility(layerId, visible) {{
      if (map.getLayer(layerId)) {{
        map.setLayoutProperty(layerId, "visibility", visible ? "visible" : "none");
      }}
    }}

    map.on("load", () => {{
      map.addSource("plants", {{
        type: "geojson",
        data: PLANTS_DATA
      }});
      map.addSource("selectedPlant", {{
        type: "geojson",
        data: {{ type: "FeatureCollection", features: [] }}
      }});
      if (BACKGROUND_OVERLAY) {{
        map.addSource("orthomosaicOverlay", {{
          type: "image",
          url: BACKGROUND_OVERLAY.image_path,
          coordinates: BACKGROUND_OVERLAY.coordinates
        }});
      }}
      map.addSource("sectors", {{
        type: "geojson",
        data: SECTORS_DATA
      }});

      if (BOUNDARY_DATA) {{
        map.addSource("farmBoundary", {{
          type: "geojson",
          data: BOUNDARY_DATA
        }});
      }}

      if (EXCLUSION_DATA) {{
        map.addSource("exclusionZones", {{
          type: "geojson",
          data: EXCLUSION_DATA
        }});
      }}

      if (BACKGROUND_OVERLAY) {{
        map.addLayer({{
          id: "orthomosaic-overlay",
          type: "raster",
          source: "orthomosaicOverlay",
          paint: {{
            "raster-opacity": 1.0,
            "raster-resampling": "linear"
          }}
        }});
      }}

      map.addLayer({{
        id: "sector-fill",
        type: "fill",
        source: "sectors",
        paint: {{
          "fill-color": ["get", "color"],
          "fill-opacity": 0.10
        }}
      }});
      map.addLayer({{
        id: "sector-outline",
        type: "line",
        source: "sectors",
        paint: {{
          "line-color": ["get", "color"],
          "line-width": 2
        }}
      }});

      if (BOUNDARY_DATA) {{
        map.addLayer({{
          id: "farm-boundary",
          type: "line",
          source: "farmBoundary",
          paint: {{
            "line-color": "#ffffff",
            "line-width": 2,
            "line-dasharray": [2, 2]
          }}
        }});
      }}

      if (EXCLUSION_DATA) {{
        map.addLayer({{
          id: "exclusion-fill",
          type: "fill",
          source: "exclusionZones",
          paint: {{
            "fill-color": "#d44",
            "fill-opacity": 0.18
          }}
        }});
        map.addLayer({{
          id: "exclusion-outline",
          type: "line",
          source: "exclusionZones",
          paint: {{
            "line-color": "#ff5555",
            "line-width": 1.5,
            "line-dasharray": [1, 1]
          }}
        }});
      }}

      map.addLayer({{
        id: "plants",
        type: "circle",
        source: "plants",
        paint: {{
          "circle-radius": [
            "case",
            ["==", ["get", "sector_id"], -1], 3, 5
          ],
          "circle-color": [
            "case",
            ["==", ["get", "sector_id"], -1], NOISE_COLOR,
            ["match", ["get", "sector_id"],
              0, SECTOR_COLORS[0],
              1, SECTOR_COLORS[1],
              2, SECTOR_COLORS[2],
              3, SECTOR_COLORS[3],
              4, SECTOR_COLORS[4],
              5, SECTOR_COLORS[5],
              6, SECTOR_COLORS[6],
              7, SECTOR_COLORS[7],
              SECTOR_COLORS[0]
            ]
          ],
          "circle-stroke-color": "#101010",
          "circle-stroke-width": 1
        }}
      }});

      map.addLayer({{
        id: "selected-plant-ring",
        type: "circle",
        source: "selectedPlant",
        paint: {{
          "circle-radius": 9,
          "circle-color": "rgba(0,0,0,0)",
          "circle-stroke-color": "#ffd84a",
          "circle-stroke-width": 3
        }}
      }});

      map.on("mousemove", "plants", (event) => {{
        const feature = event.features && event.features[0];
        if (!feature) return;
        map.getCanvas().style.cursor = "pointer";
        showTooltip(event, feature);
      }});

      map.on("mouseleave", "plants", () => {{
        map.getCanvas().style.cursor = "";
        hideTooltip();
      }});

      map.on("click", "plants", (event) => {{
        const feature = event.features && event.features[0];
        if (!feature) return;
        showDetails(feature);
      }});

      document.getElementById("togglePlants").addEventListener("change", (event) => {{
        setLayerVisibility("plants", event.target.checked);
        setLayerVisibility("selected-plant-ring", event.target.checked);
      }});
      document.getElementById("toggleSectors").addEventListener("change", (event) => {{
        setLayerVisibility("sector-fill", event.target.checked);
        setLayerVisibility("sector-outline", event.target.checked);
      }});
      document.getElementById("toggleBoundary").addEventListener("change", (event) => {{
        setLayerVisibility("farm-boundary", event.target.checked);
      }});
      document.getElementById("toggleExclusion").addEventListener("change", (event) => {{
        setLayerVisibility("exclusion-fill", event.target.checked);
        setLayerVisibility("exclusion-outline", event.target.checked);
      }});

      searchInput.addEventListener("input", () => {{
        filterPlants(searchInput.value.trim());
      }});
      clearSearch.addEventListener("click", () => {{
        searchInput.value = "";
        filterPlants("");
      }});

      document.getElementById("toggleStats").addEventListener("click", () => {{
        statsCollapsed = !statsCollapsed;
        statsBody.style.display = statsCollapsed ? "none" : "block";
        document.getElementById("toggleStats").textContent = statsCollapsed ? "Show" : "Hide";
      }});

      renderStats();
      fitToPlants();
    }});
  </script>
</body>
</html>
"""


def generate_viewer(geojson_path: str, summary_path: str, output_path: str | None = None) -> Path:
    """Generate a self-contained HTML viewer for plant IDs."""
    geojson_file = Path(geojson_path).resolve()
    summary_file = Path(summary_path).resolve()
    if output_path:
        viewer_path = Path(output_path).resolve()
    else:
        viewer_path = geojson_file.resolve().parent / "viewer.html"

    plants_data = _load_geojson_for_webmap(geojson_file)
    summary_data = _load_json(summary_file)
    sectors_data = _build_sector_hulls(plants_data)
    boundary_data = _load_optional_geojson(viewer_path.parent / "farm_boundary.geojson")
    exclusion_data = _load_optional_geojson(viewer_path.parent / "exclusion_mask_outline.geojson")
    background_overlay = _prepare_background_overlay(summary_data, viewer_path.parent)

    viewer_html = _make_html(
      plants_data=plants_data,
      sectors_data=sectors_data,
      summary_data=summary_data,
      boundary_data=boundary_data,
      exclusion_data=exclusion_data,
      background_overlay=background_overlay,
    )
    viewer_path.parent.mkdir(parents=True, exist_ok=True)
    viewer_path.write_text(viewer_html, encoding="utf-8")
    return viewer_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for viewer generation."""
    parser = argparse.ArgumentParser(description="Generate a self-contained HTML viewer for plant ID outputs.")
    parser.add_argument("--geojson", required=True, help="Path to plants.geojson.")
    parser.add_argument("--summary", required=True, help="Path to plant_id_summary.json.")
    parser.add_argument("--output", help="Optional output path for viewer.html.")
    return parser.parse_args()


def main() -> None:
    """Run the viewer generation CLI."""
    args = parse_args()
    viewer_path = generate_viewer(args.geojson, args.summary, args.output)
    print(f"Viewer written to: {viewer_path}")
    print(f"Open in browser: open {viewer_path}")


if __name__ == "__main__":
    main()
