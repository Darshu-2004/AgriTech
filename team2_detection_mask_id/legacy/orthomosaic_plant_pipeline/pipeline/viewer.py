from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from shapely.geometry import MultiPoint


SECTOR_COLORS = [
    "#1f7a8c",
    "#bf5b17",
    "#4c6a92",
    "#8f8d23",
    "#2f8f5b",
    "#954c76",
    "#b85c38",
    "#5b6b7a",
]
NOISE_COLOR = "#8a8a8a"


def _sector_color(sector_id: int) -> str:
    if sector_id == -1:
        return NOISE_COLOR
    return SECTOR_COLORS[sector_id % len(SECTOR_COLORS)]


def build_sector_polygons(plants: list[dict]) -> list[dict]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for plant in plants:
        sector_id = int(plant.get("sector_id", -1))
        if sector_id == -1:
            continue
        grouped.setdefault(sector_id, []).append((float(plant["preview_x"]), float(plant["preview_y"])))

    polygons: list[dict] = []
    for sector_id, points in grouped.items():
        if len(points) < 3:
            continue

        hull = MultiPoint(points).convex_hull
        if hull.is_empty:
            continue
        if hull.geom_type != "Polygon":
            hull = hull.buffer(12.0)
        if hull.is_empty or hull.geom_type != "Polygon":
            continue

        coordinates = [[round(x, 2), round(y, 2)] for x, y in hull.exterior.coords]
        polygons.append(
            {
                "sector_id": sector_id,
                "sector_label": f"S{sector_id:02d}",
                "color": _sector_color(sector_id),
                "points": coordinates,
            }
        )
    return polygons


def _build_summary(plants: list[dict], metadata: dict) -> dict:
    sector_counts = Counter(plant["sector_label"] for plant in plants if int(plant["sector_id"]) != -1)
    return {
        "pipeline_name": metadata["pipeline_name"],
        "source_orthomosaic": metadata["source"],
        "run_timestamp": metadata["run_timestamp"],
        "total_plants": len(plants),
        "noise_plants": sum(1 for plant in plants if int(plant["sector_id"]) == -1),
        "plants_by_sector": dict(sorted(sector_counts.items())),
        "preview_width": metadata["preview_width"],
        "preview_height": metadata["preview_height"],
    }


def generate_viewer(
    plants: list[dict],
    metadata: dict,
    preview_image_name: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sectors = build_sector_polygons(plants)
    summary = _build_summary(plants, metadata)
    plants_json = json.dumps(plants, ensure_ascii=False)
    sectors_json = json.dumps(sectors, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Orthomosaic Plant Viewer</title>
  <style>
    :root {{
      --panel: rgba(12, 22, 24, 0.9);
      --panel-border: rgba(255, 255, 255, 0.12);
      --ink: #132126;
      --muted: #5d6d73;
      --shadow: 0 18px 48px rgba(0, 0, 0, 0.18);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(31, 122, 140, 0.12), transparent 32%),
        linear-gradient(180deg, #f4f0e6 0%, #ddd8c8 100%);
      color: var(--ink);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      padding: 24px;
      border-right: 1px solid rgba(19, 33, 38, 0.08);
      background: rgba(255, 252, 246, 0.72);
      backdrop-filter: blur(6px);
    }}
    .sidebar h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.05;
    }}
    .subtitle {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    .panel {{
      background: var(--panel);
      color: #f8fbfb;
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      padding: 16px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .search {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid rgba(19, 33, 38, 0.12);
      background: rgba(255,255,255,0.78);
      font-size: 14px;
      margin-bottom: 16px;
    }}
    .toggles label {{
      display: block;
      margin: 10px 0;
      font-size: 14px;
    }}
    .stats-line {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin: 8px 0;
      font-size: 14px;
    }}
    .viewer-wrap {{
      padding: 20px;
      overflow: auto;
    }}
    .viewer-frame {{
      display: inline-block;
      position: relative;
      border-radius: 24px;
      overflow: hidden;
      box-shadow: var(--shadow);
      background: #243438;
    }}
    #bg {{
      display: block;
      max-width: min(100vw - 420px, 1400px);
      height: auto;
    }}
    #overlay {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }}
    .plant {{
      cursor: pointer;
    }}
    .plant-label {{
      font-size: 10px;
      font-weight: 700;
      paint-order: stroke;
      stroke: rgba(12, 22, 24, 0.78);
      stroke-width: 3px;
      stroke-linejoin: round;
      fill: #fffdf4;
      pointer-events: none;
    }}
    .sector {{
      fill-opacity: 0.12;
      stroke-width: 2;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(12, 22, 24, 0.94);
      color: #f5faf9;
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 12px;
      box-shadow: var(--shadow);
      display: none;
      z-index: 20;
      max-width: 260px;
    }}
    .details-grid {{
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 8px 10px;
      font-size: 13px;
    }}
    .sector-bar {{
      display: grid;
      grid-template-columns: 52px 1fr 40px;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
    }}
    .sector-bar-fill {{
      height: 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.18);
      overflow: hidden;
    }}
    .sector-bar-fill span {{
      display: block;
      height: 100%;
      border-radius: 999px;
    }}
    @media (max-width: 1100px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        border-right: 0;
        border-bottom: 1px solid rgba(19, 33, 38, 0.08);
      }}
      #bg {{
        max-width: calc(100vw - 40px);
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>Plant Map Viewer</h1>
      <p class="subtitle">Masked orthomosaic background with plant IDs and sector overlays.</p>
      <input id="search" class="search" type="text" placeholder="Search plant ID, sector, row or column" />
      <div class="panel toggles">
        <h2>Layers</h2>
        <label><input id="toggleSectors" type="checkbox" checked /> Show sectors</label>
        <label><input id="togglePlants" type="checkbox" checked /> Show plant points</label>
        <label><input id="toggleLabels" type="checkbox" /> Show ID labels</label>
      </div>
      <div class="panel">
        <h2>Run Summary</h2>
        <div id="summary"></div>
      </div>
      <div class="panel">
        <h2>Selected Plant</h2>
        <div id="details">Click a plant point to inspect its details.</div>
      </div>
    </aside>
    <main class="viewer-wrap">
      <div class="viewer-frame">
        <img id="bg" src="{preview_image_name}" alt="Masked orthomosaic preview" />
        <svg id="overlay" viewBox="0 0 {metadata["preview_width"]} {metadata["preview_height"]}" preserveAspectRatio="xMidYMid meet"></svg>
      </div>
    </main>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const plants = {plants_json};
    const sectors = {sectors_json};
    const summary = {summary_json};
    const tooltip = document.getElementById("tooltip");
    const overlay = document.getElementById("overlay");
    const details = document.getElementById("details");
    const summaryNode = document.getElementById("summary");
    const search = document.getElementById("search");
    const toggleSectors = document.getElementById("toggleSectors");
    const togglePlants = document.getElementById("togglePlants");
    const toggleLabels = document.getElementById("toggleLabels");

    function sectorColor(sectorId) {{
      if (sectorId === -1) return "{NOISE_COLOR}";
      const colors = {json.dumps(SECTOR_COLORS)};
      return colors[((sectorId % colors.length) + colors.length) % colors.length];
    }}

    function renderSummary() {{
      const entries = Object.entries(summary.plants_by_sector || {{}});
      const maxCount = Math.max(1, ...entries.map(([, count]) => Number(count)));
      const bars = entries.map(([label, count]) => {{
        const sectorId = Number(String(label).replace("S", "")) || 0;
        const width = (Number(count) / maxCount) * 100;
        return `<div class="sector-bar"><div>${{label}}</div><div class="sector-bar-fill"><span style="width:${{width}}%;background:${{sectorColor(sectorId)}}"></span></div><div>${{count}}</div></div>`;
      }}).join("");
      summaryNode.innerHTML = `
        <div class="stats-line"><span>Total plants</span><strong>${{summary.total_plants}}</strong></div>
        <div class="stats-line"><span>Noise plants</span><strong>${{summary.noise_plants}}</strong></div>
        <div class="stats-line"><span>Run time</span><strong>${{summary.run_timestamp}}</strong></div>
        <div style="margin-top:12px;">${{bars || "<div>No sector clusters available.</div>"}}</div>
      `;
    }}

    function matchesQuery(plant, query) {{
      if (!query) return true;
      const q = query.toLowerCase();
      return [
        plant.plant_id,
        plant.sector_label,
        plant.instance_id,
        plant.row_index === null ? "" : `R${{String(plant.row_index).padStart(3, "0")}}`,
        plant.col_index === null ? "" : `C${{String(plant.col_index).padStart(3, "0")}}`
      ].join(" ").toLowerCase().includes(q);
    }}

    function polygonPoints(points) {{
      return points.map(([x, y]) => `${{x}},${{y}}`).join(" ");
    }}

    function showTooltip(event, plant) {{
      tooltip.style.display = "block";
      tooltip.style.left = `${{event.clientX + 16}}px`;
      tooltip.style.top = `${{event.clientY + 16}}px`;
      tooltip.innerHTML = `<div><strong>${{plant.plant_id}}</strong></div><div>Sector: ${{plant.sector_label}}</div><div>Area: ${{plant.area_px}} px</div>`;
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function renderDetails(plant) {{
      details.innerHTML = `
        <div class="details-grid">
          <div>Plant ID</div><div>${{plant.plant_id}}</div>
          <div>Sector</div><div>${{plant.sector_label}}</div>
          <div>Row / Col</div><div>R${{String(plant.row_index || 0).padStart(3, "0")}} / C${{String(plant.col_index || 0).padStart(3, "0")}}</div>
          <div>Pixel</div><div>${{plant.pixel_x}}, ${{plant.pixel_y}}</div>
          <div>Geo</div><div>${{plant.geo_x.toFixed(4)}}, ${{plant.geo_y.toFixed(4)}}</div>
          <div>Area</div><div>${{plant.area_px}} px</div>
          <div>Mask file</div><div>${{plant.mask_file}}</div>
        </div>
      `;
    }}

    function render() {{
      const query = search.value.trim();
      overlay.innerHTML = "";

      if (toggleSectors.checked) {{
        for (const sector of sectors) {{
          const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
          polygon.setAttribute("class", "sector");
          polygon.setAttribute("points", polygonPoints(sector.points));
          polygon.setAttribute("fill", sector.color);
          polygon.setAttribute("stroke", sector.color);
          overlay.appendChild(polygon);
        }}
      }}

      const visiblePlants = plants.filter(plant => matchesQuery(plant, query));
      for (const plant of visiblePlants) {{
        if (togglePlants.checked) {{
          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("class", "plant");
          circle.setAttribute("cx", plant.preview_x);
          circle.setAttribute("cy", plant.preview_y);
          circle.setAttribute("r", plant.sector_id === -1 ? 3 : 4.5);
          circle.setAttribute("fill", sectorColor(plant.sector_id));
          circle.setAttribute("stroke", "rgba(12,22,24,0.8)");
          circle.setAttribute("stroke-width", "1");
          circle.addEventListener("mouseenter", event => showTooltip(event, plant));
          circle.addEventListener("mouseleave", hideTooltip);
          circle.addEventListener("click", () => renderDetails(plant));
          overlay.appendChild(circle);
        }}

        if (toggleLabels.checked) {{
          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          label.setAttribute("class", "plant-label");
          label.setAttribute("x", Number(plant.preview_x) + 6);
          label.setAttribute("y", Number(plant.preview_y) - 6);
          label.textContent = plant.plant_id;
          overlay.appendChild(label);
        }}
      }}
    }}

    renderSummary();
    render();
    search.addEventListener("input", render);
    toggleSectors.addEventListener("change", render);
    togglePlants.addEventListener("change", render);
    toggleLabels.addEventListener("change", render);
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
