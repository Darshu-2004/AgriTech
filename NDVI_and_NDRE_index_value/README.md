# Plant Health — Frontend Data Package

This folder contains **everything the frontend team needs** to build a map/dashboard showing per-plant NDVI, NDRE, growth stage, yield, and nitrogen status across the plantation.

---

## Folder Structure

```
frontend/
├── README.md               ← You are here
├── plant_health_api.js     ← JS helper class (load & query data)
├── example_usage.html      ← Copy-paste integration example
└── data/
    ├── plants.json         ← 13,997 plants with all index values  (5.7 MB)
    ├── summary.json        ←  KPI cards + histogram/chart data    (3.9 KB)
    ├── insights.json       ←  Full ML insights report             (3.3 KB)
    └── sectors.json        ←  14-sector aggregated stats          (2.0 KB)
```

---

## Quick Start

### 1. Include in your HTML

```html
<script src="plant_health_api.js"></script>
<script>
  const api = new PlantHealthAPI('./data');   // point to the data/ folder
  api.init().then(() => {
    const summary = api.getSummary();
    console.log('Total plants:', summary.total_plants);       // 13997
    console.log('Avg NDVI:',    summary.avg_ndvi);            // 0.5108
    console.log('Total yield:', summary.total_yield_kg, 'kg');// 480373.9
  });
</script>
```

### 2. Use in React / Vue / any bundler

```js
import PlantHealthAPI from './plant_health_api.js';

const api = new PlantHealthAPI('/data');
await api.init();
```

---

## Data Schema

### `data/plants.json` — Array of plant objects

This is the **main data source** for the map. Each plant has:

| Field | Type | Description |
|---|---|---|
| `plant_id` | string | Unique ID e.g. `"PLT-S00-R001-C001-8F00DB"` |
| `lat` | number | WGS84 latitude |
| `lon` | number | WGS84 longitude |
| `sector` | string | Sector label e.g. `"S00"` – `"S13"` |
| `row_index` | number | Grid row position within sector |
| `col_index` | number | Grid column position within sector |
| `ndvi` | number | NDVI index value (−1 to 1) |
| `ndvi_category` | string | `"Dense Vegetation"`, `"Moderate Vegetation"`, etc. |
| `ndre` | number | NDRE index value (−1 to 1) |
| `ndre_category` | string | `"Very Good Vegetation"`, `"Good Vegetation"`, etc. |
| `growth_stage` | string | `"Seedling"`, `"Vegetative"`, `"Flowering"`, `"Fruiting"`, `"Mature"` |
| `growth_confidence` | number | ML confidence 0–1 |
| `yield_kg` | number | Predicted yield in kg |
| `nitrogen_status` | string | `"Healthy"` or `"Deficient"` |
| `nitrogen_confidence` | number | ML confidence 0–1 |
| `is_nitrogen_deficient` | boolean | `true` if nitrogen deficient |
| `is_low_yield` | boolean | `true` if yield < 20 kg |
| `is_high_priority` | boolean | `true` if deficient AND low yield |

**NDVI / NDRE value interpretation:**

| Range | Meaning |
|---|---|
| `< 0` | Water / non-vegetation (critical — map as red) |
| `0 – 0.1` | Bare / severely stressed |
| `0.1 – 0.25` | Sparse / weak canopy |
| `0.25 – 0.4` | Moderate vegetation |
| `0.4 – 0.55` | Healthy canopy |
| `> 0.55` | Very dense / very healthy |

---

### `data/summary.json` — KPI cards + chart data

```jsonc
{
  "summary": {
    "total_plants":    13997,
    "avg_ndvi":        0.5108,       // plantation average NDVI
    "avg_ndre":        0.3936,       // plantation average NDRE
    "total_yield_kg":  480373.9,
    "avg_yield_kg":    34.32,
    "n_def_pct":       27.5,         // % nitrogen deficient
    "n_def_count":     3850,
    "healthy_pct":     72.7,
    "moderate_pct":    4.0,
    "stressed_pct":    2.7,
    "critical_pct":    20.6,
    "healthy":         10176,        // NDVI ≥ 0.4
    "moderate":        560,          // NDVI 0.2–0.4
    "stressed":        378,          // NDVI 0–0.2
    "critical":        2883,         // NDVI < 0
    "ndvi_max":        1.0,
    "ndvi_min":        -0.9922,
    "ndre_max":        0.8085,
    "ndre_min":        -0.9529
  },
  "chart_data": {
    "ndvi_bins":    ["<0","0-0.1","0.1-0.2",...],   // histogram bins
    "ndvi_hist":    [2883, 181, 197, ...],           // plant counts per bin
    "ndre_bins":    ["<0","0-0.1",...],
    "ndre_hist":    [5, 43, 1518, ...],
    "stage_names":  ["Seedling","Vegetative","Flowering","Fruiting","Mature"],
    "stage_counts": [3260, 559, 763, 1446, 7969],
    "sector_stats": [ { "name":"S00", "count":3249, "ndvi_mean":0.668, ... } ],
    "health":       [10176, 560, 378, 2883],
    "health_labels":["Healthy (≥0.4)","Moderate (0.2-0.4)","Stressed (0-0.2)","Critical (<0)"]
  }
}
```

---

### `data/sectors.json` — Per-sector aggregated stats

Array of 14 objects (S00–S13):

```jsonc
[
  {
    "name":       "S00",
    "count":      3249,           // plant count
    "ndvi_mean":  0.668,
    "ndre_mean":  0.405,
    "yield_total":125223.6,       // total yield in kg
    "n_def_pct":  18.9            // % nitrogen deficient
  },
  ...
]
```

> ⚠️ **Note on sector S02**: NDVI mean = −0.185, 73.4% nitrogen deficiency, avg yield only 13 kg. This is a known anomalous zone (possibly waterlogging or disease). Flag it prominently on the map.

---

### `data/insights.json` — Full ML insights

```jsonc
{
  "total_plants_analyzed": 13997,
  "growth_stage_distribution": { "Mature":7969, "Seedling":3260, ... },
  "yield_analysis": {
    "total_predicted_yield_kg": 480373.9,
    "average_yield_per_plant_kg": 34.32,
    "median_yield_kg": 43.48,
    "low_yield_plants": 3602,
    "high_yield_plants": 57
  },
  "nitrogen_analysis": {
    "deficient_plants": 3850,
    "deficient_percentage": 27.51,
    "healthy_plants": 10147
  },
  "critical_attention_needed": 3514,
  "sector_performance": { "S00": { ... }, ... },
  "recommendations": [
    { "priority":"HIGH",   "action":"Apply nitrogen fertilizer",  "affected_plants":3850 },
    { "priority":"MEDIUM", "action":"Investigate low-yield areas","affected_plants":3602 }
  ]
}
```

---

## API Reference (`plant_health_api.js`)

```js
const api = new PlantHealthAPI('./data');
await api.init();

// ── Plants ─────────────────────────────────────────────────
api.getAllPlants()                    // → Array[13997]
api.getPlantById('PLT-S00-R001-...')  // → plant object | null
api.getBySector('S02')               // → Array of plants in sector S02
api.getByPosition(1, 3)              // → plants at row 1, col 3
api.getByGrowthStage('Flowering')    // → Array of flowering plants
api.getDeficientPlants()             // → nitrogen-deficient plants
api.getLowYieldPlants()              // → yield < 20 kg
api.getHighPriorityPlants()          // → deficient AND low yield
api.filterPlants(p => p.ndvi > 0.7 && p.sector === 'S00')
api.sortBy('yield_kg', 'desc')       // → sorted copy

// ── Summary / KPIs ─────────────────────────────────────────
api.getSummary()                     // → KPI object
api.getChartData()                   // → bins, counts, sector bars

// ── Insights ───────────────────────────────────────────────
api.getInsights()                    // → full ML report
api.getYieldAnalysis()               // → yield sub-object
api.getNitrogenAnalysis()            // → nitrogen sub-object
api.getGrowthStageDistribution()     // → { Mature:N, Seedling:N, ... }
api.getRecommendations()             // → [ { priority, action, ... } ]
api.getSectorPerformance()           // → { S00: {...}, S01: {...}, ... }

// ── Sectors ────────────────────────────────────────────────
api.getAllSectors()                   // → Array[14]
api.getSectorStats('S02')            // → sector object
api.getBestSector()                  // → highest avg NDVI sector
api.getWorstSector()                 // → lowest avg NDVI sector

// ── Static Color Helpers ───────────────────────────────────
PlantHealthAPI.ndviColor(0.65)           // → '#22c55e'
PlantHealthAPI.ndreColor(0.30)           // → '#84cc16'
PlantHealthAPI.growthStageColor('Mature')// → '#22c55e'
PlantHealthAPI.yieldColor(12.5)          // → '#ef4444'
PlantHealthAPI.nitrogenColor('Deficient')// → '#ef4444'
```

---

## Map Integration (Leaflet.js)

```js
const api = new PlantHealthAPI('./data');
await api.init();

// Colour markers by NDVI value
api.getAllPlants().forEach(plant => {
  L.circleMarker([plant.lat, plant.lon], {
    radius:      5,
    fillColor:   PlantHealthAPI.ndviColor(plant.ndvi),
    fillOpacity: 0.85,
    color:       '#fff',
    weight:      0.5,
  })
  .bindPopup(`
    <b>${plant.plant_id}</b><br>
    Sector: ${plant.sector} — Row ${plant.row_index}, Col ${plant.col_index}<br>
    NDVI: ${plant.ndvi?.toFixed(3)} (${plant.ndvi_category})<br>
    NDRE: ${plant.ndre?.toFixed(3)} (${plant.ndre_category})<br>
    Growth Stage: ${plant.growth_stage}<br>
    Yield: ${plant.yield_kg} kg<br>
    Nitrogen: ${plant.nitrogen_status}
  `)
  .addTo(map);
});

// Highlight high-priority plants
api.getHighPriorityPlants().forEach(plant => {
  L.circleMarker([plant.lat, plant.lon], {
    radius: 8, fillColor: '#ef4444', fillOpacity: 0.95,
    color: '#fff', weight: 2,
  })
  .bindPopup(`<b>⚠️ HIGH PRIORITY</b><br>${plant.plant_id}`)
  .addTo(map);
});
```

---

## KPI Cards Example

```js
const s = api.getSummary();
document.getElementById('kpi-total').textContent   = s.total_plants.toLocaleString();
document.getElementById('kpi-ndvi').textContent    = s.avg_ndvi.toFixed(3);
document.getElementById('kpi-ndre').textContent    = s.avg_ndre.toFixed(3);
document.getElementById('kpi-yield').textContent   = s.total_yield_kg.toLocaleString() + ' kg';
document.getElementById('kpi-def-pct').textContent = s.n_def_pct + '%';
document.getElementById('kpi-critical').textContent= s.critical.toLocaleString() + ' plants';
```

---

## Coordinate Reference

- All coordinates in `plants.json` are **WGS84 (EPSG:4326)** — standard `lat`/`lon` suitable for Leaflet, Mapbox, Google Maps, etc.
- The plantation is located approximately at:
  - **Lat:** ~2.936° N
  - **Lon:** ~101.744° E
  - **Country:** Malaysia

---

## How This Data Was Produced

1. YOLOv8 detected ~14,500 oil palm plants from drone imagery
2. Each plant was matched to NDVI (from CSV grid) and NDRE (from GeoTIFF raster)
3. XGBoost + LightGBM + Random Forest models predicted growth stage, yield, and nitrogen status
4. This `frontend/` folder was exported from the pipeline outputs

**Pipeline contact:** nithints21 (GitHub)
