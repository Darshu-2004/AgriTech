import os
import json
import shutil
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from PIL import Image

# Define workspace directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)

csv_path = os.path.join(workspace_dir, 'plants_with_predictions_health.csv')
ortho_path = os.path.join(workspace_dir, 'dataset1', 'ggs-orthophoto (2).tif')
web_img_path = os.path.join(workspace_dir, 'orthomosaic_web.png')
html_out_path = os.path.join(workspace_dir, 'plants_health_map.html')
artifact_dir = r'C:\Users\dabbu\.gemini\antigravity\brain\9ee7a796-f500-422b-9f2c-cb0f2a00e38f'

MAX_DIM = 8000

print("=== PIPELINE STEP 2: INTERACTIVE HTML VISUALIZATION MAP ===")
if not os.path.exists(csv_path):
    print(f"Error: Database file {csv_path} not found. Please run Step 3 first.")
    exit(1)

if not os.path.exists(ortho_path):
    print(f"Error: Background RGB map not found at: {ortho_path}")
    exit(1)

# 1. Downsample the true-color orthomosaic to a web-friendly PNG image
print("Generating downsampled web-friendly background image...")
Image.MAX_IMAGE_PIXELS = None
with rasterio.open(ortho_path) as src:
    scale = min(1.0, MAX_DIM / max(src.width, src.height))
    out_w, out_h = int(src.width * scale), int(src.height * scale)
    data = src.read(
        out_shape=(src.count, out_h, out_w),
        resampling=Resampling.bilinear,
    )
    b = src.bounds  # native UTM bounding box
    bounds = {"left": b.left, "right": b.right, "bottom": b.bottom, "top": b.top}

rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
if data.shape[0] >= 4:
    alpha = data[3].astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
else:
    img = Image.fromarray(rgb, "RGB")

img.save(web_img_path, "PNG", optimize=True)
size_mb = os.path.getsize(web_img_path) / 1e6
print(f"Web background image generated: {out_w}x{out_h} ({size_mb:.1f} MB) -> {web_img_path}")

# 2. Load and prepare plant points
print("Loading crop health database...")
df = pd.read_csv(csv_path)
df = df.dropna(subset=['geo_x', 'geo_y'])

# Keep columns needed for Leaflet popups and color rendering
cols = [
    'plant_id', 'sector_label', 'geo_x', 'geo_y',
    'ndvi', 'ndre', 'osavi',
    'predicted_growth_stage',
    'health_score', 'health_status', 'health_color'
]
cols = [c for c in cols if c in df.columns]
sub = df[cols].copy()

# Rename coordinate columns for the JS template
sub = sub.rename(columns={'geo_x': 'x', 'geo_y': 'y'})
sub['x'] = sub['x'].round(3)
sub['y'] = sub['y'].round(3)

points = sub.where(pd.notna(sub), None).to_dict('records')

# 3. Render HTML template
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Crop Health Map Viewer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,Arial,sans-serif}
  #map{position:absolute;top:0;left:0;right:0;bottom:0}
  .panel{position:absolute;top:12px;right:12px;z-index:1000;background:#fff;
    padding:12px 14px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.25);
    font-size:13px;min-width:210px}
  .panel h3{margin:0 0 8px;font-size:14px}
  .panel label{display:block;margin:6px 0 2px;font-weight:600}
  .stat{color:#444;font-size:12px;margin-top:8px;line-height:1.5}
  .legend{position:absolute;bottom:18px;left:12px;z-index:1000;background:#fff;
    padding:10px 12px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.25);
    font-size:12px;line-height:1.4}
  .legend i{display:inline-block;width:14px;height:14px;margin-right:6px;
    border-radius:3px;vertical-align:-2px}
  select,input{width:100%;box-sizing:border-box}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h3>Crop Health Viewer</h3>
  <label>Color by</label>
  <select id="metric">
    <option value="health_status">Health Status</option>
    <option value="predicted_growth_stage">Growth Stage</option>
    <option value="health_score">Health Score (0-100)</option>
    <option value="ndvi">NDVI (Vigor)</option>
    <option value="ndre">NDRE (Nitrogen)</option>
  </select>
  <label>Sector</label>
  <select id="sector"><option value="ALL">All Sectors</option></select>
  <label id="minLabel">Min value <span id="minv">-1.00</span></label>
  <input type="range" id="minr" min="-1" max="1" step="0.05" value="-1"/>
  <hr style="margin:10px 0;border:none;border-top:1px solid #ddd"/>
  <label>Align X <span id="offxv">0.0</span> m</label>
  <input type="range" id="offx" min="-15" max="15" step="0.1" value="0"/>
  <label>Align Y <span id="offyv">0.0</span> m</label>
  <input type="range" id="offy" min="-15" max="15" step="0.1" value="0"/>
  <div style="font-size:11px;color:#777;margin-top:4px">Nudge markers onto the
    plant rows (corrects orthomosaic misalignment).</div>
  <div class="stat" id="stat"></div>
</div>
<div class="legend" id="legend"></div>
<script>
const BOUNDS = __BOUNDS__;          // UTM bounds {left,right,bottom,top}
const DATA = __DATA__;

const map = L.map('map', {crs: L.CRS.Simple, minZoom:-3, maxZoom:8,
                          zoomSnap:0.1, zoomDelta:0.5,
                          preferCanvas:true});
const canvasRenderer = L.canvas({padding:0.5});
const imgBounds = [[BOUNDS.bottom, BOUNDS.left],[BOUNDS.top, BOUNDS.right]];
L.imageOverlay('__IMG__', imgBounds, {opacity:1}).addTo(map);
map.fitBounds(imgBounds);

// Populating sectors dropdown
const sectors=[...new Set(DATA.map(d=>d.sector_label).filter(Boolean))].sort();
const sel=document.getElementById('sector');
sectors.forEach(s=>{const o=document.createElement('option');o.value=o.textContent=s;sel.appendChild(o);});

// Color ramps
function rampIndex(v){
  if(v===null||v===undefined||isNaN(v)) return '#888';
  const t=Math.max(0,Math.min(1,(v+1)/2));   // 0..1
  let r,g,b;
  if(t<0.5){const k=t/0.5; r=215+(255-215)*k; g=25+(255-25)*k; b=40+(191-40)*k;}
  else{const k=(t-0.5)/0.5; r=255+(0-255)*k; g=255+(104-255)*k; b=191+(55-191)*k;}
  return `rgb(${r|0},${g|0},${b|0})`;
}

const STAGE_COLORS = {
  'Seedling': '#80DEEA',
  'Vegetative': '#4DB6AC',
  'Flowering': '#FFF176',
  'Fruiting': '#FFB74D',
  'Mature': '#BA68C8',
  'NOISE': '#757575'
};

function draw(){
  const metric=document.getElementById('metric').value;
  const sector=sel.value;
  const minr=document.getElementById('minr');
  const minLabel=document.getElementById('minLabel');
  
  const isIndex = (metric==='ndvi'||metric==='ndre');
  minr.disabled = !isIndex;
  minLabel.style.opacity = isIndex?1:0.4;
  const minv=parseFloat(minr.value);
  document.getElementById('minv').textContent=minv.toFixed(2);
  
  const offX=parseFloat(document.getElementById('offx').value);
  const offY=parseFloat(document.getElementById('offy').value);
  document.getElementById('offxv').textContent=offX.toFixed(1);
  document.getElementById('offyv').textContent=offY.toFixed(1);
  
  layer.clearLayers();
  let shown=0;
  
  DATA.forEach(d=>{
    if(sector!=='ALL' && d.sector_label!==sector) return;
    
    // Determine color based on chosen metric
    let color = '#888';
    if (metric === 'health_status') {
      color = d.health_color || '#757575';
    } else if (metric === 'predicted_growth_stage') {
      color = STAGE_COLORS[d.predicted_growth_stage] || '#888';
    } else if (metric === 'health_score') {
      const s = d.health_score;
      if (Number.isFinite(s)) {
        // Red (0) -> Yellow (50) -> Green (100)
        const t = s / 100.0;
        const r = t < 0.5 ? 255 : Math.round(255 - 255 * (t - 0.5) * 2);
        const g = t < 0.5 ? Math.round(255 * t * 2) : 255;
        color = `rgb(${r},${g},0)`;
      }
    } else if (isIndex) {
      const v = d[metric];
      if (v === null || v === undefined || v < minv) return;
      color = rampIndex(v);
    }
    
    shown++;
    L.circleMarker([d.y + offY, d.x + offX],{
      renderer:canvasRenderer,
      radius:3,color:'#222',weight:.3,fillColor:color,fillOpacity:.95
    }).bindPopup(
      `<b>${d.plant_id||'Crop'}</b> &nbsp;Sector: ${d.sector_label||'-'}<br>`+
      `Growth Stage: ${d.predicted_growth_stage_name||'-'}<br>`+
      `Health Status: ${d.health_status||'-'} (${d.health_score??'-'}/100)<br>`+
      `NDVI: ${d.ndvi??'-'} &nbsp;|&nbsp; NDRE: ${d.ndre??'-'}`
    ).addTo(layer);
  });
  
  document.getElementById('stat').innerHTML=
    `Showing <b>${shown.toLocaleString()}</b> plants`;
  renderLegend(metric);
}

const layer = L.layerGroup().addTo(map);

function renderLegend(metric){
  const leg = document.getElementById('legend');
  if(metric==='health_status'){
    leg.innerHTML='<b>Health Status</b><br>'+
      '<i style="background:#2E7D32"></i>Healthy<br>'+
      '<i style="background:#FBC02D"></i>Moderate<br>'+
      '<i style="background:#FF5722"></i>Stressed<br>'+
      '<i style="background:#D32F2F"></i>Critical<br>'+
      '<i style="background:#1976D2"></i>Out of Boundary<br>'+
      '<i style="background:#757575"></i>Noise / Outlier';
  } else if(metric==='predicted_growth_stage'){
    leg.innerHTML='<b>Growth Stage</b><br>'+
      '<i style="background:#80DEEA"></i>Seedling<br>'+
      '<i style="background:#4DB6AC"></i>Vegetative<br>'+
      '<i style="background:#FFF176"></i>Flowering<br>'+
      '<i style="background:#FFB74D"></i>Fruiting<br>'+
      '<i style="background:#BA68C8"></i>Mature<br>'+
      '<i style="background:#757575"></i>Noise';
  } else if(metric==='health_score'){
    leg.innerHTML='<b>Health Score</b><br>'+
      '<i style="background:rgb(0,255,0)"></i>100 (Healthy)<br>'+
      '<i style="background:rgb(255,255,0)"></i>50 (Moderate)<br>'+
      '<i style="background:rgb(255,0,0)"></i>0 (Critical)';
  } else {
    leg.innerHTML='<b>Index value</b><br>'+
      '<i style="background:'+rampIndex(0.8)+'"></i>0.8<br>'+
      '<i style="background:'+rampIndex(0.4)+'"></i>0.4<br>'+
      '<i style="background:'+rampIndex(0.0)+'"></i>0.0<br>'+
      '<i style="background:'+rampIndex(-0.4)+'"></i>-0.4<br>'+
      '<i style="background:'+rampIndex(-0.8)+'"></i>-0.8';
  }
}

['metric','sector','minr','offx','offy'].forEach(id=>
  document.getElementById(id).addEventListener('input',draw));
draw();
</script>
</body>
</html>
"""

html_content = HTML_TEMPLATE.replace("__BOUNDS__", json.dumps(bounds)) \
                             .replace("__DATA__", json.dumps(points, separators=(',', ':'))) \
                             .replace("__IMG__", "orthomosaic_web.png")

with open(html_out_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Interactive HTML map generated: {html_out_path}")

# 4. Copy generated artifacts to output folders
if os.path.exists(artifact_dir):
    try:
        shutil.copy(web_img_path, os.path.join(artifact_dir, 'orthomosaic_web.png'))
        shutil.copy(html_out_path, os.path.join(artifact_dir, 'plants_health_map.html'))
        print("Copied interactive map files to artifact directory.")
    except Exception as e:
        print(f"Could not copy files to artifact directory: {e}")

print("Step 2 finished successfully.\n")
