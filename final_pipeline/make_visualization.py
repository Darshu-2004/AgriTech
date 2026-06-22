"""
Build a self-contained HTML map of plant-health indices over the orthomosaic.

Produces (in output/):
  - orthomosaic_web.jpg   downsampled web image of orthomosaic.tif
  - plant_health_map.html interactive Leaflet map (points colored by index)

The plant data is embedded directly in the HTML as JSON, so the page works
when opened from disk (file://) with no server.

    python make_visualization.py
"""

import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from PIL import Image

import config

WEB_IMG = config.OUTPUT_DIR / "orthomosaic_web.png"
HTML_OUT = config.OUTPUT_DIR / "plant_health_map.html"
ORTHO = config.ORTHO_TIF
# Longest side of the web basemap PNG. The source ortho is ~328 MP; a full-res
# PNG would be unusable in a browser. Alignment is unaffected by downscaling
# because the overlay is placed by its UTM bounds, not by pixel count.
MAX_DIM = 8000


def build_web_image():
    """
    Render orthomosaic.tif to a high-res PNG and return its native (UTM)
    extent. The map uses L.CRS.Simple so the image and the plant markers
    share the same projected coordinate space and align exactly.
    """
    Image.MAX_IMAGE_PIXELS = None
    with rasterio.open(str(ORTHO)) as src:
        if MAX_DIM and MAX_DIM > 0:
            scale = min(1.0, MAX_DIM / max(src.width, src.height))
        else:
            scale = 1.0  # full native resolution
        out_w, out_h = int(src.width * scale), int(src.height * scale)
        data = src.read(
            out_shape=(src.count, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        b = src.bounds  # native CRS (UTM 32647) extent

    rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
    if data.shape[0] >= 4:
        # Keep transparency so off-image areas stay clear (RGBA PNG).
        alpha = data[3].astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    else:
        img = Image.fromarray(rgb, "RGB")

    img.save(WEB_IMG, "PNG", optimize=True)
    size_mb = WEB_IMG.stat().st_size / 1e6
    print(f"[viz] web image {out_w}x{out_h} PNG ({size_mb:.1f} MB) -> {WEB_IMG.name}")
    return {"left": b.left, "right": b.right, "bottom": b.bottom, "top": b.top}


def build_points():
    df = pd.read_csv(config.OUTPUT_CSV)
    df = df.dropna(subset=["x", "y"])
    # Only plot plants that actually fall on the orthomosaic basemap.
    if "in_orthomosaic" in df.columns:
        before = len(df)
        df = df[df["in_orthomosaic"]]
        print(f"[viz] dropped {before - len(df)} plants outside orthomosaic extent")
    cols = ["plant_id", "sector_label", "x", "y",
            "ndvi", "ndre", "biomass_score", "nitrogen_score", "index_source"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    # round UTM coords to mm to keep the embedded JSON compact
    sub["x"] = sub["x"].round(3)
    sub["y"] = sub["y"].round(3)
    recs = sub.where(pd.notna(sub), None).to_dict("records")
    return recs


def render_html(bounds, points):
    data_json = json.dumps(points, separators=(",", ":"))
    bounds_json = json.dumps(bounds)
    html = HTML_TEMPLATE.replace("__BOUNDS__", bounds_json) \
                        .replace("__DATA__", data_json) \
                        .replace("__IMG__", WEB_IMG.name)
    HTML_OUT.write_text(html)
    print(f"[viz] {len(points):,} points -> {HTML_OUT}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Plant Health Map — NDVI / NDRE</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,Arial,sans-serif}
  #map{position:absolute;top:0;left:0;right:0;bottom:0}
  .panel{position:absolute;top:12px;right:12px;z-index:1000;background:#fff;
    padding:12px 14px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.25);
    font-size:13px;min-width:190px}
  .panel h3{margin:0 0 8px;font-size:14px}
  .panel label{display:block;margin:6px 0 2px;font-weight:600}
  .stat{color:#444;font-size:12px;margin-top:8px;line-height:1.5}
  .legend{position:absolute;bottom:18px;left:12px;z-index:1000;background:#fff;
    padding:10px 12px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.25);
    font-size:12px}
  .legend i{display:inline-block;width:14px;height:14px;margin-right:6px;
    border-radius:3px;vertical-align:-2px}
  select,input{width:100%;box-sizing:border-box}
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <h3>Plant Health Map</h3>
  <label>Color by</label>
  <select id="metric">
    <option value="biomass_score">Biomass (NDVI)</option>
    <option value="nitrogen_score">Nitrogen (NDRE)</option>
    <option value="ndvi">NDVI (raw)</option>
    <option value="ndre">NDRE (raw)</option>
  </select>
  <label>Sector</label>
  <select id="sector"><option value="ALL">All</option></select>
  <label id="minLabel">Min value <span id="minv">-1.00</span></label>
  <input type="range" id="minr" min="-1" max="1" step="0.05" value="-1"/>
  <hr style="margin:10px 0;border:none;border-top:1px solid #ddd"/>
  <label>Align X <span id="offxv">0.0</span> m</label>
  <input type="range" id="offx" min="-15" max="15" step="0.1" value="0"/>
  <label>Align Y <span id="offyv">0.0</span> m</label>
  <input type="range" id="offy" min="-15" max="15" step="0.1" value="0"/>
  <div style="font-size:11px;color:#777;margin-top:2px">Nudge markers onto the
    plant rows (the detections come from a separately-georeferenced ortho).</div>
  <div class="stat" id="stat"></div>
</div>
<div class="legend" id="legend"></div>
<script>
const BOUNDS = __BOUNDS__;          // native UTM extent {left,right,bottom,top}
const DATA = __DATA__;

// Use a flat (Simple) CRS so the image and the markers live in the SAME
// projected UTM space -> pixel-perfect alignment, no reprojection drift.
// Leaflet Simple coords are [y, x].
const map = L.map('map', {crs: L.CRS.Simple, minZoom:-3, maxZoom:8,
                          zoomSnap:0.1, zoomDelta:0.5,
                          preferCanvas:true});
// One shared canvas for all markers -> fast with thousands of points.
const canvasRenderer = L.canvas({padding:0.5});
const imgBounds = [[BOUNDS.bottom, BOUNDS.left],[BOUNDS.top, BOUNDS.right]];
L.imageOverlay('__IMG__', imgBounds, {opacity:1}).addTo(map);
map.fitBounds(imgBounds);

// color ramp: red (low) -> yellow -> green (high), domain [-1,1]
function ramp(v){
  if(v===null||v===undefined||isNaN(v)) return '#888';
  const t=Math.max(0,Math.min(1,(v+1)/2));   // 0..1
  let r,g,b;
  if(t<0.5){const k=t/0.5; r=215+(255-215)*k; g=25+(255-25)*k; b=40+(191-40)*k;}
  else{const k=(t-0.5)/0.5; r=255+(0-255)*k; g=255+(104-255)*k; b=191+(55-191)*k;}
  return `rgb(${r|0},${g|0},${b|0})`;
}

const layer = L.layerGroup().addTo(map);
const sectors=[...new Set(DATA.map(d=>d.sector_label).filter(Boolean))].sort();
const sel=document.getElementById('sector');
sectors.forEach(s=>{const o=document.createElement('option');o.value=o.textContent=s;sel.appendChild(o);});

// 0-100 score ramps. Biomass = brown->green (canopy density);
// Nitrogen = purple->yellow (viridis, distinct hue).
function _lerp(stops,t){const n=stops.length-1;const i=Math.min(n-1,Math.floor(t*n));
  const f=t*n-i,a=stops[i],b=stops[i+1];
  return `rgb(${(a[0]+(b[0]-a[0])*f)|0},${(a[1]+(b[1]-a[1])*f)|0},${(a[2]+(b[2]-a[2])*f)|0})`;}
const BIOMASS_STOPS=[[140,81,10],[216,179,101],[247,247,191],[127,188,65],[26,120,55]];
const NITRO_STOPS=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function rampScore(v,kind){
  if(!Number.isFinite(v)) return '#888';
  const t=Math.max(0,Math.min(1,v/100));
  return _lerp(kind==='nitrogen_score'?NITRO_STOPS:BIOMASS_STOPS,t);
}
const SCORE_TITLE={biomass_score:'Biomass score',nitrogen_score:'Nitrogen score'};

function draw(){
  const metric=document.getElementById('metric').value;
  const sector=sel.value;
  const minr=document.getElementById('minr');
  const minLabel=document.getElementById('minLabel');
  const score=(metric==='biomass_score'||metric==='nitrogen_score');
  const isIndex=(metric==='ndvi'||metric==='ndre');
  minr.disabled=!isIndex;
  minLabel.style.opacity=isIndex?1:0.4;
  const minv=parseFloat(minr.value);
  document.getElementById('minv').textContent=minv.toFixed(2);
  const offX=parseFloat(document.getElementById('offx').value);
  const offY=parseFloat(document.getElementById('offy').value);
  document.getElementById('offxv').textContent=offX.toFixed(1);
  document.getElementById('offyv').textContent=offY.toFixed(1);
  layer.clearLayers();
  let shown=0, scored=0, sum=0;
  DATA.forEach(d=>{
    if(sector!=='ALL' && d.sector_label!==sector) return;
    let color;
    if(score){
      const v=d[metric];
      if(!Number.isFinite(v)) return;
      color=rampScore(v,metric); scored++; sum+=v;
    }else{
      const v=d[metric];
      if(!Number.isFinite(v)||v<minv) return;
      color=ramp(v); scored++; sum+=v;
    }
    shown++;
    L.circleMarker([d.y + offY, d.x + offX],{
      renderer:canvasRenderer,
      radius:3,color:'#222',weight:.3,fillColor:color,fillOpacity:.95
    }).bindPopup(
      `<b>${d.plant_id||''}</b> &nbsp;Sector: ${d.sector_label||'-'}<br>`+
      `Biomass (NDVI ${d.ndvi??'-'}): ${d.biomass_score??'-'}/100<br>`+
      `Nitrogen (NDRE ${d.ndre??'-'}): ${d.nitrogen_score??'-'}/100`+
      (d.index_source==='predicted'?'<br><i>NDVI/NDRE predicted (XGBoost)</i>':'')
    ).addTo(layer);
  });
  const mean = scored ? (sum/scored) : null;
  let label, dp;
  if(score){label=SCORE_TITLE[metric].toLowerCase(); dp=1;}
  else {label=metric.toUpperCase(); dp=3;}
  document.getElementById('stat').innerHTML=
    `Showing <b>${shown.toLocaleString()}</b> plants<br>`+
    `Mean ${label}: <b>${mean!==null?mean.toFixed(dp):'-'}</b>`;
  renderLegend(metric);
}

const legend=document.getElementById('legend');
function renderLegend(metric){
  if(metric==='biomass_score'||metric==='nitrogen_score'){
    const sub=metric==='biomass_score'?'leaf area / canopy density':
                                       'chlorophyll / N uptake';
    legend.innerHTML=`<b>${SCORE_TITLE[metric]}</b><br>`+
      `<span style="font-size:11px;color:#666">${sub}</span><br>`+
      [['100 high',100],['75',75],['50',50],['25',25],['0 low',0]]
      .map(([t,v])=>`<i style="background:${rampScore(v,metric)}"></i>${t}`).join('<br>');
  }else{
    legend.innerHTML='<b>Index value</b><br>'+
      [['0.8',0.8],['0.4',0.4],['0.0',0.0],['-0.4',-0.4],['-0.8',-0.8]]
      .map(([t,v])=>`<i style="background:${ramp(v)}"></i>${t}`).join('<br>')+
      '<br><i style="background:#888"></i>no data';
  }
}

['metric','sector','minr','offx','offy'].forEach(id=>
  document.getElementById(id).addEventListener('input',draw));
draw();
</script>
</body>
</html>
"""


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bounds = build_web_image()
    points = build_points()
    render_html(bounds, points)
    print(f"\nOpen: {HTML_OUT}")


if __name__ == "__main__":
    main()
