import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import shutil

# Define workspace directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)

csv_path = os.path.join(workspace_dir, 'plants_with_predictions_health.csv')
ortho_path = os.path.join(workspace_dir, 'dataset1', 'ggs-orthophoto (2).tif')
output_img_path = os.path.join(workspace_dir, 'plants_health_map.png')
artifact_dir = r'C:\Users\dabbu\.gemini\antigravity\brain\9ee7a796-f500-422b-9f2c-cb0f2a00e38f'

def get_georef_params(tif_path):
    origin_x, origin_y = 805021.427, 325106.241
    dx, dy = 0.0134577, 0.0134582
    try:
        img = Image.open(tif_path)
        tags = img.tag_v2
        # ModelPixelScaleTag is 33550, ModelTiepointTag is 33922
        if 33550 in tags and 33922 in tags:
            ps = tags[33550]
            tp = tags[33922]
            origin_x, origin_y = tp[3], tp[4]
            dx, dy = ps[0], ps[1]
        img.close()
    except Exception as e:
        print(f"Warning: Could not read GeoTIFF tags from background map: {e}. Using fallback parameters.")
    return origin_x, origin_y, dx, dy

print("=== PIPELINE STEP 2: PREVIEW GENERATION (FOUR-TIER SCALE) ===")
if not os.path.exists(csv_path):
    print(f"Error: Database file {csv_path} not found. Please run Step 3 first.")
    exit(1)

if not os.path.exists(ortho_path):
    print(f"Error: Background RGB map not found at: {ortho_path}")
    exit(1)

df = pd.read_csv(csv_path)

print("Loading background orthomosaic map for visualization...")
img = Image.open(ortho_path)
orig_w, orig_h = img.size
scale_factor = 0.1
new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
img_arr = np.array(img_resized)

print("Plotting visualization map...")
fig, ax = plt.subplots(figsize=(10, 15), dpi=300)
ax.imshow(img_arr, origin='upper')

# Project georeferenced coordinates to background image pixels dynamically
orig_x_bg, orig_y_bg, dx_bg, dy_bg = get_georef_params(ortho_path)
px = ((df['geo_x'].values - orig_x_bg) / dx_bg) * scale_factor
py = ((orig_y_bg - df['geo_y'].values) / dy_bg) * scale_factor

# Use the dynamic health colors assigned in Step 3
colors = df['health_color'].values

ax.scatter(px, py, c=colors, s=0.8, alpha=0.7, edgecolors='none')
ax.set_title("Pineapple Crop Health Map - ML Anomaly Detection & Scoring", fontsize=14, pad=15, fontweight='bold')
ax.axis('off')

# Custom Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E7D32', label='Healthy (Score 70-100)\nHigh vigor, normal stage canopy performance'),
    Patch(facecolor='#FBC02D', label='Moderate (Score 40-70)\nExpected standard stage canopy performance'),
    Patch(facecolor='#FF5722', label='Stressed (Score 15-40)\nMild anomaly or low vegetation indices'),
    Patch(facecolor='#D32F2F', label='Critical (Score 0-15)\nSevere anomaly or absolute biological floor violation'),
    Patch(facecolor='#1976D2', label='Out of Boundary\nLocated outside multispectral sensor coverage'),
    Patch(facecolor='#757575', label='Noise / Outliers\nSegmented weeds or edge defects')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=9)

plt.tight_layout()

# Save image
plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f"Saved visualization map to: {output_img_path}")

# Copy to artifact directory if it exists
if os.path.exists(artifact_dir):
    try:
        dest = os.path.join(artifact_dir, 'plants_health_map.png')
        shutil.copy(output_img_path, dest)
        print("Copied map to artifact directory.")
    except Exception as e:
        print(f"Could not copy to artifact directory: {e}")

print("Step 2 finished successfully.\n")
