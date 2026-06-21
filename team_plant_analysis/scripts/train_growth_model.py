import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.ensemble import RandomForestClassifier

# Define workspace directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)

csv_path = os.path.join(workspace_dir, 'plants_extracted_indices.csv')
ortho_path = os.path.join(workspace_dir, 'dataset1', 'ggs-orthophoto (2).tif')
model_path = os.path.join(script_dir, 'growth_stage_rf.pkl')

print("=== PIPELINE STEP 1B: TRAINING PHYSICAL GROWTH STAGE CLASSIFIER ===")

if not os.path.exists(csv_path):
    print(f"Error: Extracted indices database {csv_path} not found. Please run Step 1 first.")
    exit(1)

if not os.path.exists(ortho_path):
    print(f"Error: Background RGB map not found at: {ortho_path}")
    exit(1)

df = pd.read_csv(csv_path)

# Extract GSD parameters using PIL tag reader
print("Extracting Ground Sampling Distance (GSD) scales from GeoTIFF...")
try:
    img = Image.open(ortho_path)
    tags = img.tag_v2
    # ModelPixelScaleTag key is 33550
    if 33550 in tags:
        gsd_x = abs(tags[33550][0])
        gsd_y = abs(tags[33550][1])
    else:
        gsd_x, gsd_y = 0.0134577, 0.0134582  # Fallback
    img.close()
except Exception as e:
    print(f"Warning: Could not read GSD tags from GeoTIFF: {e}. Using fallbacks.")
    gsd_x, gsd_y = 0.0134577, 0.0134582

print(f"  -> GSD scale: X = {gsd_x:.6f} m/px, Y = {gsd_y:.6f} m/px")

# Convert pixels to physical meters
df['canopy_area_m2'] = df['area_px'] * gsd_x * gsd_y
df['bbox_width_m'] = df['bbox_width'] * gsd_x
df['bbox_height_m'] = df['bbox_height'] * gsd_y

# Drop rows with NaN indices for training
train_df = df.dropna(subset=['osavi'])
if len(train_df) == 0:
    train_df = df  # Fallback if all are NaN

# Define physical features (invariant to flight altitude)
feature_cols = ['canopy_area_m2', 'bbox_width_m', 'bbox_height_m', 'osavi']
X = train_df[feature_cols].values

# Bootstrap target labels using physical metric bins (in square meters)
y_labels = []
for idx, row in train_df.iterrows():
    if row['sector_label'] == 'NOISE':
        y_labels.append('NOISE')
        continue
        
    area_m2 = row['canopy_area_m2']
    if area_m2 < 0.0815:
        y_labels.append('Seedling')
    elif area_m2 < 0.1177:
        y_labels.append('Vegetative')
    elif area_m2 < 0.1449:
        y_labels.append('Flowering')
    elif area_m2 < 0.1811:
        y_labels.append('Fruiting')
    else:
        y_labels.append('Mature')
y = np.array(y_labels)

print("Training a Random Forest Classifier on physical metric features...")
model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model.fit(X, y)

# Serialize model along with scale factors for inference consistency
model_data = {
    'model': model,
    'model_type': 'sklearn_rf',
    'feature_cols': feature_cols,
    'gsd_x': gsd_x,
    'gsd_y': gsd_y
}

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Model successfully saved to: {model_path}")
print("Step 1b finished successfully.\n")
