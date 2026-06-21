import pandas as pd
import numpy as np
import os
import pickle
from scipy.spatial import KDTree
from sklearn.ensemble import IsolationForest

# Define workspace directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)

csv_path = os.path.join(workspace_dir, 'plants_extracted_indices.csv')
model_path = os.path.join(script_dir, 'growth_stage_rf.pkl')
output_csv_path = os.path.join(workspace_dir, 'plants_with_predictions_health.csv')

print("=== PIPELINE STEP 3: CONTEXT-AWARE ML HEALTH & SCORING SYSTEM ===")

if not os.path.exists(csv_path):
    print(f"Error: Extracted indices database {csv_path} not found.")
    exit(1)

if not os.path.exists(model_path):
    print(f"Error: Growth stage model {model_path} not found. Please run Step 1b first.")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} records.")

# 1. Growth Stage Prediction using the serialized Random Forest model
print("Loading growth stage classifier model...")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_cols = model_data['feature_cols']
gsd_x = model_data.get('gsd_x', 0.0134577)
gsd_y = model_data.get('gsd_y', 0.0134582)

print(f"Applying physical scaling with GSD: X = {gsd_x:.6f} m, Y = {gsd_y:.6f} m")
df['canopy_area_m2'] = df['area_px'] * gsd_x * gsd_y
df['bbox_width_m'] = df['bbox_width'] * gsd_x
df['bbox_height_m'] = df['bbox_height'] * gsd_y

print("Predicting growth stages for all crops...")
valid_mask = df['sector_label'] != 'NOISE'
df['predicted_growth_stage_name'] = 'NOISE'

if valid_mask.any():
    X = df.loc[valid_mask, feature_cols].fillna(0.0).values
    df.loc[valid_mask, 'predicted_growth_stage_name'] = model.predict(X)

print("Growth stage counts:")
print(df['predicted_growth_stage_name'].value_counts())

# 2. Spatial Neighborhood Smoothing using KD-Tree
print("\nApplying KD-Tree Spatial Neighborhood Smoothing (K=8 neighbors)...")
df['osavi_smoothed'] = df['osavi']

if valid_mask.any():
    coords = df.loc[valid_mask, ['geo_x', 'geo_y']].values
    indices = df.loc[valid_mask].index.values
    
    tree = KDTree(coords)
    smoothed_vals = []
    for i, idx in enumerate(indices):
        orig_val = df.at[idx, 'osavi']
        if np.isnan(orig_val):
            smoothed_vals.append(np.nan)
            continue
            
        dists, neighbors = tree.query(coords[i], k=9)
        neighbor_indices = indices[neighbors]
        neighbor_osavis = df.loc[neighbor_indices, 'osavi'].dropna().values
        
        if len(neighbor_osavis) > 1:
            mean_nbr = np.mean(neighbor_osavis)
            smoothed_val = 0.7 * orig_val + 0.3 * mean_nbr
            smoothed_vals.append(smoothed_val)
        else:
            smoothed_vals.append(orig_val)
            
    df.loc[valid_mask, 'osavi_smoothed'] = smoothed_vals
print("Spatial smoothing completed.")

# 3. Unsupervised Anomaly Detection & Continuous Health Scoring
print("\nFitting Isolation Forest models per growth stage and computing Health Scores...")
df['health_score'] = np.nan
df['health_status'] = 'Moderate'
df['health_color'] = '#FBC02D'

stages = ['Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Mature']

for stage in stages:
    stage_mask = (df['predicted_growth_stage_name'] == stage) & valid_mask
    sub_df = df[stage_mask]
    
    if len(sub_df) < 15:
        print(f"Skipping Isolation Forest for stage '{stage}' (too few crops: {len(sub_df)})")
        for idx in sub_df.index:
            osa = df.at[idx, 'osavi']
            if np.isnan(osa):
                df.at[idx, 'health_status'] = 'Out of Boundary'
                df.at[idx, 'health_color'] = '#1976D2'
            else:
                v = np.clip((osa - 0.2) / 0.3, 0.0, 1.0) * 100
                df.at[idx, 'health_score'] = round(v, 1)
                if v < 15:
                    df.at[idx, 'health_status'], df.at[idx, 'health_color'] = 'Critical', '#D32F2F'
                elif v < 40:
                    df.at[idx, 'health_status'], df.at[idx, 'health_color'] = 'Stressed', '#FF5722'
                elif v < 70:
                    df.at[idx, 'health_status'], df.at[idx, 'health_color'] = 'Moderate', '#FBC02D'
                else:
                    df.at[idx, 'health_status'], df.at[idx, 'health_color'] = 'Healthy', '#2E7D32'
        continue
        
    print(f"Fitting Isolation Forest for stage '{stage}' ({len(sub_df)} crops)...")
    X_stage = sub_df[['osavi_smoothed', 'ndvi', 'ndre']].copy()
    
    for col in X_stage.columns:
        median_val = X_stage[col].median()
        if np.isnan(median_val):
            median_val = 0.5
        X_stage[col] = X_stage[col].fillna(median_val)
        
    model_if = IsolationForest(contamination=0.12, random_state=42)
    model_if.fit(X_stage.values)
    
    preds = model_if.predict(X_stage.values)
    scores = model_if.decision_function(X_stage.values)
    
    median_score = np.median(scores)
    std_score = np.std(scores) if np.std(scores) > 0.01 else 0.05
    score_threshold = median_score - 1.5 * std_score
    
    floor_healthy = 0.40 if stage == 'Seedling' else 0.50
    floor_unhealthy = 0.20 if stage == 'Seedling' else 0.25
    stage_mean_osavi = sub_df['osavi_smoothed'].mean()
    if np.isnan(stage_mean_osavi):
        stage_mean_osavi = 0.45
        
    for idx, pred, score in zip(sub_df.index, preds, scores):
        osa = df.at[idx, 'osavi']
        ndv = df.at[idx, 'ndvi']
        ndr = df.at[idx, 'ndre']
        
        if np.isnan(osa):
            df.at[idx, 'health_status'] = 'Out of Boundary'
            df.at[idx, 'health_color'] = '#1976D2'
            continue
            
        # 1. Base Vigor Score
        v_osa = np.clip((osa - floor_unhealthy) / (floor_healthy - floor_unhealthy), 0.0, 1.0) * 100
        v_ndv = np.clip((ndv - 0.15) / 0.45, 0.0, 1.0) * 100 if not np.isnan(ndv) else 50.0
        v_ndr = np.clip((ndr - 0.10) / 0.30, 0.0, 1.0) * 100 if not np.isnan(ndr) else 50.0
        vigor_score = 0.6 * v_osa + 0.2 * v_ndv + 0.2 * v_ndr
        
        # 2. Normalized Anomaly Score
        min_s = median_score - 2 * std_score
        max_s = median_score + std_score
        norm_score = np.clip((score - min_s) / (max_s - min_s), 0.0, 1.0) * 100
        
        # Blend
        final_score = 0.7 * vigor_score + 0.3 * norm_score
        
        # Directional Outlier Vigor Capping & Floors
        is_outlier = (score < score_threshold)
        is_really_stressed = is_outlier and (osa < stage_mean_osavi)
        
        if osa < floor_unhealthy:
            final_score = min(final_score, 14.0)  # Critical
        elif is_really_stressed:
            if osa < (floor_unhealthy + 0.05):
                final_score = min(final_score, 14.0)  # Critical
            else:
                final_score = np.clip(final_score, 15.0, 39.0)  # Stressed
        elif is_outlier and osa >= stage_mean_osavi:
            final_score = max(final_score, 70.0)  # Healthy Outlier (Hyper-grower)
            
        # Classify based on score
        if final_score < 15:
            status, color = 'Critical', '#D32F2F'
        elif final_score < 40:
            status, color = 'Stressed', '#FF5722'
        elif final_score < 70:
            status, color = 'Moderate', '#FBC02D'
        else:
            status, color = 'Healthy', '#2E7D32'
            
        df.at[idx, 'health_score'] = round(final_score, 1)
        df.at[idx, 'health_status'] = status
        df.at[idx, 'health_color'] = color

# Handle the NOISE sector
noise_mask = df['predicted_growth_stage_name'] == 'NOISE'
df.loc[noise_mask, 'health_status'] = 'NOISE'
df.loc[noise_mask, 'health_color'] = '#757575'
df.loc[noise_mask, 'health_score'] = np.nan

print("\n=== Overall Health Status Distribution (ML Score Engine) ===")
print(df['health_status'].value_counts())

# Prune columns to essential ones for output
columns_to_keep = [
    'plant_id',
    'instance_id',
    'sector_id',
    'sector_label',
    'row_index',
    'col_index',
    'pixel_x',
    'pixel_y',
    'geo_x',
    'geo_y',
    'area_px',
    'canopy_area_m2',
    'bbox_x',
    'bbox_y',
    'bbox_width',
    'bbox_height',
    'mask_file',
    'predicted_growth_stage_name',
    'health_score',
    'health_status',
    'health_color',
    'osavi',
    'ndvi',
    'ndre',
    'osavi_smoothed'
]
df = df[[col for col in columns_to_keep if col in df.columns]]

# Save final output
df.to_csv(output_csv_path, index=False)
print(f"\nFinal health database successfully saved to: {output_csv_path}")
print("Step 3 finished successfully.\n")
