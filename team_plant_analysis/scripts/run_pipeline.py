import subprocess
import os
import sys

# Define script path
script_dir = os.path.dirname(os.path.abspath(__file__))

step1_path = os.path.join(script_dir, '01_extract_indices.py')
step1b_path = os.path.join(script_dir, 'train_growth_model.py')
step3_path = os.path.join(script_dir, '03_classify_health.py')
step2_path = os.path.join(script_dir, '02_generate_preview.py')

print("=========================================================")
print("          STARTING ADVANCED CROP PIPELINE                ")
print("=========================================================\n")

# Run Step 1: Index Extraction
print("Running Step 1: Georeferenced Index Extraction & Shadow Masking...")
p1 = subprocess.run([sys.executable, step1_path], capture_output=False)
if p1.returncode != 0:
    print("\nError: Step 1 failed. Aborting pipeline.")
    sys.exit(p1.returncode)

# Run Step 1b: Train Growth Classifier Model
print("Running Step 1b: Training Growth Classifier Model...")
p1b = subprocess.run([sys.executable, step1b_path], capture_output=False)
if p1b.returncode != 0:
    print("\nError: Step 1b failed. Aborting pipeline.")
    sys.exit(p1b.returncode)

# Run Step 3: Spatial Smoothing & Health Classification
print("Running Step 3: Spatial Smoothing & Quantile Health Classification...")
p3 = subprocess.run([sys.executable, step3_path], capture_output=False)
if p3.returncode != 0:
    print("\nError: Step 3 failed. Aborting pipeline.")
    sys.exit(p3.returncode)

# Run Step 2: Visualization Health Map Generation
print("Running Step 2: Crop Health Map Generation...")
p2 = subprocess.run([sys.executable, step2_path], capture_output=False)
if p2.returncode != 0:
    print("\nError: Step 2 failed. Aborting pipeline.")
    sys.exit(p2.returncode)

# Run Step 4: PostgreSQL/PostGIS Database Export (Optional / Graceful Fail)
step4_path = os.path.join(script_dir, 'db_exporter.py')
print("Running Step 4: Exporting processed crop health data to PostgreSQL/PostGIS...")
p4 = subprocess.run([sys.executable, step4_path], capture_output=False)
if p4.returncode != 0:
    print("\n[Optional Step] Note: PostgreSQL database was not reachable. Storing output strictly in local CSV.")

print("=========================================================")
print("          PIPELINE EXECUTED SUCCESSFULLY                 ")
print("=========================================================")
print("Outputs Generated:")
print("  - Enriched Health CSV: plants_with_predictions_health.csv")
print("  - Visualization Map:   plants_health_map.png")
print("  - Postgres/PostGIS:    pineapple_crops table (if DB connected)")
print("=========================================================")
