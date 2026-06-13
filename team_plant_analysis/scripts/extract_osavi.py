import json
import math
import numpy as np
import rasterio

# ==================================================
# INPUT FILES
# ==================================================

PLANTS_GEOJSON = "plants.geojson"
OSAVI_RASTER = "OSAVI.tif"

OUTPUT_JSON = "plants_osavi.json"

# ==================================================
# HEALTH THRESHOLDS
# ==================================================

HEALTHY_THRESHOLD = 0.25
MODERATE_THRESHOLD = 0.10

HEALTH_COLORS = {
    "healthy": "#2ECC71",   # Green
    "moderate": "#F39C12",  # Amber
    "diseased": "#E74C3C"   # Red
}

# ==================================================
# CLASSIFICATION FUNCTION
# ==================================================

def classify(osavi):

    if osavi >= HEALTHY_THRESHOLD:
        return "healthy", HEALTH_COLORS["healthy"]

    elif osavi >= MODERATE_THRESHOLD:
        return "moderate", HEALTH_COLORS["moderate"]

    else:
        return "diseased", HEALTH_COLORS["diseased"]


# ==================================================
# LOAD PLANTS
# ==================================================

print("Loading plants.geojson...")

with open(PLANTS_GEOJSON, "r") as f:
    geojson = json.load(f)

features = geojson["features"]

print(f"Plants found: {len(features)}")

# ==================================================
# PROCESS OSAVI
# ==================================================

results = []

healthy_count = 0
moderate_count = 0
diseased_count = 0

outside_raster = 0
nan_pixels = 0

with rasterio.open(OSAVI_RASTER) as src:

    band = src.read(1)

    left, bottom, right, top = src.bounds

    print("\nRaster bounds:")
    print(src.bounds)

    for feature in features:

        plant_id = feature["properties"]["plant_id"]

        x, y = feature["geometry"]["coordinates"]

        # ------------------------------------------
        # Skip plants outside raster
        # ------------------------------------------

        if not (left <= x <= right and bottom <= y <= top):
            outside_raster += 1
            continue

        row, col = src.index(x, y)

        osavi = float(band[row, col])

        # ------------------------------------------
        # Skip NaN pixels
        # ------------------------------------------

        if np.isnan(osavi):
            nan_pixels += 1
            continue

        health_status, health_color = classify(osavi)

        if health_status == "healthy":
            healthy_count += 1

        elif health_status == "moderate":
            moderate_count += 1

        else:
            diseased_count += 1

        results.append({
            "plant_id": plant_id,
            "x": x,
            "y": y,
            "osavi": round(osavi, 6),
            "health_status": health_status,
            "health_color": health_color
        })

# ==================================================
# SAVE OUTPUT
# ==================================================

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

# ==================================================
# SUMMARY
# ==================================================

print("\n==============================")
print("OSAVI EXTRACTION COMPLETE")
print("==============================")

print(f"Total plants           : {len(features)}")
print(f"Outside raster         : {outside_raster}")
print(f"NaN pixels             : {nan_pixels}")
print(f"Valid plants           : {len(results)}")

print("\nHealth Summary")
print("------------------------------")
print(f"Healthy   : {healthy_count}")
print(f"Moderate  : {moderate_count}")
print(f"Diseased  : {diseased_count}")

print(f"\nOutput saved to: {OUTPUT_JSON}")