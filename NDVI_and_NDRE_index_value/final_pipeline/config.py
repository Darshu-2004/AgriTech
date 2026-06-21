"""
Central configuration for the plant-health index pipeline.

Every other module imports its paths and constants from here, so the whole
pipeline can be re-pointed at new data by editing a single file.
"""

from pathlib import Path

# ── Project layout ───────────────────────────────────────────────────────────
# config.py lives in <project>/final_pipeline/ , so the project root is one up.
ROOT_DIR     = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR   = PIPELINE_DIR / "output"

# ── Inputs (from the ML team) ────────────────────────────────────────────────
# Plant table produced by the ML team. The CSV carries per-plant geo coords,
# bounding boxes and canopy mask filenames; it was generated from
# ggs-orthophoto (2).tif, so its coordinates register exactly to that ortho.
PLANTS_FILE = ROOT_DIR / "outputs 2" / "03_ids" / "plants_with_ids.csv"

# Per-plant binary canopy masks (one PNG per plant, sized to its bbox).
MASKS_DIR = ROOT_DIR / "outputs 2" / "04_masks"

# Coordinates are UTM zone 47N (the table has no CRS field).
PLANTS_CRS = "EPSG:32647"

# Colorized index orthomosaics (4-band RGBA GeoTIFFs) from the maps/ folder.
# These cover ~99% of the plants. The index is identified by the filename
# suffix (NDVI / NDRE). An OSAVI map also exists but is intentionally unused.
NDVI_TIF = ROOT_DIR / "maps" / "Task-of-2026-03-22T070712342Z-orthophoto-NDVI.tif"
NDRE_TIF = ROOT_DIR / "maps" / "Task-of-2026-03-22T070712342Z-orthophoto-NDRE.tif"

# True-color orthomosaic, used as the visualization basemap, for shadow
# masking, and as the reference extent for the plant-coverage check. This is
# the exact ortho the plant detections were made on (pixel-perfect overlay).
ORTHO_TIF = ROOT_DIR / "ggs-orthophoto (2).tif"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_CSV = OUTPUT_DIR / "plant_health_indices.csv"

# ── Sampling behaviour ───────────────────────────────────────────────────────
# Radius (in pixels) of the square window averaged around each plant centre.
# 0 = read the single centre pixel only. A small window is more robust to the
# anti-aliasing / compression noise present in the colorized rasters.
SAMPLE_RADIUS_PX = 1

# Pixels equal to this RGB are treated as background / no-data and skipped.
BACKGROUND_RGB = (255, 255, 255)

# Real (continuous) index values exported from the multispectral source.
# Disabled in favour of the wider-coverage maps/ rasters (the CSV only covered
# ~76% of the field). Set True to switch back to continuous values.
RAW_INDEX_CSV = ROOT_DIR / "ndvi_ndre_Unknown_2026-06-10.csv"
USE_RAW_INDEX_CSV = False
RAW_MAX_MATCH_M = 0.5     # max distance from a canopy pixel to a raw sample

# Canopy-mask sampling: average each index over the plant's actual canopy
# pixels (from outputs 2/04_masks). Used (against the colorized rasters) when
# the raw index CSV is unavailable.
USE_CANOPY_MASKS = True

# GSD (m/pixel) of the ggs orthomosaic the masks were drawn on — used to turn
# canopy pixel area into physical m^2.
ORTHO_GSD_M = 0.013458

# Otsu shadow masking: sample index values only from sunlit leaf pixels
# (uses the true-color orthomosaic to detect shadow). Fallback when canopy
# masks are not available. Set False for the simpler centre-window mean.
USE_SHADOW_MASKING = True
SHADOW_WINDOW_RADIUS_PX = 5     # half-size of the ortho window per plant

# Detections labelled NOISE are weeds / false positives, not crops; their
# index + health values are forced blank.
NOISE_LABEL = "NOISE"

# Decimal precision for lat/lon written to the CSV.
COORD_DECIMALS = 8
