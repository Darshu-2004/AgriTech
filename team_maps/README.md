# AgriTech — Drone Imagery Processing Pipeline

An automated photogrammetry and multispectral analysis pipeline that interfaces directly with a local **WebODM (OpenDroneMap)** engine. The script processes raw RGB and multi-band drone captures to generate high-resolution orthophotos, Digital Surface Models (DSM), Digital Terrain Models (DTM), and multiple vegetation health indices.

---

## What the Pipeline Does

**Dual-Task Processing:** Automatically identifies RGB and multispectral imagery within a dataset and processes them through separate WebODM workflows.
* **REST API Integration:** Uses the WebODM API to create tasks, upload imagery, monitor execution status, and download completed outputs.
* **Asynchronous Task Polling:** Continuously checks task progress every 30 seconds until completion.
* **Radiometric Calibration:** Applies WebODM's `camera+sun` radiometric calibration workflow for multispectral imagery.
* **Automated Asset Extraction:** Downloads processing archives and extracts all generated outputs automatically.
* **Raster-Based Vegetation Analysis:** Computes multiple vegetation indices directly from calibrated multispectral orthophotos using Rasterio and NumPy.
* **Colorized GIS Products:** Generates both raw floating-point GeoTIFFs and visual-ready colorized vegetation maps using the `RdYlGn` color scale.

---

## Processed Analytics & Vegetation Indices

The multispectral workflow extracts the following calibrated bands:

* Red
* Green
* Near Infrared (NIR)
* Red Edge
* Alpha Mask

The following vegetation indices are generated:

### NDVI (Normalized Difference Vegetation Index)

NDVI = (NIR − Red) / (NIR + Red)

### NDRE (Normalized Difference Red Edge Index)

NDRE = (NIR − RedEdge) / (NIR + RedEdge)

### OSAVI (Optimized Soil Adjusted Vegetation Index)

OSAVI = (NIR − Red) / (NIR + Red + 0.16)

### GDVI (Green Difference Vegetation Index)

GDVI = (NIR − Green) / (NIR + Green)

Pixels outside valid image regions are removed using the alpha mask before index generation.

---

## Setup Guide

### 1. Prerequisites Installation

Install the following dependencies:

* Python 3.x
* Git
* Docker Desktop

Recommended for local photogrammetry processing:

* Minimum 8 GB RAM
* Minimum 4 CPU cores allocated to Docker

---

### 2. Booting the WebODM Engine

Open Git Bash (or WSL) and run:

```bash
git clone https://github.com/WebODM/WebODM --config core.autocrlf=input --depth 1

cd WebODM

./webodm.sh start
```

Leave this terminal running.

Once started, access:

```text
http://localhost:8000
```

Complete the initial WebODM administrator setup if required.

---

### 3. Fetching Your Local Auth Token

WebODM secures its API using JWT authentication.

1. Navigate to:

```text
http://localhost:8000/api/token-auth/
```

2. Log in using your WebODM credentials.

3. Click **POST**.

4. Copy the generated token value.

The token expires after approximately 6 hours.

---

### 4. Local Configuration

Before running the script, modify the configuration section near the top of `maps.py`:

```python
TOKEN = os.getenv("WEBODM_TOKEN")

PROJECT_ID = 1

DATASET_FOLDER = r"C:\Path\To\Drone\Dataset"

OUTPUT_FOLDER = r"C:\Path\To\Final_Output"

CAPTURE_LIMIT = 5
```

#### Configuration Notes

| Variable       | Purpose                                                          |
| -------------- | ---------------------------------------------------------------- |
| TOKEN          | JWT authentication token                                         |
| PROJECT_ID     | Existing WebODM project ID                                       |
| DATASET_FOLDER | Folder containing raw drone imagery                              |
| OUTPUT_FOLDER  | Directory where all outputs are stored                           |
| CAPTURE_LIMIT  | Limits number of captures processed; use `None` for full dataset |

---

### Dataset Naming Requirements

The script expects RGB imagery with filenames ending in:

```text
_D.JPG
```

Example:

```text
DJI_0001_D.JPG
```

For every RGB image, the script automatically searches for matching multispectral bands:

```text
DJI_0001_MS_G.TIF
DJI_0001_MS_R.TIF
DJI_0001_MS_NIR.TIF
DJI_0001_MS_RE.TIF
```

The multispectral workflow is built dynamically from these matching files.

If your drone uses a different naming convention, update the suffix filters in the script before execution.

---

### 5. Installing Dependencies

Install the required Python packages:

```bash
pip install requests numpy rasterio matplotlib
```

---

### 6. Running the Pipeline

Execute:

```bash
python maps.py
```

The script will:

1. Create an RGB processing task.
2. Wait for completion.
3. Download and extract outputs.
4. Create a multispectral processing task.
5. Wait for completion.
6. Download and extract outputs.
7. Generate vegetation index rasters.
8. Produce colorized vegetation maps.

---

## Output Structure

After successful completion, the `OUTPUT_FOLDER` contains:

```text
Final_Output/
│
├── Orthophoto.tif
├── DSM.tif
├── DTM.tif
│
├── NDVI.tif
├── NDRE.tif
├── OSAVI.tif
├── GDVI.tif
│
├── NDVI_Color.tif
├── NDRE_Color.tif
├── OSAVI_Color.tif
└── GDVI_Color.tif
```

### Output Description

| File           | Description                               |
| -------------- | ----------------------------------------- |
| Orthophoto.tif | RGB orthomosaic                           |
| DSM.tif        | Digital Surface Model                     |
| DTM.tif        | Digital Terrain Model                     |
| NDVI.tif       | Raw NDVI raster                           |
| NDRE.tif       | Raw NDRE raster                           |
| OSAVI.tif      | Raw OSAVI raster                          |
| GDVI.tif       | Raw GDVI raster                           |
| *_Color.tif    | Colorized visualization maps using RdYlGn |

---

## Current Processing Workflow

```text
Raw RGB Images
        │
        ▼
     WebODM
        │
        ▼
Orthophoto + DSM + DTM

Raw Multispectral Images
        │
        ▼
Radiometric Calibration
        │
        ▼
Multispectral Orthophoto
        │
        ▼
NDVI / NDRE / OSAVI / GDVI
        │
        ▼
GIS GeoTIFFs + Colorized Maps
```
