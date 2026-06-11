import os
import json
import time
import zipfile
import shutil
import requests
import numpy as np
import rasterio
from matplotlib import cm
# ==================================================
# CONFIG
# ==================================================

TOKEN = os.getenv("WEBODM_TOKEN")
if not TOKEN:
    raise ValueError(
        "Please set WEBODM_TOKEN environment variable"
    )

PROJECT_ID = 1

DATASET_FOLDER = r"C:\PATH\TO\DATASET"

OUTPUT_FOLDER = r"C:\PATH\TO\OUTPUT"

CAPTURE_LIMIT = 5      # Use None for full dataset

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

headers = {
    "Authorization": "JWT " + TOKEN
}

# ==================================================
# TASK MONITOR
# ==================================================

def wait_for_task(task_id):

    while True:

        r = requests.get(
            f"http://localhost:8000/api/projects/{PROJECT_ID}/tasks/{task_id}/",
            headers=headers
        )

        info = r.json()

        status = info.get("status")
        progress = info.get("running_progress")

        print(
            f"Task={task_id[:8]} "
            f"Status={status} "
            f"Progress={progress}"
        )

        if status == 40:
            print("COMPLETED")
            return

        if status == 50:
            raise Exception("TASK FAILED")

        time.sleep(30)

# ==================================================
# DOWNLOAD + EXTRACT
# ==================================================

def download_assets(task_id, folder_name):

    url = (
        f"http://localhost:8000/api/projects/"
        f"{PROJECT_ID}/tasks/{task_id}/download/all.zip"
    )

    zip_path = os.path.join(
        OUTPUT_FOLDER,
        folder_name + ".zip"
    )

    print("Downloading:", folder_name)

    r = requests.get(
        url,
        headers=headers,
        stream=True
    )

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    extract_dir = os.path.join(
        OUTPUT_FOLDER,
        folder_name
    )

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    return extract_dir

# ==================================================
# CREATE TASK
# ==================================================

def create_task(task_name, image_paths, options):

    files = []

    for path in image_paths:

        ext = os.path.splitext(path)[1].lower()

        mime = (
            "image/jpeg"
            if ext in [".jpg", ".jpeg"]
            else "image/tiff"
        )

        files.append(
            (
                "images",
                (
                    os.path.basename(path),
                    open(path, "rb"),
                    mime
                )
            )
        )

    data = {
        "name": task_name,
        "options": json.dumps(options)
    }

    r = requests.post(
        f"http://localhost:8000/api/projects/{PROJECT_ID}/tasks/",
        headers=headers,
        files=files,
        data=data
    )

    for _, fd in files:
        fd[1].close()

    if r.status_code != 201:
        raise Exception(r.text)

    return r.json()["id"]

# ==================================================
# RGB FILES
# ==================================================

rgb_files = sorted([

    os.path.join(DATASET_FOLDER, f)

    for f in os.listdir(DATASET_FOLDER)

    if f.endswith("_D.JPG")

])

if CAPTURE_LIMIT:
    rgb_files = rgb_files[:CAPTURE_LIMIT]

print("RGB Images:", len(rgb_files))

# ==================================================
# MULTISPECTRAL FILES
# ==================================================

ms_files = []

for rgb in rgb_files:

    base = os.path.basename(rgb)

    base = base.replace("_D.JPG", "")

    for suffix in [

        "_MS_G.TIF",
        "_MS_R.TIF",
        "_MS_NIR.TIF",
        "_MS_RE.TIF"

    ]:

        path = os.path.join(
            DATASET_FOLDER,
            base + suffix
        )

        if os.path.exists(path):
            ms_files.append(path)

print("MS Images:", len(ms_files))

# ==================================================
# RGB TASK
# ==================================================

rgb_options = [

    {"name":"feature-quality","value":"high"},
    {"name":"pc-quality","value":"high"},
    {"name":"orthophoto-resolution","value":1},
    {"name":"dsm","value":True},
    {"name":"dtm","value":True}

]

print("\nCreating RGB Task...")

rgb_task = create_task(
    "RGB_Task",
    rgb_files,
    rgb_options
)

print("RGB Task ID:", rgb_task)

wait_for_task(rgb_task)

rgb_folder = download_assets(
    rgb_task,
    "RGB_Output"
)

# ==================================================
# MULTISPECTRAL TASK
# ==================================================

ms_options = [

    {"name":"feature-quality","value":"high"},
    {"name":"pc-quality","value":"high"},
    {"name":"dsm","value":True},
    {"name":"dtm","value":True},
    {"name":"radiometric-calibration","value":"camera+sun"}

]

print("\nCreating MS Task...")

ms_task = create_task(
    "MS_Task",
    ms_files,
    ms_options
)

print("MS Task ID:", ms_task)

wait_for_task(ms_task)

ms_folder = download_assets(
    ms_task,
    "MS_Output"
)

# ==================================================
# COPY RGB PRODUCTS
# ==================================================

shutil.copy2(
    os.path.join(
        rgb_folder,
        "odm_orthophoto",
        "odm_orthophoto.tif"
    ),
    os.path.join(
        OUTPUT_FOLDER,
        "Orthophoto.tif"
    )
)

shutil.copy2(
    os.path.join(
        rgb_folder,
        "odm_dem",
        "dsm.tif"
    ),
    os.path.join(
        OUTPUT_FOLDER,
        "DSM.tif"
    )
)

shutil.copy2(
    os.path.join(
        rgb_folder,
        "odm_dem",
        "dtm.tif"
    ),
    os.path.join(
        OUTPUT_FOLDER,
        "DTM.tif"
    )
)

# ==================================================
# LOAD MULTISPECTRAL TIFF
# ==================================================

ms_tif = os.path.join(
    ms_folder,
    "odm_orthophoto",
    "odm_orthophoto.tif"
)

with rasterio.open(ms_tif) as src:

    profile = src.profile

    red = src.read(1).astype(np.float32)
    green = src.read(2).astype(np.float32)
    nir = src.read(3).astype(np.float32)
    rededge = src.read(4).astype(np.float32)
    mask = src.read(5)
    
    

# ==================================================
# INDICES
# ==================================================

eps = 1e-10

ndvi = (nir - red) / (nir + red + eps)

ndre = (nir - rededge) / (nir + rededge + eps)

osavi = (nir - red) / (nir + red + 0.16 + eps)

gdvi = (nir - green) / (nir + green + eps)

ndvi[mask == 0] = np.nan
ndre[mask == 0] = np.nan
osavi[mask == 0] = np.nan
gdvi[mask == 0] = np.nan
# ==================================================
# SAVE TIFFS
# ==================================================

profile.update(
    dtype=rasterio.float32,
    count=1
)

def save_tif(path, arr):

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            arr.astype(np.float32),
            1
        )

save_tif(
    os.path.join(
        OUTPUT_FOLDER,
        "NDVI.tif"
    ),
    ndvi
)

save_tif(
    os.path.join(
        OUTPUT_FOLDER,
        "NDRE.tif"
    ),
    ndre
)

save_tif(
    os.path.join(
        OUTPUT_FOLDER,
        "OSAVI.tif"
    ),
    osavi
)

save_tif(
    os.path.join(
        OUTPUT_FOLDER,
        "GDVI.tif"
    ),
    gdvi
)
def save_colored_tif(path, arr):

    arr = np.nan_to_num(arr)

    arr_min = np.percentile(arr, 2)
    arr_max = np.percentile(arr, 98)

    norm = (
        (arr - arr_min)
        /
        (arr_max - arr_min + 1e-10)
    )

    norm = np.clip(norm, 0, 1)

    cmap = cm.get_cmap("RdYlGn")

    rgb = cmap(norm)[:, :, :3]

    rgb = (rgb * 255).astype(np.uint8)

    rgb_profile = profile.copy()

    rgb_profile.update(
        dtype=rasterio.uint8,
        count=3
    )

    with rasterio.open(
        path,
        "w",
        **rgb_profile
    ) as dst:

        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)

save_colored_tif(
    os.path.join(
        OUTPUT_FOLDER,
        "NDVI_Color.tif"
    ),
    ndvi
)

save_colored_tif(
    os.path.join(
        OUTPUT_FOLDER,
        "NDRE_Color.tif"
    ),
    ndre
)

save_colored_tif(
    os.path.join(
        OUTPUT_FOLDER,
        "OSAVI_Color.tif"
    ),
    osavi
)

save_colored_tif(
    os.path.join(
        OUTPUT_FOLDER,
        "GDVI_Color.tif"
    ),
    gdvi
)
print("\n=================================")
print("ALL PROCESSING COMPLETED")
print("=================================")
print(OUTPUT_FOLDER)