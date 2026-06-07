import os
import json
import time
import zipfile
import requests

# ==============================================================================
#                               CONFIG SECTION 
# ==============================================================================
# Change these placeholders to match your local setup before running the script.

# 1. AUTHENTICATION TOKEN
# WebODM tokens expire every 6 hours. Go to http://localhost:8000/api/token-auth/
# Log in via the HTML form, copy the long 'token' string, and paste it below.
TOKEN = "YOUR_LOCAL_WEBODM_TOKEN_HERE"

# 2. WEBODM PROJECT ID
# Keeps track of which project workspace to upload to. Default is usually 1.
PROJECT_ID = 1

# 3. SOURCE DATASET PATH
# Change this to the absolute path of the folder on YOUR laptop containing the drone images.
# Example: r"C:\Users\username\Desktop\Project_Images"
DATASET_FOLDER = r"C:\PATH\TO\YOUR\LOCAL\DRONE\IMAGES"

# 4. CAPTURE LIMIT
# WebODM is highly resource-intensive. Keep this cap low (e.g., 25) for initial testing so it doesn't overload your laptop's RAM during processing.
CAPTURE_LIMIT = 25

# 5. OUTPUT DESIRED FOLDER
# Change this to the absolute path where you want the final stitched maps/ZIP files to download.
DOWNLOAD_FOLDER = r"C:\PATH\TO\YOUR\LOCAL\OUTPUT\FOLDER"

# ==============================================================================

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

headers = {
    "Authorization": "JWT " + TOKEN
}

# =====================
# FIND RGB IMAGES
# =====================
#change the filter from "_D.JPG" to just ".JPG" or ".jpg" so the script can find them if required.
jpgs = sorted([
    f for f in os.listdir(DATASET_FOLDER)
    if f.endswith("_D.JPG")
])

jpgs = jpgs[:CAPTURE_LIMIT]

print(f"Selected {len(jpgs)} RGB captures")

# =====================
# BUILD FILE LIST
# =====================

files = []

for f in jpgs:

    path = os.path.join(DATASET_FOLDER, f)

    files.append(
        (
            "images",
            (
                f,
                open(path, "rb"),
                "image/jpeg"
            )
        )
    )

# =====================
# OPTIONS
# =====================

options = [
    {"name": "feature-quality", "value": "high"},
    {"name": "pc-quality", "value": "high"},
    {"name": "orthophoto-resolution", "value": 1},
    {"name": "dsm", "value": True},
    {"name": "dtm", "value": True}
]

data = {
    "name": "Pipeline_Test_25",
    "options": json.dumps(options)
}

# =====================
# CREATE TASK
# =====================

print("Uploading...")

r = requests.post(
    f"http://localhost:8000/api/projects/{PROJECT_ID}/tasks/",
    headers=headers,
    files=files,
    data=data
)

for _, fd in files:
    fd[1].close()

if r.status_code != 201:
    print(r.text)
    raise SystemExit()

task = r.json()

task_id = task["id"]

print("Task Created")
print("Task ID:", task_id)

# =====================
# WAIT
# =====================

while True:

    r = requests.get(
        f"http://localhost:8000/api/projects/{PROJECT_ID}/tasks/{task_id}/",
        headers=headers
    )

    info = r.json()

    status = info.get("status")
    progress = info.get("running_progress")

    print(
        f"Status={status} Progress={progress}"
    )

    if status == 40:
        print("COMPLETED")
        break

    if status == 50:
        print("FAILED")
        raise SystemExit()

    time.sleep(30)

# =====================
# DOWNLOAD ALL ASSETS
# =====================

print("Downloading assets...")

download_url = (
    f"http://localhost:8000/api/projects/"
    f"{PROJECT_ID}/tasks/{task_id}/download/all.zip"
)

r = requests.get(
    download_url,
    headers=headers,
    stream=True
)

zip_path = os.path.join(
    DOWNLOAD_FOLDER,
    f"{task_id}.zip"
)

with open(zip_path, "wb") as f:
    for chunk in r.iter_content(8192):
        f.write(chunk)

print("ZIP Downloaded")

# =====================
# EXTRACT
# =====================

extract_folder = os.path.join(
    DOWNLOAD_FOLDER,
    task_id
)

os.makedirs(extract_folder, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_folder)

print("Extraction Complete")

print("\nOUTPUT LOCATION:")
print(extract_folder)