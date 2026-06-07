# AgriTech — Drone Imagery Processing Pipeline

We have implemented an automated photogrammetry pipeline that interfaces directly with a local **WebODM (OpenDroneMap)** engine to process raw drone captures into 3D point clouds, DSMs, and digital orthophotos.

---

## What We Have Done So Far
* **Environment Architecture:** Configured a local containerized deployment system using Docker to handle heavy photogrammetry workloads locally.
* **REST API Integration:** Built a complete Python automation engine (`maps.py`) using the WebODM REST API layer.
* **Asynchronous Task Management:** Implemented an automated polling script that uploads raw drone images, monitors processing status dynamically every 30 seconds, and auto-downloads/extracts completed spatial asset ZIPs.

---

## Setup Guide 

To run the pipeline script on your local machine, you must configure the background processing engine and update the local environment variables in the script. Follow these steps precisely:

### 1. Prerequisites Installation
Make sure you have the following installed on your machine:
* **Python 3.x** (Ensure you check the box to *"Add python.exe to PATH"* during installation).
* **Git for Windows** (Provides Git Bash).
* **Docker Desktop** (Required to spin up the WebODM environment).

### 2. Booting the WebODM Engine
Open **Git Bash**, navigate to your preferred workspace folder, clone the WebODM core repository, and initialize the container architecture using these commands:
git clone https://github.com/WebODM/WebODM --config core.autocrlf=input --depth 1
cd WebODM
./webodm.sh start

### 3. Fetching Your Local Auth Token
WebODM secures its API via JSON Web Tokens (JWT) which expire every 6 hours by default. Before running the script, you must grab a fresh token for your local instance:
1. Ensure WebODM is running and navigate to `http://localhost:8000/api/token-auth/` in your browser.
2. Authenticate using the username and password you created on your local WebODM admin dashboard via the native HTML POST form.
3. Click **POST**, scroll down, and copy the long generated string from the `token` key field in the JSON response payload.

### 4. Code Modification Constraints (Local Setup)
Before running `maps.py`, open the file in an editor and update the following configuration placeholders to match your laptop's local environment:

* `TOKEN`: Paste your fresh 6-hour JWT token string here.
* `DATASET_FOLDER`: Provide the absolute path to your local folder containing the raw drone imagery.
* `DOWNLOAD_FOLDER`: Define the local path where you want the final processed asset ZIPs to download.

>  **Important File Extension Check:** The script defaults to looking for files ending in `_D.JPG`. If the drone imagery dataset uses standard names (e.g., `DJI_001.JPG`), make sure to change the filter string on Line 36 from `"_D.JPG"` to `".JPG"` or `".jpg"`, otherwise the script will register 0 images.

### 5. Running the Pipeline
Once your configuration parameters are set up, open your command prompt, install the necessary dependencies, and execute the script:
pip install requests
python "maps.py"