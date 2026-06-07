# AgriTech — Secure Drone Imagery Processing Pipeline

We utilize a secure, lightweight background instance of **NodeODM (OpenDroneMap Core)** via Docker to process raw drone captures into 3D point clouds, digital surface models (DSM), and high-resolution orthophotos.

---

## Setup Guide

To spin up the local background rendering stack, configure the workspace environment, and execute the automation script:

### Step 1: Install Local Prerequisites

Ensure you have the following baseline environments active and configured on your machine:

- **Python 3.x**  
  (CRITICAL: Ensure the box for *"Add python.exe to PATH"* is checked during installation setup).

- **Docker Desktop**  
  (Must be open, authenticated, and actively running in your system taskbar).

### Step 2: Boot the Secured NodeODM Container Engine

Open your local terminal (Command Prompt, PowerShell, or Git Bash) and run the following execution command as a single line to pull down the engine and establish runtime security boundaries:

```bash
docker run -d -p 3000:3000 -v /absolute/path/to/local/storage:/var/www/data --name nodeodm_engine opendronemap/nodeodm --token MySecureProjectToken
```

- **Storage Mapping (`-v`)**: Replace `/absolute/path/to/local/storage` with a dedicated directory path on your local computer. This acts as a persistent volume mount so Docker can store processing cached files directly on your disk.

- **Token Authentication (`--token`)**: The engine is secured with an explicit protection key (`MySecureProjectToken2026`). The Python pipeline will automatically be rejected by the container if this key doesn't match.

### Step 3: Local Script Environment Parameters

Before executing the pipeline, open the `orthomosaic_pipeline.py` script inside an editor and adjust the parameters located in the **CONFIG SECTION** at the top of the file:

- **`API_HOST`**: Kept as `"localhost"` since the container runs natively on your current machine.

- **`API_PORT`**: Map directly to `3000` to hook into the standalone NodeODM port.

- **`API_TOKEN`**: Leave this matched to `"MySecureProjectToken2026"` to pass the container security guard.

- **`SRC_IMAGES`**: Provide the absolute folder path where your target farm drone imagery is stored.

- **`OUTPUT_MAPS`**: Define the target folder directory path where the finalized stitched outputs should download.

### Step 4: Install Dependencies and Execute

Once your folder paths match your current laptop directories, open your command terminal, navigate to the `team_maps` folder, and run:

```bash
pip install pyodm

python "orthomosaic.py"
```