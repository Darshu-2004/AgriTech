import os
import time
from pyodm import Node
from pyodm.exceptions import TaskFailedError

# ==============================================================================
#                               CONFIG SECTION
# ==============================================================================
# Change these variables to match your local setup before executing.

# 1. NODEODM CONNECTION PARAMETERS
API_HOST = "localhost"
API_PORT = 3000

# 2. CONTAINER AUTHENTICATION
# Must exactly match the string used in your 'docker run --token <token>' setup
API_TOKEN = "MySecureProjectToken"

# 3. DIRECTORY TARGETS
SRC_IMAGES = r"C:\PATH\TO\YOUR\LOCAL\DRONE\IMAGES"
OUTPUT_MAPS = r"C:\PATH\TO\YOUR\LOCAL\OUTPUT\FOLDER"
# ==============================================================================

def run_orthomosaic_pipeline(image_folder, output_dir, api_host, api_port, api_token):
    """
    Stitches drone imagery into a high-quality orthomosaic map using NodeODM API.
    Supports mixed JPG (RGB) and TIF (Multispectral) datasets.
    """
    # 1. Initialize Connection to Docker Instance
    print(f"Connecting to NodeODM at http://{api_host}:{api_port}...")
    node = Node(api_host, api_port, token=api_token)
    
    # 2. Gather All Source Images (JPG & TIF)
    supported_extensions = ('.jpg', '.jpeg', '.tif', '.tiff')
    image_paths = [
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.lower().endswith(supported_extensions)
    ]
    
    if not image_paths:
        raise FileNotFoundError(f"No valid imagery found in {image_folder}")
        
    print(f"Loaded {len(image_paths)} images for processing.")

    # 3. Configure Processing Flags for High-Quality Multispectral Mapping
    processing_options = {
        "feature-quality": "high",          # High quality feature extraction (use 'ultra' if accuracy > time)
        "dsm": True,                        # Generate Digital Surface Model (crucial for orthorectification)
        "orthophoto-resolution": 2.0,       # Ground sampling distance (e.g., 2 cm/pixel)
        "dtm": True,                        # Generates Digital Terrain Model (Bare Earth only)
        "radiometric-calibration": "camera+sun", # Essential for correct NIR/RedEdge reflectance values (NDVI)
        "pc-quality": "high",               # Point cloud density processing tier
        "max-concurrency": 2                # Set to 2 to prevent local laptop RAM from bottlenecking at scale
    }

    # 4. Trigger the Stitching Task
    print("Uploading images and spinning up stitching workflow...")
    try:
        task = node.create_task(image_paths, options=processing_options, name="Plant_Health_Orthomosaic")
        print(f"Task successfully initialized. UUID: {task.uuid}")
        
        # 5. Monitor Status Safely
        while True:
            info = task.info()
            if info.status.name == "COMPLETED":
                print("\nStitching completed successfully!")
                break
            elif info.status.name in ["FAILED", "CANCELED"]:
                raise TaskFailedError(f"Task tracking stopped with state: {info.status.name}. Error: {info.last_error}")
            
            # Print a clean progression readout on a single line
            print(f"Current Status: {info.status.name} | Progress: {info.progress}%       ", end="\r")
            time.sleep(10)

        # 6. Extract and Download the Stitched GeoTIFF Assets
        print(f"Downloading final GeoTIFF orthomosaic to: {output_dir}")
        task.download_assets(output_dir)
        print("Pipeline run successfully completed.")

    except TaskFailedError as e:
        print(f"\nExecution failed: {e}")
    except Exception as e:
        print(f"\nAn unexpected network or pipeline error occurred: {e}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_MAPS, exist_ok=True)
    
    run_orthomosaic_pipeline(
        image_folder=SRC_IMAGES,
        output_dir=OUTPUT_MAPS,
        api_host=API_HOST,
        api_port=API_PORT,
        api_token=API_TOKEN
    )