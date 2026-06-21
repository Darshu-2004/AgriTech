import pandas as pd
import numpy as np
import os
import matplotlib
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import multiprocessing

# Define workspace directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)

csv_path = os.path.join(workspace_dir, 'outputs', '03_ids', 'plants_with_ids.csv')
output_csv_path = os.path.join(workspace_dir, 'plants_extracted_indices.csv')

osavi_path = os.path.join(workspace_dir, 'dataset1', 'Task-of-2026-03-22T063838646Z-orthophoto-OSAVI.tif')
ndvi_path = os.path.join(workspace_dir, 'dataset1', 'Task-of-2026-03-22T070712342Z-orthophoto-NDVI.tif')
ndre_path = os.path.join(workspace_dir, 'dataset1', 'Task-of-2026-03-22T070712342Z-orthophoto-NDRE.tif')
ortho_path = os.path.join(workspace_dir, 'dataset1', 'ggs-orthophoto (2).tif')

# Pure-numpy Otsu's thresholding
def otsu_threshold(gray_img):
    flat = gray_img.ravel()
    if len(flat) == 0:
        return 50.0
    hist, bin_edges = np.histogram(flat, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    w1 = np.where(weight1 == 0, 1e-5, weight1)
    w2 = np.where(weight2 == 0, 1e-5, weight2)
    
    mean1 = np.cumsum(hist * bin_centers) / w1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]
    
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if len(variance12) == 0:
        return 50.0
    idx = np.argmax(variance12)
    return bin_centers[idx]

# Multiprocessing worker function
def extract_chunk_worker(args):
    (chunk_df, osavi_path, ndvi_path, ndre_path, ortho_path,
     osa_georef, ndv_georef, ndr_georef,
     dc_osa, dr_osa, dc_ndv, dr_ndv, dc_ndr, dr_ndr,
     osavi_is_float, ndvi_is_float, ndre_is_float) = args

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import numpy as np
    import matplotlib
    
    # Load OSAVI, NDVI, NDRE arrays using PIL
    with Image.open(osavi_path) as img:
        osavi_arr = np.array(img)
    with Image.open(ndvi_path) as img:
        ndvi_arr = np.array(img)
    with Image.open(ndre_path) as img:
        ndre_arr = np.array(img)
        
    # Check and squeeze dimensions to 2D if single band
    if len(osavi_arr.shape) == 3 and osavi_arr.shape[2] == 1:
        osavi_arr = osavi_arr[:, :, 0]
    if len(ndvi_arr.shape) == 3 and ndvi_arr.shape[2] == 1:
        ndvi_arr = ndvi_arr[:, :, 0]
    if len(ndre_arr.shape) == 3 and ndre_arr.shape[2] == 1:
        ndre_arr = ndre_arr[:, :, 0]

    # Open RGB mosaic file descriptor for lazy windowed crops
    src_rgb = Image.open(ortho_path)
    w_rgb, h_rgb = src_rgb.size
    
    h_osa, w_osa = osavi_arr.shape[0], osavi_arr.shape[1]
    h_ndv, w_ndv = ndvi_arr.shape[0], ndvi_arr.shape[1]
    h_ndr, w_ndr = ndre_arr.shape[0], ndre_arr.shape[1]
    
    # Initialize color maps for RGB/RGBA conversions
    cmap = matplotlib.colormaps['RdYlGn']
    val_space = np.linspace(1.0, -1.0, 2000)
    cmap_space = np.linspace(0.0, 1.0, 2000)
    cmap_colors = (cmap(cmap_space)[:, :3] * 255).astype(np.float32)
    
    local_cache = {}
    
    def rgb_to_index_cached(r, g, b):
        key = (int(r), int(g), int(b))
        if key in local_cache:
            return local_cache[key]
        diff = cmap_colors - np.array([r, g, b], dtype=np.float32)
        dist = np.sum(diff**2, axis=1)
        best_idx = np.argmin(dist)
        val = val_space[best_idx]
        local_cache[key] = val
        return val

    def extract_pixel_val(arr, is_float, r, c, w_arr, h_arr):
        if 0 <= c < w_arr and 0 <= r < h_arr:
            p = arr[r, c]
            if is_float:
                return float(p)
            else:
                if len(p) >= 4 and p[3] == 0:  # Transparent pixel check
                    return np.nan
                return rgb_to_index_cached(p[0], p[1], p[2])
        return np.nan

    results = []
    orig_x_osa, orig_y_osa, dx_osa, dy_osa = osa_georef
    orig_x_ndv, orig_y_ndv, dx_ndv, dy_ndv = ndv_georef
    orig_x_ndr, orig_y_ndr, dx_ndr, dy_ndr = ndr_georef
    
    for idx, row in chunk_df.iterrows():
        gx = row['geo_x']
        gy = row['geo_y']
        px_x = int(row['pixel_x'])
        px_y = int(row['pixel_y'])
        
        if row['sector_label'] == 'NOISE':
            results.append((idx, np.nan, np.nan, np.nan))
            continue
            
        # --- SHADOW MASKING ---
        # Lazy load 10x10 crop window from disk dynamically using PIL (No memory leak!)
        radius = 5
        y_min, y_max = max(0, px_y - radius), min(h_rgb, px_y + radius)
        x_min, x_max = max(0, px_x - radius), min(w_rgb, px_x + radius)
        
        patch_img = src_rgb.crop((x_min, y_min, x_max, y_max))
        patch = np.array(patch_img)
        sunlit_offsets = []
        
        if patch.size > 0:
            brightness = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
            t_otsu = otsu_threshold(brightness)
            thresh_val = max(50.0, t_otsu)  # Agronomic floor safety
            sunlit_mask = brightness > thresh_val
            
            y_indices, x_indices = np.where(sunlit_mask)
            for dy, dx in zip(y_indices, x_indices):
                sunlit_offsets.append((dy - radius, dx - radius))
                
        if not sunlit_offsets:
            sunlit_offsets = [(0, 0)]
            
        # --- EXTRACT SPECTRAL INDICES ---
        osa_list, ndv_list, ndr_list = [], [], []
        
        for dy, dx in sunlit_offsets:
            # 1. OSAVI
            c_osa = int(round(((gx + dx * 0.0135) - orig_x_osa) / dx_osa)) + dc_osa
            r_osa = int(round((orig_y_osa - (gy - dy * 0.0135)) / dy_osa)) + dr_osa
            val = extract_pixel_val(osavi_arr, osavi_is_float, r_osa, c_osa, w_osa, h_osa)
            if not np.isnan(val):
                osa_list.append(val)
                
            # 2. NDVI / NDRE
            c_ndv = int(round(((gx + dx * 0.0135) - orig_x_ndv) / dx_ndv)) + dc_ndv
            r_ndv = int(round((orig_y_ndv - (gy - dy * 0.0135)) / dy_ndv)) + dr_ndv
            v_ndv = extract_pixel_val(ndvi_arr, ndvi_is_float, r_ndv, c_ndv, w_ndv, h_ndv)
            v_ndr = extract_pixel_val(ndre_arr, ndre_is_float, r_ndv, c_ndv, w_ndv, h_ndv)
            if not np.isnan(v_ndv): ndv_list.append(v_ndv)
            if not np.isnan(v_ndr): ndr_list.append(v_ndr)
            
        osa_val = np.mean(osa_list) if osa_list else np.nan
        ndv_val = np.mean(ndv_list) if ndv_list else np.nan
        ndr_val = np.mean(ndr_list) if ndr_list else np.nan
        
        results.append((idx, osa_val, ndv_val, ndr_val))
        
    src_rgb.close()
    return results

def get_ms_georef_params(tif_path):
    img = Image.open(tif_path)
    tags = img.tag_v2
    # ModelPixelScaleTag is 33550, ModelTiepointTag is 33922
    if 33550 in tags and 33922 in tags:
        ps = tags[33550]
        tp = tags[33922]
        img.close()
        return tp[3], tp[4], ps[0], ps[1]
    img.close()
    raise ValueError(f"GeoTIFF tags missing in {tif_path}")

def inspect_map_format(path):
    img = Image.open(path)
    bands = len(img.getbands())
    is_float = (bands == 1)
    print(f"Map: {os.path.basename(path)} -> Bands: {bands}, Float32: {is_float}")
    img.close()
    return is_float

if __name__ == '__main__':
    import time
    print("=== PIPELINE STEP 1: INDEX EXTRACTION (MULTIPROCESSING + PILLOW) ===")
    
    # Verify source file presence
    for path in [csv_path, osavi_path, ndvi_path, ndre_path, ortho_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at: {path}")
            exit(1)
            
    # Load coordinates
    print("Loading plants coordinates...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} plant records.")
    
    # Inspect maps
    osavi_is_float = inspect_map_format(osavi_path)
    ndvi_is_float = inspect_map_format(ndvi_path)
    ndre_is_float = inspect_map_format(ndre_path)
    
    # Get georef params
    osa_georef = get_ms_georef_params(osavi_path)
    ndv_georef = get_ms_georef_params(ndvi_path)
    ndr_georef = get_ms_georef_params(ndre_path)
    
    # Configured shifts
    dc_osa, dr_osa = 4, 2
    dc_ndv, dr_ndv = 12, 7
    dc_ndr, dr_ndr = 12, 7
    
    # Parallelize extraction
    num_cores = min(multiprocessing.cpu_count(), 8)
    print(f"Parallelizing extraction across {num_cores} CPU cores...")
    
    # Split dataframe into chunks
    chunks = np.array_split(df, num_cores)
    args_list = []
    for chunk in chunks:
        args_list.append((
            chunk, osavi_path, ndvi_path, ndre_path, ortho_path,
            osa_georef, ndv_georef, ndr_georef,
            dc_osa, dr_osa, dc_ndv, dr_ndv, dc_ndr, dr_ndr,
            osavi_is_float, ndvi_is_float, ndre_is_float
        ))
        
    t0 = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        all_results = pool.map(extract_chunk_worker, args_list)
    t1 = time.time()
    print(f"Parallel extraction completed in {t1 - t0:.2f} seconds.")
    
    # Flatten and sort results
    flattened = [item for sublist in all_results for item in sublist]
    flattened.sort(key=lambda x: x[0])
    
    osavi_values = [x[1] for x in flattened]
    ndvi_values = [x[2] for x in flattened]
    ndre_values = [x[3] for x in flattened]
    
    df['osavi'] = osavi_values
    df['ndvi'] = ndvi_values
    df['ndre'] = ndre_values
    
    # Force index values to NaN for coordinates labeled as 'NOISE' (weeds/outliers)
    noise_mask = df['sector_label'] == 'NOISE'
    df.loc[noise_mask, ['osavi', 'ndvi', 'ndre']] = np.nan
    
    # Save CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Extracted index values successfully. Output saved to: {output_csv_path}")
    
    # Check correlations
    sub = df.dropna(subset=['ndvi', 'osavi', 'ndre'])
    if len(sub) > 1:
        corr_ndvi_osavi = np.corrcoef(sub['ndvi'], sub['osavi'])[0, 1]
        corr_ndvi_ndre = np.corrcoef(sub['ndvi'], sub['ndre'])[0, 1]
        print(f"Pearson correlations:")
        print(f"  NDVI vs OSAVI: {corr_ndvi_osavi:.4f}")
        print(f"  NDVI vs NDRE: {corr_ndvi_ndre:.4f}")
    else:
        print("Could not compute correlations: no valid rows found.")
        
    print("Step 1 finished successfully.\n")
