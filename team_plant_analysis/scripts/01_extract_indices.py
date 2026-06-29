import pandas as pd
import numpy as np
import os
import multiprocessing
import rasterio

# Define workspace directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)

csv_path = os.path.join(workspace_dir, 'outputs', '03_ids', 'plants_with_ids.csv')
output_csv_path = os.path.join(workspace_dir, 'plants_extracted_indices.csv')

osavi_path = os.path.join(workspace_dir, 'dataset1', 'Task-of-2026-03-22T063838646Z-orthophoto-OSAVI.tif')
ndvi_path = os.path.join(workspace_dir, 'dataset1', 'Task-of-2026-03-22T070712342Z-orthophoto-NDVI.tif')
ndre_path = os.path.join(workspace_dir, 'dataset1', 'Task-of-2026-03-22T070712342Z-orthophoto-NDRE.tif')
ortho_path = os.path.join(workspace_dir, 'dataset1', 'ggs-orthophoto (2).tif')
masks_dir = os.path.join(workspace_dir, 'outputs', '04_masks')


# Multiprocessing worker function
def extract_chunk_worker(args):
    (chunk_df, osavi_path, ndvi_path, ndre_path, ortho_path, masks_dir) = args

    import numpy as np
    import rasterio
    import os
    from PIL import Image
    from colormaps import NDVI_CMAP, NDRE_CMAP, OSAVI_CMAP

    # Load coordinate transforms
    with rasterio.open(ortho_path) as osrc:
        ggs_transform = osrc.transform

    def load_raster_data(tif_path, dx=0.0, dy=0.0):
        with rasterio.open(tif_path) as isrc:
            itf = isrc.transform
            rgb = isrc.read()[:3]
            H, W = rgb.shape[1], rgb.shape[2]
        A = (~itf) * ggs_transform  # maps ggs pixel coordinate to index pixel coordinate
        inv_itf = ~itf
        shift_c = inv_itf.a * dx + inv_itf.b * dy
        shift_r = inv_itf.d * dx + inv_itf.e * dy
        return A, rgb, H, W, shift_c, shift_r

    # Load rasters into memory in each worker
    A_osa, rgb_osa, H_osa, W_osa, shift_c_osa, shift_r_osa = load_raster_data(osavi_path)
    A_ndv, rgb_ndv, H_ndv, W_ndv, shift_c_ndv, shift_r_ndv = load_raster_data(ndvi_path)
    A_ndr, rgb_ndr, H_ndr, W_ndr, shift_c_ndr, shift_r_ndr = load_raster_data(ndre_path)

    def map_pixels(A, ocol, orow, H, W, shift_c, shift_r):
        ic = np.round(A.a * ocol + A.b * orow + A.c + shift_c).astype(int)
        ir = np.round(A.d * ocol + A.e * orow + A.f + shift_r).astype(int)
        inb = (ir >= 0) & (ir < H) & (ic >= 0) & (ic < W)
        return ir, ic, inb

    bg = np.array([255, 255, 255], dtype=np.int16)
    results = []

    for idx, row in chunk_df.iterrows():
        if row['sector_label'] == 'NOISE':
            results.append((idx, np.nan, np.nan, np.nan, np.nan, 0.0))
            continue

        mpath = os.path.join(masks_dir, str(row['mask_file']))
        if not os.path.exists(mpath):
            results.append((idx, np.nan, np.nan, np.nan, np.nan, 0.0))
            continue

        with Image.open(mpath) as mask_img:
            mask = np.asarray(mask_img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            results.append((idx, np.nan, np.nan, np.nan, np.nan, 0.0))
            continue

        # Map to native ggs orthophoto coordinates
        ocol = row['bbox_x'] + xs
        orow = row['bbox_y'] + ys

        # --- OSAVI soil gate: keep only vegetation canopy pixels ---
        ir_osa, ic_osa, inb_osa = map_pixels(A_osa, ocol, orow, H_osa, W_osa, shift_c_osa, shift_r_osa)
        keep = np.ones(len(ocol), dtype=bool)
        osa_val = np.nan

        if inb_osa.any():
            cols_o = rgb_osa[:, ir_osa[inb_osa], ic_osa[inb_osa]].T
            not_bg = ~np.all(cols_o.astype(np.int16) == bg, axis=1)
            ovals, _, _ = OSAVI_CMAP.classify(cols_o)
            veg = not_bg & (ovals >= 0.2)
            keep_inb = keep[inb_osa]
            keep_inb[:] = veg
            keep[inb_osa] = keep_inb
            if veg.any():
                osa_val = round(float(ovals[veg].mean()), 3)

        soil_frac = round(1.0 - keep.mean(), 3)
        if not keep.any():
            results.append((idx, np.nan, np.nan, np.nan, np.nan, soil_frac))
            continue

        ocol_veg, orow_veg = ocol[keep], orow[keep]

        # --- Sample NDVI ---
        ndvi_val = np.nan
        ir_ndv, ic_ndv, inb_ndv = map_pixels(A_ndv, ocol_veg, orow_veg, H_ndv, W_ndv, shift_c_ndv, shift_r_ndv)
        if inb_ndv.any():
            cols_n = rgb_ndv[:, ir_ndv[inb_ndv], ic_ndv[inb_ndv]].T
            not_bg = ~np.all(cols_n.astype(np.int16) == bg, axis=1)
            kept_n = cols_n[not_bg]
            if len(kept_n) > 0:
                vals_n, _, _ = NDVI_CMAP.classify(kept_n)
                ndvi_val = round(float(vals_n.mean()), 3)

        # --- Sample NDRE ---
        ndre_val = np.nan
        ir_ndr, ic_ndr, inb_ndr = map_pixels(A_ndr, ocol_veg, orow_veg, H_ndr, W_ndr, shift_c_ndr, shift_r_ndr)
        if inb_ndr.any():
            cols_r = rgb_ndr[:, ir_ndr[inb_ndr], ic_ndr[inb_ndr]].T
            not_bg = ~np.all(cols_r.astype(np.int16) == bg, axis=1)
            kept_r = cols_r[not_bg]
            if len(kept_r) > 0:
                vals_r, _, _ = NDRE_CMAP.classify(kept_r)
                ndre_val = round(float(vals_r.mean()), 3)

        results.append((idx, osa_val, ndvi_val, ndre_val, osa_val, soil_frac))

    return results


if __name__ == '__main__':
    import time
    print("=== PIPELINE STEP 1: INDEX EXTRACTION (CANOPY MASK + OSAVI GATING) ===")

    # Verify source file presence
    for path in [csv_path, osavi_path, ndvi_path, ndre_path, ortho_path, masks_dir]:
        if not os.path.exists(path):
            print(f"Error: Required file or directory not found at: {path}")
            exit(1)

    # Load coordinates
    print("Loading plants coordinates...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} plant records.")

    # Parallelize extraction
    num_cores = min(multiprocessing.cpu_count(), 8)
    print(f"Parallelizing extraction across {num_cores} CPU cores...")

    # Split dataframe into chunks
    chunks = np.array_split(df, num_cores)
    args_list = []
    for chunk in chunks:
        args_list.append((
            chunk, osavi_path, ndvi_path, ndre_path, ortho_path, masks_dir
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
    osavi_raw = [x[4] for x in flattened]
    canopy_soil_frac = [x[5] for x in flattened]

    df['osavi'] = osavi_values
    df['ndvi'] = ndvi_values
    df['ndre'] = ndre_values
    df['osavi_raw'] = osavi_raw
    df['canopy_soil_frac'] = canopy_soil_frac

    # Force index values to NaN for coordinates labeled as 'NOISE'
    noise_mask = df['sector_label'] == 'NOISE'
    df.loc[noise_mask, ['osavi', 'ndvi', 'ndre', 'osavi_raw', 'canopy_soil_frac']] = np.nan

    # Save CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Extracted index values successfully. Output saved to: {output_csv_path}")

    # Check correlations
    sub = df.dropna(subset=['ndvi', 'osavi', 'ndre'])
    if len(sub) > 1:
        corr_ndvi_osavi = np.corrcoef(sub['ndvi'], sub['osavi'])[0, 1]
        corr_ndvi_ndre = np.corrcoef(sub['ndvi'], sub['ndre'])[0, 1]
        print(f"Pearson correlations (veg-only pixels):")
        print(f"  NDVI vs OSAVI: {corr_ndvi_osavi:.4f}")
        print(f"  NDVI vs NDRE: {corr_ndvi_ndre:.4f}")
    else:
        print("Could not compute correlations: no valid rows found.")

    print("Step 1 finished successfully.\n")
