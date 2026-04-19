import os
import numpy as np
import rasterio

def calculate_ndvi_for_patch(rgb_path, nir_path, output_path):
    with rasterio.open(rgb_path) as src_rgb:
        red = src_rgb.read(1).astype(float)
        meta = src_rgb.meta.copy()
        
    with rasterio.open(nir_path) as src_nir:
        nir = src_nir.read(1).astype(float)

    np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.nan_to_num((nir - red) / (nir + red), nan=-1.0, posinf=-1.0, neginf=-1.0)
    
    meta.update({"count": 1, "dtype": 'float32'})
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

def run(rgb_dir, nir_dir, ndvi_out_dir):
    os.makedirs(ndvi_out_dir, exist_ok=True)
    rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]
    print(f"THere are {len(rgb_files)} RGB patches. Doing Ndvi")
    
    processed_count, missing_nir_count = 0, 0

    for fname in rgb_files:
        rgb_path, nir_path = os.path.join(rgb_dir, fname), os.path.join(nir_dir, fname)
        if os.path.exists(nir_path):
            calculate_ndvi_for_patch(rgb_path, nir_path, os.path.join(ndvi_out_dir, fname.replace('.png', '.tif').replace('.jpg', '.tif')))
            processed_count += 1
        else: missing_nir_count += 1
            
        if processed_count % 500 == 0 and processed_count > 0: print(f"Processed {processed_count} patches...")

    print(f"Ndvi patches are {processed_count}, skipped {missing_nir_count} nir")