# import os
# import rasterio
# from rasterio.windows import Window
# import numpy as np
# from PIL import Image
# import gc  # <-- Added garbage collection

# def generate_patches(file_path, output_dir, patch_size=512, rotation=0):
#     if not os.path.exists(file_path):
#         print(f"Error: Could not find file at {file_path}")
#         return

#     print(f"Processing in EXTREME low-memory mode: {file_path}")
#     os.makedirs(output_dir, exist_ok=True)

#     # 1. Chokehold on GDAL cache (limit to 64MB so the Nano doesn't crash)
#     with rasterio.Env(GDAL_CACHEMAX=64000000):
#         with rasterio.open(file_path) as src:
#             width = src.width
#             height = src.height

#             total_rows = (height + patch_size - 1) // patch_size
#             total_cols = (width + patch_size - 1) // patch_size
#             print(f"There are {total_rows * total_cols} patches to extract.")

#             for row in range(total_rows):
#                 for col in range(total_cols):
#                     window = Window(col * patch_size, row * patch_size, patch_size, patch_size)
                    
#                     data = src.read(window=window)
#                     data = np.transpose(data, (1, 2, 0))

#                     if data.shape[2] == 1:
#                         data = data.squeeze(axis=2)

#                     h, w = data.shape[:2]
#                     if h != patch_size or w != patch_size:
#                         if len(data.shape) == 3: 
#                             padded = np.zeros((patch_size, patch_size, data.shape[2]), dtype=data.dtype)
#                             padded[:h, :w, :] = data
#                         else: 
#                             padded = np.zeros((patch_size, patch_size), dtype=data.dtype)
#                             padded[:h, :w] = data
#                         data = padded

#                     patch_img = Image.fromarray(data)
#                     patch_filename = f"patch_{row}_{col}.tif"
#                     patch_img.save(os.path.join(output_dir, patch_filename))

#                     # 2. Explicitly nuke the variables from RAM
#                     del data
#                     del patch_img
#                     if 'padded' in locals():
#                         del padded
                
#                 # 3. Force Python to empty the trash after every row
#                 gc.collect()

#             print(f"Finished extracting to: {output_dir}")



"""
step1_patching.py  –  Jetson Nano optimised
============================================
Key changes vs. original
  • Streaming row-by-row crop  – the full rotated image is NEVER held in RAM
    twice; each row of patches is cropped, saved, then discarded.
  • Context-manager for Image  – ensures the file handle is closed promptly.
  • Region-based open (via box= crop trick) – avoids holding the full NumPy
    array when only a slice is needed.
  • Progress bar uses \r (single line) to avoid log spam.
  • Rotation is done lazily only when needed.
"""
"""
step1_patching.py  –  Streaming-safe for large GeoTIFFs (30k x 25k px)
=======================================================================
Strategy: use rasterio to read ONE 512x512 window at a time directly
from disk. The full image is NEVER loaded into RAM. Peak RAM per patch
is ~0.75 MB (512x512x3 bytes) regardless of source image size.

Rotation: skipped for now (pass rotation=0 or just don't pass it).
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


def generate_patches(file_path: str,
                     output_dir: str,
                     patch_size: int = 512,
                     rotation: int = 0) -> None:

    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return

    if rotation != 0:
        print(f"  WARNING: Rotation disabled in streaming mode. Ignoring rotation={rotation}.")

    print(f"  Patching (streaming): {file_path}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with rasterio.open(file_path) as src:
            width      = src.width
            height     = src.height
            n_bands    = src.count
            total_rows = (height + patch_size - 1) // patch_size
            total_cols = (width  + patch_size - 1) // patch_size
            total      = total_rows * total_cols

            print(f"    Image : {width} x {height} px  |  {n_bands} band(s)")
            print(f"    Patches: {total}  ({total_rows} rows x {total_cols} cols)")
            print(f"    Peak RAM per patch: ~{patch_size*patch_size*n_bands/1e6:.2f} MB")

            saved = 0

            for row in range(total_rows):
                upper  = row * patch_size
                read_h = min(patch_size, height - upper)

                for col in range(total_cols):
                    left   = col * patch_size
                    read_w = min(patch_size, width - left)

                    # Read ONLY this window from disk — no full image in RAM
                    window = Window(left, upper, read_w, read_h)
                    data   = src.read(window=window)   # (bands, h, w)
                    arr    = np.transpose(data, (1, 2, 0))  # -> (h, w, bands)

                    # Build PIL image based on band count
                    if n_bands == 1:
                        band_arr = arr[:, :, 0]
                        if band_arr.dtype == np.float32:
                            pil_img = Image.fromarray(band_arr, mode='F')
                        elif band_arr.dtype == np.uint16:
                            pil_img = Image.fromarray(band_arr.astype(np.int32), mode='I')
                        else:
                            pil_img = Image.fromarray(band_arr.astype(np.uint8), mode='L')
                    else:
                        rgb = arr[:, :, :3]
                        if rgb.dtype != np.uint8:
                            lo, hi = rgb.min(), rgb.max()
                            if hi > lo:
                                rgb = ((rgb - lo) / (hi - lo) * 255).astype(np.uint8)
                            else:
                                rgb = np.zeros_like(rgb, dtype=np.uint8)
                        pil_img = Image.fromarray(rgb, mode='RGB')

                    # Pad edge patches to full patch_size x patch_size
                    if pil_img.size != (patch_size, patch_size):
                        bg = 0.0 if pil_img.mode == 'F' else 0
                        padded = Image.new(pil_img.mode, (patch_size, patch_size), bg)
                        padded.paste(pil_img, (0, 0))
                        pil_img = padded

                    pil_img.save(os.path.join(output_dir, f"patch_{row}_{col}.tif"))
                    saved += 1

                pct = saved / total * 100
                print(f"    Row {row+1:>4}/{total_rows}  |  {saved}/{total} patches  ({pct:.1f}%)", end="\r")

        print(f"\n    Done: {saved} patches saved -> {output_dir}")

    except Exception as exc:
        print(f"\n  Patching failed: {exc}")
        raise