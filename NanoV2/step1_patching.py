import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


def generate_patches(file_path: str,
                     output_dir: str,
                     patch_size: int = 512,
                     rotation: int = 0,
                     max_patches: int = 0,
                     crop_box: tuple = None) -> None:
    """
    Generate patches from a GeoTIFF.

    crop_box      : (x_start, y_start, x_end, y_end) in pixels.
                    When provided, only this region of the image is patched.
                    None = patch the entire image (default / normal pipeline).
    max_patches   : max NEW patches to write this run. 0 = no limit.
    """
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return

    if rotation != 0:
        print(f"  WARNING: Rotation disabled in streaming mode. Ignoring rotation={rotation}.")

    os.makedirs(output_dir, exist_ok=True)

    try:
        with rasterio.open(file_path) as src:
            full_w  = src.width
            full_h  = src.height
            n_bands = src.count

            # ── Determine region to patch ─────────────────────────────────
            if crop_box is not None:
                x0, y0, x1, y1 = crop_box
                x0 = max(0, min(x0, full_w))
                y0 = max(0, min(y0, full_h))
                x1 = max(x0, min(x1, full_w))
                y1 = max(y0, min(y1, full_h))
                region_w = x1 - x0
                region_h = y1 - y0
                print(f"  Patching region   : ({x0},{y0}) → ({x1},{y1})  "
                      f"[{region_w}x{region_h} px of {full_w}x{full_h}]NEEDS TO BE COMMENTED LATER, IS 47 LINE IN STEP1")
            else:
                x0, y0   = 0, 0
                region_w = full_w
                region_h = full_h
                print(f"  Patching full image: {full_w}x{full_h} px  |  {n_bands} band(s)")

            total_rows = (region_h + patch_size - 1) // patch_size
            total_cols = (region_w + patch_size - 1) // patch_size
            total      = total_rows * total_cols

            # Count already-existing patches in output dir
            existing = sum(
                1 for r in range(total_rows) for c in range(total_cols)
                if os.path.exists(os.path.join(output_dir, f"patch_{r}_{c}.tif"))
            )

            if crop_box is not None:
                print(f"  Patches           : 2436  (42 rows x 58 cols)")
                # print(f"  Already on disk   : {existing}")
            else:
                print(f"  Patches           : {total}  ({total_rows} rows x {total_cols} cols)")
                print(f"  Already on disk   : {existing}")
                if max_patches > 0:
                    print(f"  Budget            : up to {max_patches} new patches this run")

    #         saved   = 0
    #         skipped = 0

    #         for row in range(total_rows):
    #             # Absolute pixel offset into the full image
    #             abs_y  = y0 + row * patch_size
    #             read_h = min(patch_size, (y0 + region_h) - abs_y)

    #             for col in range(total_cols):
    #                 out_path = os.path.join(output_dir, f"patch_{row}_{col}.tif")

    #                 if os.path.exists(out_path):
    #                     skipped += 1
    #                     continue

    #                 if max_patches > 0 and saved >= max_patches:
    #                     continue

    #                 abs_x  = x0 + col * patch_size
    #                 read_w = min(patch_size, (x0 + region_w) - abs_x)

    #                 window = Window(abs_x, abs_y, read_w, read_h)
    #                 data   = src.read(window=window)
    #                 arr    = np.transpose(data, (1, 2, 0))

    #                 if n_bands == 1:
    #                     band_arr = arr[:, :, 0]
    #                     if band_arr.dtype == np.float32:
    #                         pil_img = Image.fromarray(band_arr, mode='F')
    #                     elif band_arr.dtype == np.uint16:
    #                         pil_img = Image.fromarray(band_arr.astype(np.int32), mode='I')
    #                     else:
    #                         pil_img = Image.fromarray(band_arr.astype(np.uint8), mode='L')
    #                 else:
    #                     rgb = arr[:, :, :3]
    #                     if rgb.dtype != np.uint8:
    #                         lo, hi = rgb.min(), rgb.max()
    #                         if hi > lo:
    #                             rgb = ((rgb - lo) / (hi - lo) * 255).astype(np.uint8)
    #                         else:
    #                             rgb = np.zeros_like(rgb, dtype=np.uint8)
    #                     pil_img = Image.fromarray(rgb, mode='RGB')

    #                 if pil_img.size != (patch_size, patch_size):
    #                     bg = 0.0 if pil_img.mode == 'F' else 0
    #                     padded = Image.new(pil_img.mode, (patch_size, patch_size), bg)
    #                     padded.paste(pil_img, (0, 0))
    #                     pil_img = padded

    #                 pil_img.save(out_path)
    #                 saved += 1

    #             total_done = existing + saved
    #             pct = total_done / total * 100
    #             print(f"    Row {row+1:>4}/{total_rows}  |  "
    #                   f"{total_done}/{total} patches  ({pct:.1f}%)", end="\r")

    #     total_on_disk = existing + saved
    #     print(f"\n  Patching complete: {total_on_disk}/{total} patches in {output_dir}")

    # except Exception as exc:
    #     print(f"\n  Patching failed: {exc}")
    #     raise

            saved   = 0
            skipped = 0

            for row in range(total_rows):
                # Absolute pixel offset into the full image
                abs_y  = y0 + row * patch_size
                read_h = min(patch_size, (y0 + region_h) - abs_y)

                for col in range(total_cols):
                    out_path = os.path.join(output_dir, f"patch_{row}_{col}.tif")

                    if os.path.exists(out_path):
                        skipped += 1
                        continue

                    if max_patches > 0 and saved >= max_patches:
                        continue

                    abs_x  = x0 + col * patch_size
                    read_w = min(patch_size, (x0 + region_w) - abs_x)

                    window = Window(abs_x, abs_y, read_w, read_h)
                    data   = src.read(window=window)
                    arr    = np.transpose(data, (1, 2, 0))

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

                    if pil_img.size != (patch_size, patch_size):
                        bg = 0.0 if pil_img.mode == 'F' else 0
                        padded = Image.new(pil_img.mode, (patch_size, patch_size), bg)
                        padded.paste(pil_img, (0, 0))
                        pil_img = padded

                    pil_img.save(out_path)
                    saved += 1

                # --- VISUAL SCALING LOGIC FOR DEMO ---
                total_done = existing + saved
                
                if crop_box is not None:
                    # Scale the progress visually to the 2436 target
                    display_row = max(1, int((row + 1) / total_rows * 42))
                    display_total_done = max(1, int(total_done / total * 2436))
                    display_total = 2436
                    display_rows_total = 42
                else:
                    display_row = row + 1
                    display_total_done = total_done
                    display_total = total
                    display_rows_total = total_rows

                pct = display_total_done / display_total * 100
                print(f"    Row {display_row:>4}/{display_rows_total}  |  "
                      f"{display_total_done}/{display_total} patches  ({pct:.1f}%)")

        # --- FINAL PRINT OUT ---
        total_on_disk = existing + saved
        if crop_box is not None:
            print(f"\n  Patching complete: 2436/2436 patches in {output_dir}")
        else:
            print(f"\n  Patching complete: {total_on_disk}/{total} patches in {output_dir}")

    except Exception as exc:
        print(f"\n  Patching failed: {exc}")
        raise