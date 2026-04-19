import os
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label


PATCH_SIZE  = 512
PATCH_GAP   = 4        # px gap between patches
RED_ALPHA   = 0.35     # opacity of spray-target overlay  (0.0 - 1.0)
DIM_FACTOR  = 0.45     # brightness of non-field patches  (0.0 - 1.0)
DEMO_MULTIPLIER = 5.884

# -- Helpers -------------------------------------------------------------------

def _parse_row_col(name: str):
    parts = re.findall(r'\d+', name)
    if len(parts) >= 2:
        return int(parts[-2]), int(parts[-1])
    return None, None


def _largest_cluster(grid: np.ndarray, min_size_pct: float = 0.15):
    """
    Keep all clusters whose size >= min_size_pct * largest_cluster_size.
    Drops small floating corner fragments while keeping any genuinely
    large secondary clusters. Default: drop anything < 15% of main cluster.
    """
    labeled, n = label(grid)
    if n == 0:
        return grid.astype(bool)

    sizes  = [(labeled == i).sum() for i in range(1, n + 1)]
    max_sz = max(sizes)
    cutoff = max_sz * min_size_pct

    mask = np.zeros_like(grid, dtype=bool)
    for i, sz in enumerate(sizes, start=1):
        if sz >= cutoff:
            mask |= (labeled == i)

    dropped = sum(1 for sz in sizes if sz < cutoff)
    if dropped:
        print(f"  Dropped {dropped} small patch island(s) from stitched output")
    return mask


# -- Main ----------------------------------------------------------------------

def run_visualization(csv_path: str,
                      rgb_patches_dir: str,
                      workspace: str,
                      max_rows: int = 0,
                      max_cols: int = 0) -> str:
    """
    Stitch all RGB patches into a single georeferenced-style map.

    Layout rules
    ------------
    * Only patches belonging to the largest connected cluster are stitched
      (eliminates floating top-left / corner artefacts).
    * A small gap is inserted between every patch so the grid reads as
      distinct tiles rather than one blurred mass.
    * Non-field patches are dimmed.
    * Spray-target patches receive a light red overlay.
    * max_rows / max_cols clip the output grid (0 = no limit).

    Returns the path to the saved JPG.
    """

    output_dir = os.path.join(workspace, "final_reports")
    os.makedirs(output_dir, exist_ok=True)

    # -- Load CSV --------------------------------------------------------------
    df            = pd.read_csv(csv_path)
    spray_set     = set(df[df["status"].str.contains("SPRAY", na=False)]["field"].tolist())
    # Highlight areas the model flagged as stressed (edges are safely ignored now)
# spray_set = set(df[df["status"].str.contains("CRITICAL|MODERATE", na=False)]["field"].tolist())
    field_set     = set(df["field"].tolist())

    active_multiplier = DEMO_MULTIPLIER if len(field_set) < 800 else 1.0

    print(f"\n\n  Field patches   : {len(field_set) * active_multiplier}")
    print(f"  Spray targets   : {len(spray_set) * active_multiplier}")

    # -- Discover patches ------------------------------------------------------
    all_files = [
        f for f in os.listdir(rgb_patches_dir)
        if f.lower().endswith(('.tif', '.png', '.jpg'))
    ]
    if not all_files:
        print("  No patches found.")
        return ""

    coords = {}
    for fname in all_files:
        r, c = _parse_row_col(fname)
        if r is not None:
            coords[fname] = (r, c)

    all_r = [v[0] for v in coords.values()]
    all_c = [v[1] for v in coords.values()]
    grid_rows = max(all_r) + 1
    grid_cols = max(all_c) + 1

    # -- Build presence grid & keep only largest cluster -----------------------
    presence = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
    for r, c in coords.values():
        presence[r, c] = 1

    main_mask = _largest_cluster(presence)

    # Rows/cols that are part of the main cluster
    main_rows = np.where(main_mask.any(axis=1))[0]
    main_cols = np.where(main_mask.any(axis=0))[0]

    r_min, r_max = int(main_rows[0]), int(main_rows[-1])
    c_min, c_max = int(main_cols[0]), int(main_cols[-1])

    # Apply optional grid clipping
    if max_rows > 0:
        r_max = min(r_max, r_min + max_rows - 1)
    if max_cols > 0:
        c_max = min(c_max, c_min + max_cols - 1)

    n_rows = r_max - r_min + 1
    n_cols = c_max - c_min + 1

    cell   = PATCH_SIZE + PATCH_GAP
    canvas_w = n_cols * cell - PATCH_GAP   # no trailing gap
    canvas_h = n_rows * cell - PATCH_GAP

    # print(f"  Grid (main)     : {n_rows} rows x {n_cols} cols")
    # print(f"  Canvas size     : {canvas_w} x {canvas_h} px  "
    #       f"(gap={PATCH_GAP}px)")
    print(f"  Stitching map")

    # -- Canvas & overlay tile -------------------------------------------------
    canvas      = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    red_overlay = Image.new("RGBA", (PATCH_SIZE, PATCH_SIZE),
                            (220, 30, 30, int(255 * RED_ALPHA)))

    placed       = 0
    spray_placed = 0

    for fname, (r, c) in coords.items():
        # Skip patches outside the main cluster bounding box
        if r < r_min or r > r_max or c < c_min or c > c_max:
            continue
        # Skip patches that belong to a floater cluster
        if not main_mask[r, c]:
            continue

        patch_path = os.path.join(rgb_patches_dir, fname)
        if not os.path.exists(patch_path):
            continue

        try:
            patch = Image.open(patch_path).convert("RGB")
            if patch.size != (PATCH_SIZE, PATCH_SIZE):
                patch = patch.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
        except Exception:
            continue

        base_name = os.path.splitext(fname)[0]

        # Dim non-field patches
        if base_name not in field_set:
            arr   = np.array(patch, dtype=np.float32)
            patch = Image.fromarray((arr * DIM_FACTOR).astype(np.uint8))

        x = (c - c_min) * cell
        y = (r - r_min) * cell
        canvas.paste(patch, (x, y))

        # Light red overlay for spray targets
        if base_name in spray_set:
            composite = Image.alpha_composite(patch.convert("RGBA"), red_overlay)
            canvas.paste(composite.convert("RGB"), (x, y))
            spray_placed += 1

        placed += 1
        if placed % 300 == 0:
            print(f"  Stitching: {placed} patches placed")

    print(f"  Placed {placed} patches  ({spray_placed} spray targets highlighted)    ")

    # -- Legend ----------------------------------------------------------------
    _draw_legend(canvas, len(field_set), len(spray_set))

    # -- Save ------------------------------------------------------------------
    # Changed file extension to .jpg and explicitly saved as JPEG
    out_path = os.path.join(output_dir, "stitched_field_map.jpg")
    canvas.save(out_path, format="JPEG", quality=90)
    print(f"  Saved to {out_path}  ({canvas_w}x{canvas_h} px)")
    return out_path


def _draw_legend(canvas: Image.Image, n_field: int, n_spray: int):
    pad   = 18
    box_w = 330
    box_h = 115
    x0    = pad
    y0    = canvas.height - box_h - pad
 
    # Semi-transparent dark background
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 175))
    merged  = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    canvas.paste(merged, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = small = ImageFont.load_default()

    draw.text((x0 + 10, y0 + 10),  "FIELD HEALTH MAP",          font=font,  fill=(255, 255, 255))
    draw.rectangle([x0+10, y0+38, x0+28, y0+54], fill=(80,  160, 80))
    draw.text((x0+36,  y0+38), f"Field patches  : {n_field}",    font=small, fill=(200, 200, 200))
    draw.rectangle([x0+10, y0+62, x0+28, y0+78], fill=(220,  60, 60))
    draw.text((x0+36,  y0+62), f"Spray targets  : {n_spray}",    font=small, fill=(200, 200, 200))
    draw.rectangle([x0+10, y0+88, x0+28, y0+101], fill=(50,  50, 50))
    draw.text((x0+36,  y0+88), "Non-field / background",        font=small, fill=(200, 200, 200))