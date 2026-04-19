"""
Demo configuration — tuned values for live presentation.

This file is only imported when the pipeline is run with the -y flag:
    python run_pipeline.py -y

The normal pipeline (python run_pipeline.py) never touches this file.
All crop coordinates are in pixels relative to the full source image.
"""

# ── Input images ──────────────────────────────────────────────────────────────
RGB_IMAGE = r"new data/big images/Rgb.tif"
NIR_IMAGE = r"new data/big images/Nir.tif"

# ── Pixel crop box for patching: (x_start, y_start, x_end, y_end) ────────────
# Step 1 only patches this region of the full image.
# Tune these to control how long patching takes on demo day.
# Use your GUI crop tool to find good coordinates, then paste them here.
#
#   Full image is 30000 x 25000 px → 3060 patches total
#   A 6000 x 5000 px crop → ~110 patches → ~7s patching on Nano
#   A 8000 x 7000 px crop → ~210 patches → ~12s patching on Nano
#
# PATCHING_CROP = (0, 0, 8000, 7000)   # (x0, y0, x1, y1) — tune this
PATCHING_CROP = (16689, 15627, 28073, 24529)

# ── Separate output dirs so demo patches never mix with preloaded ones ────────
DEMO_WORKSPACE    = r"workspace"                  # <-- Changed to dummy
DEMO_RGB_PATCHES  = r"workspace/Rgb_patches"      # <-- Changed to dummy
DEMO_NIR_PATCHES  = r"workspace/Nir_patches"

# ── Per-step patch budgets ────────────────────────────────────────────────────
STEP1_MAX_PATCHES = 0    # 0 = patch entire crop box, no extra limit
STEP2_MAX_PATCHES = 150  # classify up to 150 new patches (~33s on Nano)
STEP3_MAX_PATCHES = 0    # NDVI is fast, run on all field patches
STEP4_MAX_PATCHES = 0    # inference on full preloaded field list

# ── Preloaded patch count (from PC full run) ──────────────────────────────────
PRELOADED_PATCHES = 3060

# ── Visualization grid clip ───────────────────────────────────────────────────
STEP5_MAX_ROWS = 0   # 0 = full grid
STEP5_MAX_COLS = 0