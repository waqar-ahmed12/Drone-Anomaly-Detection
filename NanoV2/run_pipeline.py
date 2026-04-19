import os, gc, time, sys, json, argparse
import torch

# ── Optional: print memory stats on Jetson ──────────────────────────────────
def _mem_stats(label: str):
    try:
        import subprocess
        free_mb = int(subprocess.check_output(
            ["free", "-m"]).decode().split()[8])
        print(f"  [{label}] System free RAM: {free_mb} MB", flush=True)
    except Exception:
        pass
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e6
        total = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"  [{label}] GPU VRAM: {used:.0f} / {total:.0f} MB used", flush=True)

def _release_gpu(label: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    _mem_stats(label or "after release")

# ── Jetson-specific torch settings ──────────────────────────────────────────
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False
USE_FP16 = torch.cuda.is_available()

import step1_patching      as patching
import step2_classification as classification
import step3_ndvi           as ndvi
import step4_inference      as inference
import step5_visualization  as visualization

# ── Load config ──────────────────────────────────────────────────────────────
CONFIG_PATH = "pipeline_config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"  Config not found: {CONFIG_PATH}. Using defaults.")
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_demo_config():
    """Load demo_config.py as a module and return it as a plain dict."""
    import importlib.util
    spec   = importlib.util.spec_from_file_location("demo_config", "demo_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ── Paths ────────────────────────────────────────────────────────────────────
CLASSIFIER_WEIGHTS  = r"weights/field_model_pytorch.pth"
AE_WEIGHTS          = r"weights/multimodal_ae_epoch_15.pth"
WORKSPACE           = r"pipeline_workspace"
INFERENCE_BATCH_SIZE = 8

# ── Helpers ──────────────────────────────────────────────────────────────────
def _check_swap():
    try:
        import subprocess
        out = subprocess.check_output(["free", "-m"]).decode()
        swap_total = int(out.split("\n")[2].split()[1])
        if swap_total < 2048:
            print("\nthere is less than 2 GB swap detected.")
            
    except Exception:
        pass

def _stage(name: str, fn, *args, **kwargs):
    print(f"  FIle : {name}")
    _mem_stats("before")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
    except MemoryError:
        print(f"\n   MemoryError in '{name}'. Try enabling swap or reducing")
        print("    INFERENCE_BATCH_SIZE in run_pipeline.py.")
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s")
    _release_gpu(name)
    return result

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _check_swap()

    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Drone crop health pipeline")
    parser.add_argument("-y", action="store_true",
                        help="Use demo configuration (demo_config.py)")
    args = parser.parse_args()

    DEMO_MODE = args.y

    if DEMO_MODE:
        d = load_demo_config()
        RGB_IMAGE         = d.RGB_IMAGE
        NIR_IMAGE         = d.NIR_IMAGE
        PRELOADED_PATCHES = d.PRELOADED_PATCHES
        STEP1_MAX_PATCHES = d.STEP1_MAX_PATCHES
        STEP2_MAX_PATCHES = d.STEP2_MAX_PATCHES
        STEP3_MAX_PATCHES = d.STEP3_MAX_PATCHES
        STEP4_MAX_PATCHES = d.STEP4_MAX_PATCHES
        STEP5_MAX_ROWS    = d.STEP5_MAX_ROWS
        STEP5_MAX_COLS    = d.STEP5_MAX_COLS
        PATCHING_CROP     = d.PATCHING_CROP
        WORKSPACE         = d.DEMO_WORKSPACE
    else:
        cfg = load_config()
        RGB_IMAGE         = cfg.get("rgb_image",  r"new data/big images/Rgb.tif")
        NIR_IMAGE         = cfg.get("nir_image",  r"new data/big images/Nir.tif")
        PRELOADED_PATCHES = cfg.get("preloaded_patches", 0)
        STEP1_MAX_PATCHES = cfg.get("step1_max_patches", 0)
        STEP2_MAX_PATCHES = cfg.get("step2_max_patches", 0)
        STEP3_MAX_PATCHES = cfg.get("step3_max_patches", 0)
        STEP4_MAX_PATCHES = cfg.get("step4_max_patches", 0)
        STEP5_MAX_ROWS    = cfg.get("step5_max_rows", 0)
        STEP5_MAX_COLS    = cfg.get("step5_max_cols", 0)
        PATCHING_CROP     = None   # full image

    rgb_patches_dir  = os.path.join(WORKSPACE, "Rgb_patches")
    nir_patches_dir  = os.path.join(WORKSPACE, "Nir_patches")
    ndvi_patches_dir = os.path.join(WORKSPACE, "Ndvi_patches")
    field_txt        = os.path.join(WORKSPACE, "field_list.txt")
    non_field_txt    = os.path.join(WORKSPACE, "non_field_list.txt")

    print("\n  Start the pipline here:")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  FP16 inference : {USE_FP16}")
    print(f"  Batch size     : {INFERENCE_BATCH_SIZE}")
    print(f"  RGB input      : {RGB_IMAGE}")
    print(f"  NIR input      : {NIR_IMAGE}\n\n")
    # if PRELOADED_PATCHES > 0:
    #     print(f"  Resuming from  : {PRELOADED_PATCHES} existing patches")

    t_total = time.time()

    # 1 – Patching (CPU-only, memory-streamed, skips existing patches)
    # _stage("Patching RGB",
    #        patching.generate_patches,
    #        RGB_IMAGE, rgb_patches_dir, patch_size=512, rotation=0,
    #        max_patches=STEP1_MAX_PATCHES, crop_box=PATCHING_CROP)

    # _stage("Patching NIR",
    #        patching.generate_patches,
    #        NIR_IMAGE, nir_patches_dir, patch_size=512, rotation=0,
    #        max_patches=STEP1_MAX_PATCHES, crop_box=PATCHING_CROP)

    # # 2 – Classification (MobileNetV2, GPU)
    #     # Runs only on patches beyond preloaded_patches, appends to existing lists
    # _stage("Field Classification",
    #        classification.run,
    #        rgb_patches_dir, CLASSIFIER_WEIGHTS, field_txt, non_field_txt,
    #        preloaded_patches=PRELOADED_PATCHES,
    #        max_patches=STEP2_MAX_PATCHES)

    # # 3 – NDVI (CPU, field patches only, skips existing)
    # _stage("NDVI Calculation",
    #        ndvi.run,
    #        rgb_patches_dir, nir_patches_dir, ndvi_patches_dir,
    #        field_list_path=field_txt,
    #        max_patches=STEP3_MAX_PATCHES)

    # # 4 – Autoencoder inference (GPU, FP16 if available)
    # csv_path = _stage("Autoencoder Inference",
    #                   inference.run_pipeline_batch,
    #                   field_txt, rgb_patches_dir, ndvi_patches_dir,
    #                   AE_WEIGHTS, WORKSPACE,
    #                   batch_size=INFERENCE_BATCH_SIZE,
    #                   use_fp16=USE_FP16,
    #                   max_patches=STEP4_MAX_PATCHES,
                    #   )
    csv_path = "pipeline_workspace/field_health_summary.csv"

    if csv_path:
        # 5 – Visualisation (CPU, stitches full patch grid with overlays)
        _stage("Visualisation",
               visualization.run_visualization,
               csv_path, rgb_patches_dir, WORKSPACE,
               max_rows=STEP5_MAX_ROWS,
               max_cols=STEP5_MAX_COLS)

        print("\n  Pipeline complete.")
        print(f"  Total wall-clock time: {(time.time()-t_total)/60:.1f} min")
        print(f"  Results in: {WORKSPACE}/final_reports/")
    else:
        print("\n  Pipeline stopped: no valid field patches found.")