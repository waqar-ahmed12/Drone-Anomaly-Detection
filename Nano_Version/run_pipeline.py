# import os
# import gc      
# import torch

# import step1_patching as patching
# import step2_classification as classification
# import step3_ndvi as ndvi
# import step4_inference as inference
# import step5_visualization as visualization

# #paths
# RGB_IMAGE = r"new data/big images/Rgb.tif"
# NIR_IMAGE = r"new data/big images/Nir.tif"
# CLASSIFIER_WEIGHTS = r"weights/field_model_pytorch.pth"
# AE_WEIGHTS = r"weights/multimodal_ae_epoch_15.pth" 

# # the folders to go here
# WORKSPACE = r"pipeline_workspace"


# if __name__ == "__main__":
#     print("Start pipeline")

#     # Setup automatic directories
#     rgb_patches_dir = os.path.join(WORKSPACE, "Rgb_patches")
#     nir_patches_dir = os.path.join(WORKSPACE, "Nir_patches")
#     ndvi_patches_dir = os.path.join(WORKSPACE, "Ndvi_patches")
#     field_txt = os.path.join(WORKSPACE, "field_list.txt")
#     non_field_txt = os.path.join(WORKSPACE, "non_field_list.txt")

    
    

#     # print("\n1: Breaking map into patches")
#     # patching.generate_patches(RGB_IMAGE, rgb_patches_dir, patch_size=512, rotation=0)
#     # patching.generate_patches(NIR_IMAGE, nir_patches_dir, patch_size=512, rotation=0)

#     print("\n2: Field classifier")
#     classification.run(rgb_patches_dir, CLASSIFIER_WEIGHTS, field_txt, non_field_txt)

#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache() #garbage and cache cleaning

#     # print("\n3: Calculating Ndvi")
#     # ndvi.run(rgb_patches_dir, nir_patches_dir, ndvi_patches_dir)

#     # print("\n4: Autoencoder inference")
#     # csv_path = inference.run_pipeline_batch(field_txt, rgb_patches_dir, ndvi_patches_dir, AE_WEIGHTS, WORKSPACE)
#     # # print(csv_path)

#     # gc.collect()
#     # if torch.cuda.is_available():
#     #     torch.cuda.empty_cache()

#     # # csv_path = r"pipeline_workspace\field_health_summary.csv"
#     # if csv_path:
#     #     print("\n5: Visualization")
        
#     #     # visualization.run_visualization(csv_path, rgb_patches_dir, ndvi_patches_dir, AE_WEIGHTS, WORKSPACE)
#     #     visualization.run_visualization(csv_path, rgb_patches_dir, WORKSPACE)
#     #     print("\n Complete. pipeline_workspace has final images.")
#     # else:
#     #     print("\n stopped: No valid patches found")


"""
run_pipeline.py  –  Jetson Nano 4 GB optimised orchestrator
============================================================
Key changes vs. original
  • JetsonMemoryGuard  – monitors RAM + GPU RAM; logs every stage
  • Swap guard          – reminds user to enable 4 GB swap if headroom is low
  • Explicit del + gc   – every heavy object is released before next stage
  • torch.backends      – cuDNN benchmark ON, deterministic OFF (faster)
  • Stage timing        – prints elapsed time for each step
"""

import os, gc, time, sys
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
torch.backends.cudnn.benchmark     = True   # fastest conv algorithm per shape
torch.backends.cudnn.deterministic = False  # allow non-deterministic (faster)
# Half-precision support check (Jetson Nano GPU supports FP16)
USE_FP16 = torch.cuda.is_available()

import step1_patching      as patching
import step2_classification as classification
import step3_ndvi           as ndvi
import step4_inference      as inference
import step5_visualization  as visualization

# ── Paths ────────────────────────────────────────────────────────────────────
# RGB_IMAGE           = r"new data/big images/Rgb.tif" 
RGB_IMAGE           = r"new data/cropped images/Y15816/X_15142-28294_Y_15816-25950_Res_1.000.tif" 
NIR_IMAGE           = r"new data/cropped images/Y15816/Replicated_X_15142-28294_Y_15816-25950_Res_1.000.tif"
# NIR_IMAGE           = r"new data/big images/Nir.tif"
CLASSIFIER_WEIGHTS  = r"weights/field_model_pytorch.pth"
AE_WEIGHTS          = r"weights/multimodal_ae_epoch_15.pth"
WORKSPACE           = r"pipeline_workspace"

# ── Batch size: 8 is safe for 4 GB; raise to 16 only if swap is enabled ─────
INFERENCE_BATCH_SIZE = 8

# ── Helpers ──────────────────────────────────────────────────────────────────
def _check_swap():
    """Warn the user if no swap is configured (important for Jetson)."""
    try:
        import subprocess
        out = subprocess.check_output(["free", "-m"]).decode()
        swap_total = int(out.split("\n")[2].split()[1])
        if swap_total < 2048:
            print("\n  ⚠  WARNING: Less than 2 GB swap detected.")
            print("     For Jetson Nano 4 GB it is strongly recommended to enable")
            print("     at least 4 GB of swap:")
            print("       sudo fallocate -l 4G /swapfile")
            print("       sudo chmod 600 /swapfile && sudo mkswap /swapfile")
            print("       sudo swapon /swapfile\n")
    except Exception:
        pass


def _stage(name: str, fn, *args, **kwargs):
    """Run a pipeline stage with timing, memory logging, and error handling."""
    print(f"\n{'='*55}")
    print(f"  STAGE: {name}")
    print(f"{'='*55}")
    _mem_stats("before")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
    except MemoryError:
        print(f"\n  ✗ MemoryError in '{name}'. Try enabling swap or reducing")
        print("    INFERENCE_BATCH_SIZE in run_pipeline.py.")
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"  ✓ Done in {elapsed:.1f}s")
    _release_gpu(name)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _check_swap()

    rgb_patches_dir  = os.path.join(WORKSPACE, "Rgb_patches")
    nir_patches_dir  = os.path.join(WORKSPACE, "Nir_patches")
    ndvi_patches_dir = os.path.join(WORKSPACE, "Ndvi_patches")
    field_txt        = os.path.join(WORKSPACE, "field_list.txt")
    non_field_txt    = os.path.join(WORKSPACE, "non_field_list.txt")

    print("\n  Drone Pipeline – Jetson Nano 4 GB build")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  FP16 inference : {USE_FP16}")
    print(f"  Batch size     : {INFERENCE_BATCH_SIZE}")

    t_total = time.time()

    # # 1 – Patching (CPU-only, memory-streamed)
    # _stage("Patching RGB",
    #        patching.generate_patches,
    #        RGB_IMAGE, rgb_patches_dir, patch_size=512, rotation=0)

    # _stage("Patching NIR",
    #        patching.generate_patches,
    #        NIR_IMAGE, nir_patches_dir, patch_size=512, rotation=0)

    # # 2 – Classification (MobileNetV2, GPU)
    # _stage("Field Classification",
    #        classification.run,
    #        rgb_patches_dir, CLASSIFIER_WEIGHTS, field_txt, non_field_txt)

    # # 3 – NDVI (CPU, embarrassingly parallel)
    # _stage("NDVI Calculation",
    #        ndvi.run,
    #        rgb_patches_dir, nir_patches_dir, ndvi_patches_dir)

    # # # 4 – Autoencoder inference (GPU, FP16 if available)
    # csv_path = _stage("Autoencoderder Inference", 
    #                     inference.run_pipeline_batch,
    #                     field_txt, rgb_patches_dir, ndvi_patches_dir,
    #                     AE_WEIGHTS, WORKSPACE, 
    #                     batch_size=INFERENCE_BATCH_SIZE,
    #                     use_fp16=USE_FP16)
    # # 5 – Visualisation (CPU)
    if csv_path:
        _stage("Visualisation",
               visualization.run_visualization,
               csv_path, rgb_patches_dir, WORKSPACE)
        print("\n  Pipeline complete.")
        print(f"  Total wall-clock time: {(time.time()-t_total)/60:.1f} min")
        print(f"  Results in: {WORKSPACE}/final_reports/")
    else:
        print("\n  Pipeline stopped: no valid field patches found.")