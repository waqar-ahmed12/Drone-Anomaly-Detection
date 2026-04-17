import os
import gc      
import torch

import step1_patching as patching
import step2_classification as classification
import step3_ndvi as ndvi
import step4_inference as inference
import step5_visualization as visualization

# --- 1. SET YOUR MASTER PATHS HERE ---
RGB_IMAGE = r"new data\big images\Rgb.tif"
NIR_IMAGE = r"new data\big images\Nir.tif"
CLASSIFIER_WEIGHTS = r"weights\field_model_pytorch.pth"
AE_WEIGHTS = r"weights\multimodal_ae_epoch_15.pth" 

# Where you want the folders to go
WORKSPACE = r"pipeline_workspace"




if __name__ == "__main__":
    print("Start pipeline")

    # Setup automatic directories
    rgb_patches_dir = os.path.join(WORKSPACE, "Rgb_patches")
    nir_patches_dir = os.path.join(WORKSPACE, "Nir_patches")
    ndvi_patches_dir = os.path.join(WORKSPACE, "Ndvi_patches")
    field_txt = os.path.join(WORKSPACE, "field_list.txt")
    non_field_txt = os.path.join(WORKSPACE, "non_field_list.txt")

    
    

    print("\n1: Breaking map into patches")
    patching.generate_patches(RGB_IMAGE, rgb_patches_dir, patch_size=512, rotation=90)
    patching.generate_patches(NIR_IMAGE, nir_patches_dir, patch_size=512, rotation=90)

    print("\n2: Field classifier")
    classification.run(rgb_patches_dir, CLASSIFIER_WEIGHTS, field_txt, non_field_txt)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache() #garbage and cache cleaning

    print("\n3: Calculating Ndvi")
    ndvi.run(rgb_patches_dir, nir_patches_dir, ndvi_patches_dir)

    print("\n4: Autoencoder inference")
    csv_path = inference.run_pipeline_batch(field_txt, rgb_patches_dir, ndvi_patches_dir, AE_WEIGHTS, WORKSPACE)
    # print(csv_path)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # csv_path = r"pipeline_workspace\field_health_summary.csv"
    if csv_path:
        print("\n5: Visualization")
        
        # visualization.run_visualization(csv_path, rgb_patches_dir, ndvi_patches_dir, AE_WEIGHTS, WORKSPACE)
        visualization.run_visualization(csv_path, rgb_patches_dir, WORKSPACE)
        print("\n Complete. pipeline_workspace has final images.")
    else:
        print("\n stopped: No valid patches found")