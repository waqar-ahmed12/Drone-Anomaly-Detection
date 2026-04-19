# import os
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# from scipy import ndimage
# from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion
# import re

# # Logic Tuning
# DL_THRESHOLD    = 0.3
# DL_WEIGHT       = 0.50
# COLOR_WEIGHT    = 0.25
# TEXTURE_WEIGHT  = 0.15
# CONTEXT_WEIGHT  = 0.10
# FINAL_THRESHOLD = 0.45 
# CLOSING_RADIUS  = 4
# MAX_HOLE_SIZE   = 300
# MIN_ISLAND_SIZE = 8
# MIN_FIELD_CLUSTER = 40
# EROSION_RADIUS  = 2

# def score_color(img_np):
#     r, g, b = img_np[:, :, 0].astype(float), img_np[:, :, 1].astype(float), img_np[:, :, 2].astype(float)
#     soil_score = np.clip((r - b) / 128.0, 0, 1)          
#     veg_score = np.clip((g - np.maximum(r, b)) / 64.0, 0, 1)
#     grey_penalty = 1.0 - np.clip(1.0 - (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 128.0, 0, 1)
#     return float(np.clip((0.6 * soil_score + 0.4 * veg_score - 0.3 * grey_penalty).mean(), 0, 1))

# def score_texture(img_np):
#     grey = img_np.mean(axis=2).astype(float)
#     from scipy.ndimage import maximum_filter, minimum_filter
#     local_range = (maximum_filter(grey, size=5) - minimum_filter(grey, size=5)) / 255.0      
#     return float(np.clip((float(local_range.mean()) - 0.04) / 0.16, 0, 1))

# def score_context(r, c, conf_grid, radius=2):
#     r0, r1 = max(0, r - radius), min(conf_grid.shape[0], r + radius + 1)
#     c0, c1 = max(0, c - radius), min(conf_grid.shape[1], c + radius + 1)
#     valid = conf_grid[r0:r1, c0:c1][conf_grid[r0:r1, c0:c1] > 0]
#     return float(valid.mean()) if len(valid) > 0 else 0.5

# def make_disk_kernel(radius):
#     y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
#     return ((x ** 2 + y ** 2) <= radius ** 2).astype(bool)

# def apply_spatial_logic(raw_grid):
#     closed = binary_closing(raw_grid, structure=make_disk_kernel(CLOSING_RADIUS)).astype(int)
#     inverted = 1 - closed
#     labeled_holes, n_holes = ndimage.label(inverted)
#     filled = closed.copy()
    
#     for i, size in enumerate(ndimage.sum(inverted, labeled_holes, range(1, n_holes + 1)), start=1):
#         if size <= MAX_HOLE_SIZE:
#             hole_mask = labeled_holes == i
#             border = closed[binary_dilation(hole_mask, iterations=2) & ~hole_mask]
#             if len(border) > 0 and border.mean() > 0.6: filled[hole_mask] = 1

#     inv2 = 1 - filled
#     labeled_inv, _ = ndimage.label(inv2)
#     for i, size in enumerate(ndimage.sum(inv2, labeled_inv, range(1, labeled_inv.max() + 1)), start=1):
#         if size <= MIN_ISLAND_SIZE: filled[labeled_inv == i] = 1

#     labeled_f, nf = ndimage.label(filled)
#     if nf > 1:
#         sizes_f = ndimage.sum(filled, labeled_f, range(1, nf + 1))
#         main = np.argmax(sizes_f) + 1
#         result = np.zeros_like(filled)
#         for i, size in enumerate(sizes_f, start=1):
#             if i == main or size >= MIN_FIELD_CLUSTER: result[labeled_f == i] = 1
#         filled = result

#     smoothed = np.maximum(binary_opening(filled, structure=make_disk_kernel(EROSION_RADIUS)).astype(int),
#                           binary_erosion(filled, structure=make_disk_kernel(CLOSING_RADIUS + 1))).astype(int)
#     return smoothed

# def run(patches_dir, weights_path, field_txt_out, non_field_txt_out):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"     MobileNet on {device.type.upper()}")
    
#     model = models.mobilenet_v2()
#     model.classifier[1] = nn.Sequential(nn.Linear(model.last_channel, 1), nn.Sigmoid())
#     model.load_state_dict(torch.load(weights_path, map_location=device))
#     model.to(device).eval()

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)), transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     files = [f for f in os.listdir(patches_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
#     total_files = len(files)
#     print(f"     Found {total_files} patches.")
    
#     max_r, max_c = 0, 0
#     patch_meta = []

#     for i, fname in enumerate(files):
#         if i > 0 and i % 500 == 0:
#             print(f"      Sorted {i} out of {total_files} patches")

#         parts = re.findall(r'\d+', fname)
#         if len(parts) < 2: continue
#         r, c = int(parts[-2]), int(parts[-1])
#         max_r, max_c = max(max_r, r), max(max_c, c)

#         img_pil = Image.open(os.path.join(patches_dir, fname)).convert('RGB')
#         img_np = np.array(img_pil)
        
#         with torch.no_grad(): dl_prob = model(transform(img_pil).unsqueeze(0).to(device)).item()

#         patch_meta.append({
#             'name': fname, 'r': r, 'c': c, 'dl': 1.0 - dl_prob,
#             'color': score_color(img_np), 'texture': score_texture(img_np)
#         })

#     print("     now spatial logic")
#     conf_grid, name_grid = np.zeros((max_r + 1, max_c + 1)), np.empty((max_r + 1, max_c + 1), dtype=object)
#     for p in patch_meta:
#         conf_grid[p['r'], p['c']] = (DL_WEIGHT*p['dl'] + COLOR_WEIGHT*p['color'] + TEXTURE_WEIGHT*p['texture']) / (DL_WEIGHT+COLOR_WEIGHT+TEXTURE_WEIGHT)
#         name_grid[p['r'], p['c']] = p['name']

#     final_conf = np.zeros_like(conf_grid)
#     for p in patch_meta:
#         final_conf[p['r'], p['c']] = float(DL_WEIGHT*p['dl'] + COLOR_WEIGHT*p['color'] + TEXTURE_WEIGHT*p['texture'] + CONTEXT_WEIGHT*score_context(p['r'], p['c'], conf_grid))

#     refined = apply_spatial_logic((final_conf >= FINAL_THRESHOLD).astype(int))

#     os.makedirs(os.path.dirname(field_txt_out), exist_ok=True)
#     with open(field_txt_out, 'w') as ff, open(non_field_txt_out, 'w') as nf:
#         for r in range(max_r + 1):
#             for c in range(max_c + 1):
#                 # if fname := name_grid[r, c]:' remove the walrus operator
#                 fname = name_grid[r, c]
#                 if fname:
#                     (ff if refined[r, c] == 1 else nf).write(f"{fname}\n")
                    
#     print(f"     Done, fields saved to: {field_txt_out}")
    
#     del model
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion
import re
import gc

# --- Logic Tuning Constants ---
DL_THRESHOLD    = 0.3
DL_WEIGHT       = 0.50
COLOR_WEIGHT    = 0.25
TEXTURE_WEIGHT  = 0.15
CONTEXT_WEIGHT  = 0.10
FINAL_THRESHOLD = 0.45
CLOSING_RADIUS  = 4
MAX_HOLE_SIZE   = 300
MIN_ISLAND_SIZE = 8
MIN_FIELD_CLUSTER = 40
EROSION_RADIUS  = 2
BATCH_SIZE      = 4

# --- Helper Functions ---

def score_color(img_np):
    r, g, b = img_np[:, :, 0].astype(float), img_np[:, :, 1].astype(float), img_np[:, :, 2].astype(float)
    soil_score = np.clip((r - b) / 128.0, 0, 1)         
    veg_score = np.clip((g - np.maximum(r, b)) / 64.0, 0, 1)
    grey_penalty = 1.0 - np.clip(1.0 - (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 128.0, 0, 1)
    return float(np.clip((0.6 * soil_score + 0.4 * veg_score - 0.3 * grey_penalty).mean(), 0, 1))

def score_texture(img_np):
    grey = img_np.mean(axis=2).astype(float)
    from scipy.ndimage import maximum_filter, minimum_filter
    local_range = (maximum_filter(grey, size=5) - minimum_filter(grey, size=5)) / 255.0
    return float(np.clip((float(local_range.mean()) - 0.04) / 0.16, 0, 1))

def score_context(r, c, conf_grid, radius=2):
    r0, r1 = max(0, r - radius), min(conf_grid.shape[0], r + radius + 1)
    c0, c1 = max(0, c - radius), min(conf_grid.shape[1], c + radius + 1)
    valid = conf_grid[r0:r1, c0:c1][conf_grid[r0:r1, c0:c1] > 0]
    return float(valid.mean()) if len(valid) > 0 else 0.5

def make_disk_kernel(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ((x ** 2 + y ** 2) <= radius ** 2).astype(bool)

def apply_spatial_logic(raw_grid):
    closed = binary_closing(raw_grid, structure=make_disk_kernel(CLOSING_RADIUS)).astype(int)
    inverted = 1 - closed
    labeled_holes, n_holes = ndimage.label(inverted)
    filled = closed.copy()

    for i, size in enumerate(ndimage.sum(inverted, labeled_holes, range(1, n_holes + 1)), start=1):
        if size <= MAX_HOLE_SIZE:
            hole_mask = labeled_holes == i
            border = closed[binary_dilation(hole_mask, iterations=2) & ~hole_mask]
            if len(border) > 0 and border.mean() > 0.6: filled[hole_mask] = 1

    inv2 = 1 - filled
    labeled_inv, _ = ndimage.label(inv2)
    for i, size in enumerate(ndimage.sum(inv2, labeled_inv, range(1, labeled_inv.max() + 1)), start=1):
        if size <= MIN_ISLAND_SIZE: filled[labeled_inv == i] = 1

    labeled_f, nf = ndimage.label(filled)
    if nf > 1:
        sizes_f = ndimage.sum(filled, labeled_f, range(1, nf + 1))
        main = np.argmax(sizes_f) + 1
        result = np.zeros_like(filled)
        for i, size in enumerate(sizes_f, start=1):
            if i == main or size >= MIN_FIELD_CLUSTER: result[labeled_f == i] = 1
        filled = result

    smoothed = np.maximum(binary_opening(filled, structure=make_disk_kernel(EROSION_RADIUS)).astype(int),
                          binary_erosion(filled, structure=make_disk_kernel(CLOSING_RADIUS + 1))).astype(int)
    return smoothed

# --- Main Run Function ---

def run(patches_dir, weights_path, field_txt_out, non_field_txt_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"     MobileNet running on {device.type.upper()}")

    # 1. Model Setup
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Sequential(nn.Linear(model.last_channel, 1), nn.Sigmoid())
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    files = [f for f in os.listdir(patches_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
    total_files = len(files)
    print(f"     Found {total_files} patches. Starting Batch Processing (Size: {BATCH_SIZE})...")

    max_r, max_c = 0, 0
    patch_meta = []

    # 2. Batch Processing Loop
    # for i in range(0, total_files, BATCH_SIZE):
    #     batch_files = files[i : i + BATCH_SIZE]
    #     batch_tensors = []
    #     batch_imgs_np = []
        
    #     for fname in batch_files:
    #         img_path = os.path.join(patches_dir, fname)
    #         img_pil = Image.open(img_path).convert('RGB')
            
    #         # Extract row/col info
    #         parts = re.findall(r'\d+', fname)
    #         if len(parts) >= 2:
    #             r, c = int(parts[-2]), int(parts[-1])
    #             max_r, max_c = max(max_r, r), max(max_c, c)
                
    #             batch_imgs_np.append(np.array(img_pil))
    #             batch_tensors.append(transform(img_pil))
            
    #         img_pil.close()

    #     if not batch_tensors: continue

    #     # GPU Inference
    #     input_batch = torch.stack(batch_tensors).to(device)
    #     with torch.no_grad():
    #         output = model(input_batch)
    #         probs = output.squeeze().cpu().numpy()
            
    #         # Handle edge case where batch size is 1
    #         if len(batch_files) == 1:
    #             probs = [probs.item()]

    #     # Post-Inference calculations (CPU)
    #     for j, fname in enumerate(batch_files):
    #         parts = re.findall(r'\d+', fname)
    #         r, c = int(parts[-2]), int(parts[-1])
            
    #         patch_meta.append({
    #             'name': fname, 'r': r, 'c': c, 
    #             'dl': 1.0 - probs[j],
    #             'color': score_color(batch_imgs_np[j]), 
    #             'texture': score_texture(batch_imgs_np[j])
    #         })

    #     # Memory Cleanup
    #     del input_batch
    #     del batch_tensors
    #     del batch_imgs_np
    #     if i % 160 == 0:
    #         print(f"      Processed {i}/{total_files} patches...")
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()


    for i in range(0, total_files, BATCH_SIZE):
        batch_files = files[i : i + BATCH_SIZE]
        batch_tensors = []
        color_scores = []
        texture_scores = []
        
        for fname in batch_files:
            img_path = os.path.join(patches_dir, fname)
            with Image.open(img_path).convert('RGB') as img_pil:
                # Extract row/col info
                parts = re.findall(r'\d+', fname)
                if len(parts) >= 2:
                    r, c = int(parts[-2]), int(parts[-1])
                    max_r, max_c = max(max_r, r), max(max_c, c)
                    
                    # Calculate SciPy scores immediately and discard the raw array 
                    # instead of storing the heavy raw numpy arrays in memory
                    img_np = np.array(img_pil)
                    color_scores.append(score_color(img_np))
                    texture_scores.append(score_texture(img_np))
                    
                    # Store only the resized tensor for the GPU
                    batch_tensors.append(transform(img_pil))

        if not batch_tensors: continue

        # GPU Inference
        input_batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            output = model(input_batch)
            probs = output.squeeze().cpu().numpy()
            
            if len(batch_files) == 1:
                probs = [probs.item()]

        # Post-Inference calculations (CPU)
        for j, fname in enumerate(batch_files):
            parts = re.findall(r'\d+', fname)
            r, c = int(parts[-2]), int(parts[-1])
            
            patch_meta.append({
                'name': fname, 'r': r, 'c': c, 
                'dl': 1.0 - probs[j],
                'color': color_scores[j], 
                'texture': texture_scores[j]
            })

        # Aggressive Memory Cleanup EVERY batch
        del input_batch
        del batch_tensors
        del color_scores
        del texture_scores
        
        # Run cleanup every batch for the Nano
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"      Processed {min(i + BATCH_SIZE, total_files)}/{total_files} patches...")

        
    # 3. Spatial Logic & Grid Building
    print("     Applying spatial logic and neighborhood context...")
    conf_grid = np.zeros((max_r + 1, max_c + 1))
    name_grid = np.empty((max_r + 1, max_c + 1), dtype=object)
    
    for p in patch_meta:
        conf_grid[p['r'], p['c']] = (DL_WEIGHT*p['dl'] + COLOR_WEIGHT*p['color'] + TEXTURE_WEIGHT*p['texture']) / (DL_WEIGHT+COLOR_WEIGHT+TEXTURE_WEIGHT)
        name_grid[p['r'], p['c']] = p['name']

    final_conf = np.zeros_like(conf_grid)
    for p in patch_meta:
        final_conf[p['r'], p['c']] = float(DL_WEIGHT*p['dl'] + COLOR_WEIGHT*p['color'] + TEXTURE_WEIGHT*p['texture'] + CONTEXT_WEIGHT*score_context(p['r'], p['c'], conf_grid))

    refined = apply_spatial_logic((final_conf >= FINAL_THRESHOLD).astype(int))

    # 4. Save Results (No Walrus Operator)
    os.makedirs(os.path.dirname(field_txt_out), exist_ok=True)
    with open(field_txt_out, 'w') as ff, open(non_field_txt_out, 'w') as nf:
        for r in range(max_r + 1):
            for c in range(max_c + 1):
                fname = name_grid[r, c]
                if fname:
                    if refined[r, c] == 1:
                        ff.write(f"{fname}\n")
                    else:
                        nf.write(f"{fname}\n")

    print(f"     Done! Field sorting results saved to: {os.path.dirname(field_txt_out)}")

    # Final Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
         torch.cuda.empty_cache()


 