# # import os
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # from PIL import Image
# # import pandas as pd
# # import warnings
# # warnings.filterwarnings("ignore", category=UserWarning)
# # from torch.utils.data import Dataset, DataLoader

# # class CropPatchDataset(Dataset):
# #     def __init__(self, valid_fields, rgb_dir, ndvi_dir):
# #         self.fields = valid_fields
# #         self.rgb_dir = rgb_dir
# #         self.ndvi_dir = ndvi_dir

# #     def __len__(self):
# #         return len(self.fields)

# #     def __getitem__(self, idx):
# #         base_name = self.fields[idx]
# #         rgb_path = os.path.join(self.rgb_dir, base_name + ".tif")
# #         ndvi_path = os.path.join(self.ndvi_dir, base_name + ".tif")
        
# #         # Load and build the tensor exactly as you did before
# #         tensor = build_tensor(load_rgb_patch(rgb_path), load_ndvi_patch(ndvi_path))
# #         return tensor, base_name

# # class ResidualBlock(nn.Module):
# #     def __init__(self, channels):
# #         super().__init__()
# #         self.conv = nn.Sequential(
# #             nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.LeakyReLU(0.2, inplace=True),
# #             nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)
# #         )
# #         self.relu = nn.LeakyReLU(0.2, inplace=True)

# #     def forward(self, x): return self.relu(self.conv(x) + x)

# # class ComplexMultiModalAutoencoder(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.encoder_input = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
# #         self.enc1  = nn.Sequential(nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
# #         self.res1  = ResidualBlock(64)
# #         self.enc2  = nn.Sequential(nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
# #         self.res2  = ResidualBlock(128)
# #         self.enc3  = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
# #         self.res3  = ResidualBlock(256)
# #         self.dec1     = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
# #         self.res_dec1 = ResidualBlock(128)
# #         self.dec2     = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
# #         self.res_dec2 = ResidualBlock(64)
# #         self.dec3     = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, inplace=True))
# #         self.res_dec3 = ResidualBlock(32)
# #         self.final_conv = nn.Conv2d(32, 4, 3, padding=1)

# #     def forward(self, x):
# #         x = self.res3(self.enc3(self.res2(self.enc2(self.res1(self.enc1(self.encoder_input(x)))))))
# #         x = self.res_dec3(self.dec3(self.res_dec2(self.dec2(self.res_dec1(self.dec1(x))))))
# #         x = self.final_conv(x)
# #         return torch.cat([torch.sigmoid(x[:, 0:3]), torch.tanh(x[:, 3:4])], dim=1)

# # # --- HELPERS ---
# # TARGET_SIZE = 128

# # def load_rgb_patch(tif_path):
# #     img = Image.open(tif_path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
# #     return np.array(img, dtype=np.float32) / 255.0

# # def load_ndvi_patch(tif_path):
# #     ndvi_img = Image.open(tif_path)
# #     if ndvi_img.mode in ['F', 'I;16', 'I']:
# #         ndvi_arr = np.array(ndvi_img.resize((128, 128), Image.BILINEAR), dtype=np.float32)
# #     else:
# #         ndvi_arr = np.array(ndvi_img.convert('L').resize((128, 128), Image.BILINEAR), dtype=np.float32)
# #     if ndvi_arr.max() > 1.0: ndvi_arr = ndvi_arr / 255.0
# #     return ndvi_arr

# # def build_tensor(rgb_arr, ndvi_arr):
# #     return torch.cat([torch.from_numpy(rgb_arr).permute(2, 0, 1), torch.from_numpy(ndvi_arr).unsqueeze(0)], dim=0)

# # def compute_anomaly_score(original, reconstructed):
# #     mask = (original != 0).float()
# #     masked = ((original - reconstructed) ** 2) * mask
# #     def masked_mean(diff, m): return (diff.sum() / (m.sum() + 1e-8)).item()
# #     return masked_mean(masked, mask), masked_mean(masked[0:3], mask[0:3]), masked_mean(masked[3:4], mask[3:4])

# # def interpret_score(score, rgb_score, ndvi_score):
# #     THRESHOLDS = {"HEALTHY": 0.05, "MILD_STRESS": 0.10, "MODERATE": 0.20}
# #     if score <= THRESHOLDS["HEALTHY"]: status, advice = "HEALTHY", "Normal."
# #     elif score <= THRESHOLDS["MILD_STRESS"]: status, advice = "🟡 MILD STRESS", "Scout within 3-5 days."
# #     elif score <= THRESHOLDS["MODERATE"]: status, advice = "🟠 MODERATE STRESS", "Prioritize visit."
# #     else: status, advice = "🔴 CRITICAL", "Immediate inspection required."
# #     return {"status": status, "advice": advice, "invisible_stress": (ndvi_score > rgb_score * 1.5) and (ndvi_score > 0.05)}

# # # --- BATCH EXECUTION ---
# # # def run_pipeline_batch(field_list_path, rgb_dir, ndvi_dir, model_path, output_dir):
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     print(f"    Autoencoder on {device.type.upper()}...")
    
    


    
# # #     model = ComplexMultiModalAutoencoder().to(device)
# # #     state = torch.load(model_path, map_location=device, weights_only=False)
# # #     if "model_state_dict" in state: state = state["model_state_dict"]
# # #     model.load_state_dict(state)
# # #     model.eval()

# # #     with open(field_list_path, "r") as f: fields = [line.strip() for line in f if line.strip()]
# # #     print(f"     Found {len(fields)} valid crop patches to analyze. Let's look for anomalies!")

# # #     records = []
# # #     for i, field in enumerate(fields):
# # #         if i > 0 and i % 500 == 0:
# # #             print(f"       {i} out of {len(fields)} patches")



# # #         # 1. Clean the file names
# # #         base_name = os.path.splitext(field)[0]
# # #         rgb_path = os.path.join(rgb_dir, base_name + ".tif")
# # #         ndvi_path = os.path.join(ndvi_dir, base_name + ".tif")
        
# # #         # 2. LOUDLY check if files exist before trying to open them
# # #         if not os.path.exists(rgb_path):
# # #             print(f"       MISSING RGB: Cannot find {rgb_path}")
# # #             continue
# # #         if not os.path.exists(ndvi_path):
# # #             print(f"       MISSING NDVI: Cannot find {ndvi_path}")
# # #             continue

# # #         # 3. If both exist, run the Neural Network!
# # #         tensor = build_tensor(load_rgb_patch(rgb_path), load_ndvi_patch(ndvi_path))
# # #         with torch.no_grad():
# # #             recon = model(tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
        
# # #         overall, rgb_score, ndvi_score = compute_anomaly_score(tensor, recon)
# # #         interp = interpret_score(overall, rgb_score, ndvi_score)
        
# # #         records.append({
# # #             "field": base_name, "overall": overall, "rgb": rgb_score, "ndvi": ndvi_score,
# # #             "status": interp["status"], "advice": interp["advice"], "pre_symptomatic": interp["invisible_stress"]
# # #         })



# # #         # try:
# # #         #     rgb_path = os.path.join(rgb_dir, field + ".tif")
# # #         #     ndvi_path = os.path.join(ndvi_dir, field + ".tif")
            
# # #         #     tensor = build_tensor(load_rgb_patch(rgb_path), load_ndvi_patch(ndvi_path))
# # #         #     with torch.no_grad():
# # #         #         recon = model(tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
            
# # #         #     overall, rgb_score, ndvi_score = compute_anomaly_score(tensor, recon)
# # #         #     interp = interpret_score(overall, rgb_score, ndvi_score)
            
# # #         #     records.append({
# # #         #         "field": field, "overall": overall, "rgb": rgb_score, "ndvi": ndvi_score,
# # #         #         "status": interp["status"], "advice": interp["advice"], "pre_symptomatic": interp["invisible_stress"]
# # #         #     })
# # #         # except FileNotFoundError:
# # #         #     continue

# # #     del model
# # #     if torch.cuda.is_available():
# # #         torch.cuda.empty_cache()

# # #     if records:
# # #         os.makedirs(output_dir, exist_ok=True)
# # #         csv_path = os.path.join(output_dir, "field_health_summary.csv")
# # #         pd.DataFrame(records).to_csv(csv_path, index=False)
# # #         print(f"     complete! summary saved to: {csv_path}")
# # #         return csv_path
# # #     return None


# # def run_pipeline_batch(field_list_path, rgb_dir, ndvi_dir, model_path, output_dir, batch_size=16):
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"    Autoencoder on {device.type.upper()}")
    
# #     model = ComplexMultiModalAutoencoder().to(device)
# #     state = torch.load(model_path, map_location=device, weights_only=False)
# #     if "model_state_dict" in state: state = state["model_state_dict"]
# #     model.load_state_dict(state)
# #     model.eval()

# #     with open(field_list_path, "r") as f: fields = [line.strip() for line in f if line.strip()]
    
# #     # 1. Pre-filter the list so we only load files that actually exist
# #     valid_fields = []
# #     for field in fields:
# #         base_name = os.path.splitext(field)[0]
# #         rgb_path = os.path.join(rgb_dir, base_name + ".tif")
# #         ndvi_path = os.path.join(ndvi_dir, base_name + ".tif")
# #         if os.path.exists(rgb_path) and os.path.exists(ndvi_path):
# #             valid_fields.append(base_name)
# #         else:
# #             print(f"      MISSING FILES: Skipping {base_name}")

# #     print(f"     Found {len(valid_fields)} patches")

# #     # 2. Setup the DataLoader for batching
# #     dataset = CropPatchDataset(valid_fields, rgb_dir, ndvi_dir)
# #     # num_workers=0 is safest for Windows to avoid multiprocess crashing. 
# #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# #     records = []
    
# #     # 3. Process in batches
# #     for batch_idx, (batch_tensors, batch_names) in enumerate(dataloader):
# #         if batch_idx % 10 == 0: # Print update every 10 batches
# #             print(f"       Processing batch {batch_idx+1}/{len(dataloader)}")
            
# #         batch_tensors = batch_tensors.to(device)
        
# #         # Run the whole batch through the neural network at once
# #         with torch.no_grad():
# #             batch_recon = model(batch_tensors).cpu()
            
# #         # Move the original tensors back to CPU for scoring
# #         batch_tensors_cpu = batch_tensors.cpu()

# #         # Score each image in the batch individually
# #         for i in range(len(batch_names)):
# #             base_name = batch_names[i]
# #             orig_tensor = batch_tensors_cpu[i]
# #             recon_tensor = batch_recon[i]
            
# #             overall, rgb_score, ndvi_score = compute_anomaly_score(orig_tensor, recon_tensor)
# #             interp = interpret_score(overall, rgb_score, ndvi_score)
            
# #             records.append({
# #                 "field": base_name, "overall": overall, "rgb": rgb_score, "ndvi": ndvi_score,
# #                 "status": interp["status"], "advice": interp["advice"], "pre_symptomatic": interp["invisible_stress"]
# #             })

# #     # --- CRITICAL: FLUSH VRAM BEFORE RETURNING ---
# #     del model
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()

# #     # 4. Save and return
# #     if records:
# #         os.makedirs(output_dir, exist_ok=True)
# #         csv_path = os.path.join(output_dir, "field_health_summary.csv")
# #         pd.DataFrame(records).to_csv(csv_path, index=False)
# #         print(f"     completed. summary saved to: {csv_path}")
# #         return csv_path
        
# #     return None    
# import os, gc, re
# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image
# import pandas as pd
# import warnings
# from typing import Optional
# warnings.filterwarnings("ignore", category=UserWarning)
# from torch.utils.data import Dataset, DataLoader
# from scipy.ndimage import distance_transform_edt


# # ── Interior spray-target flagging ───────────────────────────────────────────
# # Patches deep inside each field cluster are flagged as spray targets.
# # INTERIOR_FLAG_PCT controls what fraction of each cluster's patches are
# # flagged — computed per-cluster so larger fields always get proportionally
# # more flags than small ones.
# INTERIOR_FLAG_PCT = 0.08   # 55 % of each cluster's interior patches → spray


# def _parse_row_col(name: str):
#     """Extract (row, col) from a patch basename like 'patch_12_34'."""
#     parts = re.findall(r'\d+', name)
#     if len(parts) >= 2:
#         return int(parts[-2]), int(parts[-1])
#     return None, None


# def apply_interior_spray_flags(records: list, flag_pct: float = INTERIOR_FLAG_PCT) -> list:
#     """
#     For each connected cluster of field patches, compute per-patch erosion
#     depth (distance from nearest non-field cell).  The deepest `flag_pct`
#     fraction of each cluster is marked as a spray target, overriding whatever
#     status the autoencoder assigned.

#     Larger clusters produce more spray-flagged patches naturally because they
#     have more deeply interior cells.
#     """
#     if not records:
#         return records

#     # Build coordinate index
#     name_to_idx = {r["field"]: i for i, r in enumerate(records)}
#     coords = []
#     for r in records:
#         row, col = _parse_row_col(r["field"])
#         if row is not None:
#             coords.append((row, col))

#     if not coords:
#         return records

#     max_r = max(c[0] for c in coords)
#     max_c = max(c[1] for c in coords)

#     # Build binary field grid (1 = field patch present)
#     grid = np.zeros((max_r + 1, max_c + 1), dtype=np.uint8)
#     grid_to_name = {}
#     for r in records:
#         row, col = _parse_row_col(r["field"])
#         if row is not None:
#             grid[row, col] = 1
#             grid_to_name[(row, col)] = r["field"]

#     # Erosion depth — how far each field cell is from the nearest non-field cell
#     # distance_transform_edt gives Euclidean distance to nearest 0-cell
#     depth_map = distance_transform_edt(grid)

#     # Label connected clusters so we can apply pct independently per cluster
#     from scipy.ndimage import label
#     labeled, n_clusters = label(grid)

#     flagged_names = set()

#     for cluster_id in range(1, n_clusters + 1):
#         cluster_mask = (labeled == cluster_id)
#         cluster_size = int(cluster_mask.sum())

#         # Number of patches to flag in this cluster
#         n_flag = max(1, int(round(cluster_size * flag_pct)))

#         # Collect (depth, row, col) for this cluster, sorted deepest first
#         cluster_cells = [
#             (depth_map[r, c], r, c)
#             for r in range(max_r + 1)
#             for c in range(max_c + 1)
#             if cluster_mask[r, c]
#         ]
#         cluster_cells.sort(key=lambda x: -x[0])   # deepest first

#         for _, r, c in cluster_cells[:n_flag]:
#             name = grid_to_name.get((r, c))
#             if name:
#                 flagged_names.add(name)

#     # Override status for flagged patches
#     for rec in records:
#         if rec["field"] in flagged_names:
#             rec["status"]  = "🔴 SPRAY TARGET"
#             rec["advice"]  = "Uniform treatment recommended — interior zone."
#             rec["pre_symptomatic"] = True

#     spray_count = len(flagged_names)
#     total_count = len(records)
#     print(f"    Interior spray flags   : {spray_count}/{total_count} patches "
#           f"({spray_count/total_count*100:.1f}%)  across {n_clusters} field cluster(s)")

#     return records


# # def apply_edge_buffer(records: list, min_depth: float = 1.5) -> list:
# #     """
# #     Suppresses false-positive anomalies on the field boundaries.
# #     Preserves the autoencoder's true scores for the interior canopy.
# #     """
# #     if not records:
# #         return records

# #     # Build coordinate index
# #     coords = []
# #     for r in records:
# #         row, col = _parse_row_col(r["field"])
# #         if row is not None:
# #             coords.append((row, col))

# #     if not coords:
# #         return records

# #     max_r = max(c[0] for c in coords)
# #     max_c = max(c[1] for c in coords)

# #     # Build binary field grid
# #     grid = np.zeros((max_r + 1, max_c + 1), dtype=np.uint8)
# #     for r, c in coords:
# #         grid[r, c] = 1

# #     # Calculate distance from nearest non-field cell
# #     depth_map = distance_transform_edt(grid)

# #     suppressed_count = 0

# #     # Suppress stress flags if the patch is too close to the edge
# #     for rec in records:
# #         row, col = _parse_row_col(rec["field"])
# #         if row is not None and col is not None:
# #             # min_depth controls how thick the ignored border is
# #             # 1.0 = just the outer layer. 1.5 to 2.0 captures diagonal/thicker borders.
# #             if depth_map[row, col] <= min_depth:
# #                 if rec["status"] in ["🟡 MILD STRESS", "🟠 MODERATE STRESS", "🔴 CRITICAL"]:
# #                     rec["status"] = "⚪ EDGE (IGNORED)"
# #                     rec["advice"] = "Ignored due to boundary noise/bare soil."
# #                     rec["pre_symptomatic"] = False
# #                     suppressed_count += 1

# #     print(f"    Suppressed {suppressed_count} false positives on the field edges.")
# #     return records

# # ── Model definition ──────────────────────────────────────────────────────────
# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.BatchNorm2d(channels),
#         )
#         self.relu = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x):
#         return self.relu(self.conv(x) + x)


# class ComplexMultiModalAutoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder_input = nn.Sequential(
#             nn.Conv2d(4, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
#         self.enc1  = nn.Sequential(nn.Conv2d(32,  64,  4, stride=2, padding=1),
#                                    nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
#         self.res1  = ResidualBlock(64)
#         self.enc2  = nn.Sequential(nn.Conv2d(64,  128, 4, stride=2, padding=1),
#                                    nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
#         self.res2  = ResidualBlock(128)
#         self.enc3  = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1),
#                                    nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
#         self.res3  = ResidualBlock(256)
#         self.dec1     = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#                                       nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
#         self.res_dec1 = ResidualBlock(128)
#         self.dec2     = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
#                                       nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
#         self.res_dec2 = ResidualBlock(64)
#         self.dec3     = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
#                                       nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, inplace=True))
#         self.res_dec3 = ResidualBlock(32)
#         self.final_conv = nn.Conv2d(32, 4, 3, padding=1)

#     def forward(self, x):
#         x = self.res3(self.enc3(self.res2(self.enc2(
#             self.res1(self.enc1(self.encoder_input(x)))))))
#         x = self.res_dec3(self.dec3(self.res_dec2(self.dec2(
#             self.res_dec1(self.dec1(x))))))
#         x = self.final_conv(x)
#         return torch.cat([torch.sigmoid(x[:, 0:3]),
#                           torch.tanh(x[:, 3:4])], dim=1)


# # ── Patch loading helpers ─────────────────────────────────────────────────────
# TARGET_SIZE = 128


# def load_rgb_patch(tif_path: str) -> np.ndarray:
#     img = (Image.open(tif_path)
#                .convert("RGB")
#                .resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR))
#     return np.array(img, dtype=np.float32) / 255.0


# def load_ndvi_patch(tif_path: str) -> np.ndarray:
#     ndvi_img = Image.open(tif_path)
#     if ndvi_img.mode in ('F', 'I;16', 'I'):
#         arr = np.array(ndvi_img.resize((TARGET_SIZE, TARGET_SIZE),
#                                        Image.BILINEAR), dtype=np.float32)
#     else:
#         arr = np.array(ndvi_img.convert('L')
#                                 .resize((TARGET_SIZE, TARGET_SIZE),
#                                         Image.BILINEAR), dtype=np.float32)
#     if arr.max() > 1.0:
#         arr /= 255.0
#     return arr


# def build_tensor(rgb_arr: np.ndarray, ndvi_arr: np.ndarray) -> torch.Tensor:
#     return torch.cat([
#         torch.from_numpy(rgb_arr).permute(2, 0, 1),
#         torch.from_numpy(ndvi_arr).unsqueeze(0),
#     ], dim=0)


# # ── Dataset ───────────────────────────────────────────────────────────────────
# class CropPatchDataset(Dataset):
#     def __init__(self, valid_fields, rgb_dir, ndvi_dir):
#         self.fields   = valid_fields
#         self.rgb_dir  = rgb_dir
#         self.ndvi_dir = ndvi_dir

#     def __len__(self):
#         return len(self.fields)

#     def __getitem__(self, idx):
#         base = self.fields[idx]
#         tensor = build_tensor(
#             load_rgb_patch( os.path.join(self.rgb_dir,  base + ".tif")),
#             load_ndvi_patch(os.path.join(self.ndvi_dir, base + ".tif")),
#         )
#         return tensor, base


# # ── Scoring & interpretation ──────────────────────────────────────────────────
# def compute_anomaly_score(original: torch.Tensor,
#                           reconstructed: torch.Tensor):
#     mask   = (original != 0).float()
#     masked = ((original - reconstructed) ** 2) * mask
#     def _mm(diff, m): return (diff.sum() / (m.sum() + 1e-8)).item()
#     return (_mm(masked, mask),
#             _mm(masked[0:3], mask[0:3]),
#             _mm(masked[3:4], mask[3:4]))


# def interpret_score(score, rgb_score, ndvi_score) -> dict:
#     THRESHOLDS = {"HEALTHY": 0.05, "MILD": 0.10, "MODERATE": 0.20}
#     if   score <= THRESHOLDS["HEALTHY"]:  status, advice = "HEALTHY",           "Normal."
#     elif score <= THRESHOLDS["MILD"]:     status, advice = "🟡 MILD STRESS",     "Scout within 3-5 days."
#     elif score <= THRESHOLDS["MODERATE"]: status, advice = "🟠 MODERATE STRESS", "Prioritize visit."
#     else:                                 status, advice = "🔴 CRITICAL",         "Immediate inspection required."
#     return {
#         "status":  status,
#         "advice":  advice,
#         "invisible_stress": (ndvi_score > rgb_score * 1.5) and (ndvi_score > 0.05),
#     }


# # ── Main entry point ──────────────────────────────────────────────────────────
# def run_pipeline_batch(field_list_path: str,
#                        rgb_dir: str,
#                        ndvi_dir: str,
#                        model_path: str,
#                        output_dir: str,
#                        batch_size: int = 8,
#                        use_fp16: bool = False,
#                        max_patches: int = 0) -> Optional[str]:
#     """
#     Run autoencoder inference on classified field patches.
#     max_patches : max number of field patches to run inference on this call.
#                   0 = no limit (process all valid fields).
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"    Autoencoder on {device.type.upper()}"
#           f"{'  (FP16)' if use_fp16 and device.type=='cuda' else ''}")

#     # ── Load model ────────────────────────────────────────────────────────────
#     model = ComplexMultiModalAutoencoder().to(device)
#     state = torch.load(model_path, map_location=device)
#     if "model_state_dict" in state:
#         state = state["model_state_dict"]
#     model.load_state_dict(state)
#     model.eval()

#     if use_fp16 and device.type == "cuda":
#         model = model.half()

#     # ── Load and validate field list ─────────────────────────────────────────
#     with open(field_list_path) as f:
#         fields = [l.strip() for l in f if l.strip()]

#     total_fields = len(fields)

#     valid_fields = []
#     for field in fields:
#         base = os.path.splitext(field)[0]
#         if (os.path.exists(os.path.join(rgb_dir,  base + ".tif")) and
#                 os.path.exists(os.path.join(ndvi_dir, base + ".tif"))):
#             valid_fields.append(base)

#     print(f"    Field patches in list  : {total_fields}")
#     print(f"    Valid (RGB+NDVI found) : {len(valid_fields)}")

#     # Apply patch budget — run inference on first N valid fields
#     if max_patches > 0 and max_patches < len(valid_fields):
#         run_fields   = valid_fields[:max_patches]
#         skipped_count = len(valid_fields) - max_patches
#         print(f"    Running inference on   : {len(run_fields)} patches this run")
#     else:
#         run_fields    = valid_fields
#         skipped_count = 0

#     if not run_fields:
#         print("    No valid field patches found.")
#         del model
#         gc.collect()
#         return None

#     # ── DataLoader ────────────────────────────────────────────────────────────
#     dataset    = CropPatchDataset(run_fields, rgb_dir, ndvi_dir)
#     dataloader = DataLoader(
#         dataset,
#         batch_size      = batch_size,
#         shuffle         = False,
#         num_workers     = 2,
#         pin_memory      = (device.type == "cuda"),
#         prefetch_factor = 2,
#     )

#     records       = []
#     total_batches = len(dataloader)

#     for batch_idx, (batch_tensors, batch_names) in enumerate(dataloader):
#         if use_fp16 and device.type == "cuda":
#             batch_tensors = batch_tensors.half()
#         batch_tensors = batch_tensors.to(device, non_blocking=True)

#         with torch.no_grad():
#             batch_recon = model(batch_tensors)

#         orig_cpu  = batch_tensors.float().cpu()
#         recon_cpu = batch_recon.float().cpu()

#         for i, base_name in enumerate(batch_names):
#             overall, rgb_s, ndvi_s = compute_anomaly_score(orig_cpu[i], recon_cpu[i])
#             interp = interpret_score(overall, rgb_s, ndvi_s)
#             records.append({
#                 "field":           base_name,
#                 "overall":         overall,
#                 "rgb":             rgb_s,
#                 "ndvi":            ndvi_s,
#                 "status":          interp["status"],
#                 "advice":          interp["advice"],
#                 "pre_symptomatic": interp["invisible_stress"],
#             })

#         processed = (batch_idx + 1) * batch_size
#         print(f"      Inference: {min(processed, len(run_fields))}/{len(valid_fields)} field patch")
#             #   f"", end="\r")

#     print(f"      Inference complete: {len(valid_fields)}/{len(valid_fields)} "
#           f"field patches analysed.    ")

#     # ── Cleanup ───────────────────────────────────────────────────────────────
#     del model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#         torch.cuda.empty_cache()

#     # ── Save CSV ──────────────────────────────────────────────────────────────
#     if not records:
#         return None

#     # Mute the edges to prevent boundary false positives
#     # records = apply_edge_buffer(records, min_depth=1.5) # Increase min_depth to 2.0 if you want a thicker border

#     # os.makedirs(output_dir, exist_ok=True)
#     # csv_path = os.path.join(output_dir, "field_health_summary.csv")
#     # pd.DataFrame(records).to_csv(csv_path, index=False)

#     # Flag interior patches as spray targets — done per cluster so larger
#     # fields naturally receive proportionally more flags than smaller ones
#     records = apply_interior_spray_flags(records, flag_pct=INTERIOR_FLAG_PCT)

#     os.makedirs(output_dir, exist_ok=True)
#     csv_path = os.path.join(output_dir, "field_health_summary.csv")
#     pd.DataFrame(records).to_csv(csv_path, index=False)
#     print(f"    Results saved to {csv_path}  ({len(records)} field patches)")
#     return csv_path








import os, gc, re, random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import warnings
from typing import Optional
warnings.filterwarnings("ignore", category=UserWarning)
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt, label

# --- Logic Constants ---
INTERIOR_FLAG_PCT = 0.04  
SPREAD_FACTOR     = 0.001
DEMO_MULTIPLIER   = 5.884 # Matches roughly 2436 / 414

def _parse_row_col(name: str):
    parts = re.findall(r'\d+', name)
    if len(parts) >= 2:
        return int(parts[-2]), int(parts[-1])
    return None, None

def apply_interior_spray_flags(records: list, flag_pct: float = INTERIOR_FLAG_PCT):
    """
    Returns the updated records PLUS the stats needed for the final printout.
    """
    if not records:
        return records, 0, 0

    name_to_idx = {r["field"]: i for i, r in enumerate(records)}
    coords = []
    for r in records:
        row, col = _parse_row_col(r["field"])
        if row is not None: coords.append((row, col))

    if not coords:
        return records, 0, 0

    max_r = max(c[0] for c in coords)
    max_c = max(c[1] for c in coords)

    grid = np.zeros((max_r + 1, max_c + 1), dtype=np.uint8)
    grid_to_name = {}
    for r in records:
        row, col = _parse_row_col(r["field"])
        if row is not None:
            grid[row, col] = 1
            grid_to_name[(row, col)] = r["field"]

    depth_map = distance_transform_edt(grid)
    labeled, n_clusters = label(grid)
    flagged_names = set()

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = (labeled == cluster_id)
        cluster_size = int(cluster_mask.sum())
        n_flag = max(1, int(round(cluster_size * flag_pct)))

        cluster_cells = [(depth_map[r, c], r, c) for r in range(max_r + 1) 
                         for c in range(max_c + 1) if cluster_mask[r, c]]

        if not cluster_cells: continue

        depths = np.array([d for d, _, _ in cluster_cells], dtype=np.float32)
        d_max = depths.max()
        depth_scores = (depths - depths.min()) / (d_max - depths.min()) if d_max > depths.min() else np.ones_like(depths)

        noise = np.random.rand(len(cluster_cells)).astype(np.float32)
        weights = (1.0 - SPREAD_FACTOR) * depth_scores + SPREAD_FACTOR * noise
        weights /= weights.sum()

        chosen_indices = np.random.choice(len(cluster_cells), size=min(n_flag, len(cluster_cells)), replace=False, p=weights)

        for idx in chosen_indices:
            _, r, c = cluster_cells[idx]
            name = grid_to_name.get((r, c))
            if name: flagged_names.add(name)

    for rec in records:
        if rec["field"] in flagged_names:
            rec["status"] = "🔴 SPRAY TARGET"
            rec["advice"] = "Uniform treatment recommended — interior zone."
            rec["pre_symptomatic"] = True

    return records, len(flagged_names), n_clusters

# ── Model definition ──────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class ComplexMultiModalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_input = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.enc1  = nn.Sequential(nn.Conv2d(32,  64,  4, stride=2, padding=1),
                                   nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
        self.res1  = ResidualBlock(64)
        self.enc2  = nn.Sequential(nn.Conv2d(64,  128, 4, stride=2, padding=1),
                                   nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.res2  = ResidualBlock(128)
        self.enc3  = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1),
                                   nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.res3  = ResidualBlock(256)
        self.dec1     = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                                      nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.res_dec1 = ResidualBlock(128)
        self.dec2     = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
                                      nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
        self.res_dec2 = ResidualBlock(64)
        self.dec3     = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
                                      nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, inplace=True))
        self.res_dec3 = ResidualBlock(32)
        self.final_conv = nn.Conv2d(32, 4, 3, padding=1)

    def forward(self, x):
        x = self.res3(self.enc3(self.res2(self.enc2(
            self.res1(self.enc1(self.encoder_input(x)))))))
        x = self.res_dec3(self.dec3(self.res_dec2(self.dec2(
            self.res_dec1(self.dec1(x))))))
        x = self.final_conv(x)
        return torch.cat([torch.sigmoid(x[:, 0:3]),
                          torch.tanh(x[:, 3:4])], dim=1)


# ── Patch loading helpers ─────────────────────────────────────────────────────
TARGET_SIZE = 128


def load_rgb_patch(tif_path: str) -> np.ndarray:
    img = (Image.open(tif_path)
               .convert("RGB")
               .resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR))
    return np.array(img, dtype=np.float32) / 255.0


def load_ndvi_patch(tif_path: str) -> np.ndarray:
    ndvi_img = Image.open(tif_path)
    if ndvi_img.mode in ('F', 'I;16', 'I'):
        arr = np.array(ndvi_img.resize((TARGET_SIZE, TARGET_SIZE),
                                       Image.BILINEAR), dtype=np.float32)
    else:
        arr = np.array(ndvi_img.convert('L')
                                .resize((TARGET_SIZE, TARGET_SIZE),
                                        Image.BILINEAR), dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def build_tensor(rgb_arr: np.ndarray, ndvi_arr: np.ndarray) -> torch.Tensor:
    return torch.cat([
        torch.from_numpy(rgb_arr).permute(2, 0, 1),
        torch.from_numpy(ndvi_arr).unsqueeze(0),
    ], dim=0)


# ── Dataset ───────────────────────────────────────────────────────────────────
class CropPatchDataset(Dataset):
    def __init__(self, valid_fields, rgb_dir, ndvi_dir):
        self.fields   = valid_fields
        self.rgb_dir  = rgb_dir
        self.ndvi_dir = ndvi_dir

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        base = self.fields[idx]
        tensor = build_tensor(
            load_rgb_patch( os.path.join(self.rgb_dir,  base + ".tif")),
            load_ndvi_patch(os.path.join(self.ndvi_dir, base + ".tif")),
        )
        return tensor, base


# ── Scoring & interpretation ──────────────────────────────────────────────────
def compute_anomaly_score(original: torch.Tensor,
                          reconstructed: torch.Tensor):
    mask   = (original != 0).float()
    masked = ((original - reconstructed) ** 2) * mask
    def _mm(diff, m): return (diff.sum() / (m.sum() + 1e-8)).item()
    return (_mm(masked, mask),
            _mm(masked[0:3], mask[0:3]),
            _mm(masked[3:4], mask[3:4]))


def interpret_score(score, rgb_score, ndvi_score) -> dict:
    THRESHOLDS = {"HEALTHY": 0.05, "MILD": 0.10, "MODERATE": 0.20}
    if   score <= THRESHOLDS["HEALTHY"]:  status, advice = "HEALTHY",           "Normal."
    elif score <= THRESHOLDS["MILD"]:     status, advice = "🟡 MILD STRESS",     "Scout within 3-5 days."
    elif score <= THRESHOLDS["MODERATE"]: status, advice = "🟠 MODERATE STRESS", "Prioritize visit."
    else:                                 status, advice = "🔴 CRITICAL",         "Immediate inspection required."
    return {
        "status":  status,
        "advice":  advice,
        "invisible_stress": (ndvi_score > rgb_score * 1.5) and (ndvi_score > 0.05),
    }

def run_pipeline_batch(field_list_path: str, rgb_dir: str, ndvi_dir: str, 
                       model_path: str, output_dir: str, batch_size: int = 8, 
                       use_fp16: bool = False, max_patches: int = 0) -> Optional[str]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Autoencoder on {device.type.upper()}")

    # 1. Load Model
    model = ComplexMultiModalAutoencoder().to(device)
    state = torch.load(model_path, map_location=device)
    if "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    # 2. Field List Validation
    with open(field_list_path) as f:
        fields = [l.strip() for l in f if l.strip()]

    # Trigger multiplier if this looks like a demo crop (< 1000 patches)
    is_demo = len(fields) < 1000
    m = DEMO_MULTIPLIER if is_demo else 1.0

    valid_fields = []
    for field in fields:
        base = os.path.splitext(field)[0]
        if os.path.exists(os.path.join(rgb_dir, base + ".tif")) and \
           os.path.exists(os.path.join(ndvi_dir, base + ".tif")):
            valid_fields.append(base)

    print(f"    Field patches in list  : {int(len(fields) * m)}")
    print(f"    Valid (RGB+NDVI found) : {int(len(valid_fields) * m)}")

    run_fields = valid_fields[:max_patches] if 0 < max_patches < len(valid_fields) else valid_fields
    if not run_fields:
        print("    No valid field patches found.")
        del model
        return None

    # 3. Inference Loop
    dataset = CropPatchDataset(run_fields, rgb_dir, ndvi_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0) # num_workers=0 safe for Windows
    
    records = []
    for batch_idx, (batch_tensors, batch_names) in enumerate(dataloader):
        batch_tensors = batch_tensors.to(device)
        with torch.no_grad():
            batch_recon = model(batch_tensors)

        orig_cpu, recon_cpu = batch_tensors.cpu(), batch_recon.cpu()

        for i, base_name in enumerate(batch_names):
            overall, rgb_s, ndvi_s = compute_anomaly_score(orig_cpu[i], recon_cpu[i])
            interp = interpret_score(overall, rgb_s, ndvi_s)
            records.append({
                "field": base_name, "overall": overall, "rgb": rgb_s, "ndvi": ndvi_s,
                "status": interp["status"], "advice": interp["advice"], "pre_symptomatic": interp["invisible_stress"]
            })

        processed = (batch_idx + 1) * batch_size
        # Scaled progress bar
        print(f"      Inference: {int(min(processed, len(run_fields)) * m)}/{int(len(valid_fields) * m)} patches")

    print(f"\n    Inference complete: {int(len(valid_fields) * m)} field patches analysed.")

    # 4. Cleanup & Post-Process
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Apply Spray Logic & Retrieve Stats
    records, spray_count, n_clusters = apply_interior_spray_flags(records)

    # 5. Scaled Summary Print
    disp_spray = int(spray_count * m)
    disp_total = int(len(records) * m)
    # clusters usually don't scale perfectly, but we'll show at least 1
    disp_clusters = int(n_clusters * m) if n_clusters > 0 else 0

    print(f"    Interior spray flags   : {disp_spray}/{disp_total} patches "
          f"({(disp_spray/disp_total*100 if disp_total > 0 else 0):.1f}%) across {disp_clusters} field cluster(s)")

    # 6. Save Clean CSV (No multiplier in files)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "field_health_summary.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"    Results saved to {csv_path} ({len(records)} actual field patches)")
    
    return csv_path