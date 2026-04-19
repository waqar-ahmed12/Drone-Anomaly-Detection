# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# from torch.utils.data import Dataset, DataLoader

# class CropPatchDataset(Dataset):
#     def __init__(self, valid_fields, rgb_dir, ndvi_dir):
#         self.fields = valid_fields
#         self.rgb_dir = rgb_dir
#         self.ndvi_dir = ndvi_dir

#     def __len__(self):
#         return len(self.fields)

#     def __getitem__(self, idx):
#         base_name = self.fields[idx]
#         rgb_path = os.path.join(self.rgb_dir, base_name + ".tif")
#         ndvi_path = os.path.join(self.ndvi_dir, base_name + ".tif")
        
#         # Load and build the tensor exactly as you did before
#         tensor = build_tensor(load_rgb_patch(rgb_path), load_ndvi_patch(ndvi_path))
#         return tensor, base_name

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)
#         )
#         self.relu = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x): return self.relu(self.conv(x) + x)

# class ComplexMultiModalAutoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder_input = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
#         self.enc1  = nn.Sequential(nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
#         self.res1  = ResidualBlock(64)
#         self.enc2  = nn.Sequential(nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
#         self.res2  = ResidualBlock(128)
#         self.enc3  = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
#         self.res3  = ResidualBlock(256)
#         self.dec1     = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
#         self.res_dec1 = ResidualBlock(128)
#         self.dec2     = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
#         self.res_dec2 = ResidualBlock(64)
#         self.dec3     = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, inplace=True))
#         self.res_dec3 = ResidualBlock(32)
#         self.final_conv = nn.Conv2d(32, 4, 3, padding=1)

#     def forward(self, x):
#         x = self.res3(self.enc3(self.res2(self.enc2(self.res1(self.enc1(self.encoder_input(x)))))))
#         x = self.res_dec3(self.dec3(self.res_dec2(self.dec2(self.res_dec1(self.dec1(x))))))
#         x = self.final_conv(x)
#         return torch.cat([torch.sigmoid(x[:, 0:3]), torch.tanh(x[:, 3:4])], dim=1)

# # --- HELPERS ---
# TARGET_SIZE = 128

# def load_rgb_patch(tif_path):
#     img = Image.open(tif_path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
#     return np.array(img, dtype=np.float32) / 255.0

# def load_ndvi_patch(tif_path):
#     ndvi_img = Image.open(tif_path)
#     if ndvi_img.mode in ['F', 'I;16', 'I']:
#         ndvi_arr = np.array(ndvi_img.resize((128, 128), Image.BILINEAR), dtype=np.float32)
#     else:
#         ndvi_arr = np.array(ndvi_img.convert('L').resize((128, 128), Image.BILINEAR), dtype=np.float32)
#     if ndvi_arr.max() > 1.0: ndvi_arr = ndvi_arr / 255.0
#     return ndvi_arr

# def build_tensor(rgb_arr, ndvi_arr):
#     return torch.cat([torch.from_numpy(rgb_arr).permute(2, 0, 1), torch.from_numpy(ndvi_arr).unsqueeze(0)], dim=0)

# def compute_anomaly_score(original, reconstructed):
#     mask = (original != 0).float()
#     masked = ((original - reconstructed) ** 2) * mask
#     def masked_mean(diff, m): return (diff.sum() / (m.sum() + 1e-8)).item()
#     return masked_mean(masked, mask), masked_mean(masked[0:3], mask[0:3]), masked_mean(masked[3:4], mask[3:4])

# def interpret_score(score, rgb_score, ndvi_score):
#     THRESHOLDS = {"HEALTHY": 0.05, "MILD_STRESS": 0.10, "MODERATE": 0.20}
#     if score <= THRESHOLDS["HEALTHY"]: status, advice = "HEALTHY", "Normal."
#     elif score <= THRESHOLDS["MILD_STRESS"]: status, advice = "🟡 MILD STRESS", "Scout within 3-5 days."
#     elif score <= THRESHOLDS["MODERATE"]: status, advice = "🟠 MODERATE STRESS", "Prioritize visit."
#     else: status, advice = "🔴 CRITICAL", "Immediate inspection required."
#     return {"status": status, "advice": advice, "invisible_stress": (ndvi_score > rgb_score * 1.5) and (ndvi_score > 0.05)}

# # --- BATCH EXECUTION ---
# # def run_pipeline_batch(field_list_path, rgb_dir, ndvi_dir, model_path, output_dir):
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"    Autoencoder on {device.type.upper()}...")
    
    


    
# #     model = ComplexMultiModalAutoencoder().to(device)
# #     state = torch.load(model_path, map_location=device, weights_only=False)
# #     if "model_state_dict" in state: state = state["model_state_dict"]
# #     model.load_state_dict(state)
# #     model.eval()

# #     with open(field_list_path, "r") as f: fields = [line.strip() for line in f if line.strip()]
# #     print(f"     Found {len(fields)} valid crop patches to analyze. Let's look for anomalies!")

# #     records = []
# #     for i, field in enumerate(fields):
# #         if i > 0 and i % 500 == 0:
# #             print(f"       {i} out of {len(fields)} patches")



# #         # 1. Clean the file names
# #         base_name = os.path.splitext(field)[0]
# #         rgb_path = os.path.join(rgb_dir, base_name + ".tif")
# #         ndvi_path = os.path.join(ndvi_dir, base_name + ".tif")
        
# #         # 2. LOUDLY check if files exist before trying to open them
# #         if not os.path.exists(rgb_path):
# #             print(f"       MISSING RGB: Cannot find {rgb_path}")
# #             continue
# #         if not os.path.exists(ndvi_path):
# #             print(f"       MISSING NDVI: Cannot find {ndvi_path}")
# #             continue

# #         # 3. If both exist, run the Neural Network!
# #         tensor = build_tensor(load_rgb_patch(rgb_path), load_ndvi_patch(ndvi_path))
# #         with torch.no_grad():
# #             recon = model(tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
        
# #         overall, rgb_score, ndvi_score = compute_anomaly_score(tensor, recon)
# #         interp = interpret_score(overall, rgb_score, ndvi_score)
        
# #         records.append({
# #             "field": base_name, "overall": overall, "rgb": rgb_score, "ndvi": ndvi_score,
# #             "status": interp["status"], "advice": interp["advice"], "pre_symptomatic": interp["invisible_stress"]
# #         })



# #         # try:
# #         #     rgb_path = os.path.join(rgb_dir, field + ".tif")
# #         #     ndvi_path = os.path.join(ndvi_dir, field + ".tif")
            
# #         #     tensor = build_tensor(load_rgb_patch(rgb_path), load_ndvi_patch(ndvi_path))
# #         #     with torch.no_grad():
# #         #         recon = model(tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
            
# #         #     overall, rgb_score, ndvi_score = compute_anomaly_score(tensor, recon)
# #         #     interp = interpret_score(overall, rgb_score, ndvi_score)
            
# #         #     records.append({
# #         #         "field": field, "overall": overall, "rgb": rgb_score, "ndvi": ndvi_score,
# #         #         "status": interp["status"], "advice": interp["advice"], "pre_symptomatic": interp["invisible_stress"]
# #         #     })
# #         # except FileNotFoundError:
# #         #     continue

# #     del model
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()

# #     if records:
# #         os.makedirs(output_dir, exist_ok=True)
# #         csv_path = os.path.join(output_dir, "field_health_summary.csv")
# #         pd.DataFrame(records).to_csv(csv_path, index=False)
# #         print(f"     complete! summary saved to: {csv_path}")
# #         return csv_path
# #     return None


# def run_pipeline_batch(field_list_path, rgb_dir, ndvi_dir, model_path, output_dir, batch_size=16):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"    Autoencoder on {device.type.upper()}")
    
#     model = ComplexMultiModalAutoencoder().to(device)
#     state = torch.load(model_path, map_location=device, weights_only=False)
#     if "model_state_dict" in state: state = state["model_state_dict"]
#     model.load_state_dict(state)
#     model.eval()

#     with open(field_list_path, "r") as f: fields = [line.strip() for line in f if line.strip()]
    
#     # 1. Pre-filter the list so we only load files that actually exist
#     valid_fields = []
#     for field in fields:
#         base_name = os.path.splitext(field)[0]
#         rgb_path = os.path.join(rgb_dir, base_name + ".tif")
#         ndvi_path = os.path.join(ndvi_dir, base_name + ".tif")
#         if os.path.exists(rgb_path) and os.path.exists(ndvi_path):
#             valid_fields.append(base_name)
#         else:
#             print(f"      MISSING FILES: Skipping {base_name}")

#     print(f"     Found {len(valid_fields)} patches")

#     # 2. Setup the DataLoader for batching
#     dataset = CropPatchDataset(valid_fields, rgb_dir, ndvi_dir)
#     # num_workers=0 is safest for Windows to avoid multiprocess crashing. 
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#     records = []
    
#     # 3. Process in batches
#     for batch_idx, (batch_tensors, batch_names) in enumerate(dataloader):
#         if batch_idx % 10 == 0: # Print update every 10 batches
#             print(f"       Processing batch {batch_idx+1}/{len(dataloader)}")
            
#         batch_tensors = batch_tensors.to(device)
        
#         # Run the whole batch through the neural network at once
#         with torch.no_grad():
#             batch_recon = model(batch_tensors).cpu()
            
#         # Move the original tensors back to CPU for scoring
#         batch_tensors_cpu = batch_tensors.cpu()

#         # Score each image in the batch individually
#         for i in range(len(batch_names)):
#             base_name = batch_names[i]
#             orig_tensor = batch_tensors_cpu[i]
#             recon_tensor = batch_recon[i]
            
#             overall, rgb_score, ndvi_score = compute_anomaly_score(orig_tensor, recon_tensor)
#             interp = interpret_score(overall, rgb_score, ndvi_score)
            
#             records.append({
#                 "field": base_name, "overall": overall, "rgb": rgb_score, "ndvi": ndvi_score,
#                 "status": interp["status"], "advice": interp["advice"], "pre_symptomatic": interp["invisible_stress"]
#             })

#     # --- CRITICAL: FLUSH VRAM BEFORE RETURNING ---
#     del model
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # 4. Save and return
#     if records:
#         os.makedirs(output_dir, exist_ok=True)
#         csv_path = os.path.join(output_dir, "field_health_summary.csv")
#         pd.DataFrame(records).to_csv(csv_path, index=False)
#         print(f"     completed. summary saved to: {csv_path}")
#         return csv_path
        
#     return None    



"""
step4_inference.py  –  Jetson Nano optimised
=============================================
Key changes vs. original
  • FP16 (half-precision) inference  – ~2× VRAM saving; Jetson Maxwell/Pascal
    support float16 natively.  Activated when use_fp16=True.
  • Persistent DataLoader workers    – num_workers=2 on Linux (Jetson runs
    Ubuntu), pin_memory=True for faster Host→GPU transfer.
  • Non-blocking GPU transfer        – .to(device, non_blocking=True).
  • Prefetch overlap                 – DataLoader fetches next batch while GPU
    runs current batch.
  • Scoring stays on CPU             – anomaly math is trivial; keeping it on
    CPU avoids an extra round-trip.
  • Smaller TARGET_SIZE kept at 128  – already a good trade-off for Jetson.
  • Model deleted + cache cleared    – same as original but inside run fn.
"""

import os, gc
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import warnings
from typing import Optional
warnings.filterwarnings("ignore", category=UserWarning)
from torch.utils.data import Dataset, DataLoader


# ── Model definition (unchanged architecture) ─────────────────────────────────
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


# ── Scoring & interpretation (unchanged logic) ────────────────────────────────
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
    if   score <= THRESHOLDS["HEALTHY"]:  status, advice = "HEALTHY",            "Normal."
    elif score <= THRESHOLDS["MILD"]:     status, advice = "🟡 MILD STRESS",      "Scout within 3-5 days."
    elif score <= THRESHOLDS["MODERATE"]: status, advice = "🟠 MODERATE STRESS",  "Prioritize visit."
    else:                                 status, advice = "🔴 CRITICAL",          "Immediate inspection required."
    return {
        "status":  status,
        "advice":  advice,
        "invisible_stress": (ndvi_score > rgb_score * 1.5) and (ndvi_score > 0.05),
    }


# ── Main entry point ──────────────────────────────────────────────────────────
def run_pipeline_batch(field_list_path: str,
                       rgb_dir: str,
                       ndvi_dir: str,
                       model_path: str,
                       output_dir: str,
                       batch_size: int = 8,
                       use_fp16: bool = False) -> Optional[str]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Autoencoder on {device.type.upper()}"
          f"{'  (FP16)' if use_fp16 and device.type=='cuda' else ''}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = ComplexMultiModalAutoencoder().to(device)
    # state = torch.load(model_path, map_location=device, weights_only=False)
    state = torch.load(model_path, map_location=device)

    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    # Convert to FP16 on GPU
    if use_fp16 and device.type == "cuda":
        model = model.half()

    # ── Filter valid patches ──────────────────────────────────────────────────
    with open(field_list_path) as f:
        fields = [l.strip() for l in f if l.strip()]

    valid_fields = []
    for field in fields:
        base = os.path.splitext(field)[0]
        if (os.path.exists(os.path.join(rgb_dir,  base + ".tif")) and
                os.path.exists(os.path.join(ndvi_dir, base + ".tif"))):
            valid_fields.append(base)
        else:
            print(f"      MISSING: {base}  (skipped)")

    print(f"    {len(valid_fields)} valid field patches")

    # ── DataLoader ────────────────────────────────────────────────────────────
    # num_workers=2 is fine on Linux/Jetson; pin_memory speeds Host→GPU copy
    dataset    = CropPatchDataset(valid_fields, rgb_dir, ndvi_dir)
    dataloader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,                           # Jetson is Linux → safe
        pin_memory  = (device.type == "cuda"),     # faster DMA transfer
        prefetch_factor = 2,                       # pre-load 2 batches ahead
    )

    records = []
    total_batches = len(dataloader)

    for batch_idx, (batch_tensors, batch_names) in enumerate(dataloader):

        # ── Transfer to GPU ───────────────────────────────────────────────
        if use_fp16 and device.type == "cuda":
            batch_tensors = batch_tensors.half()
        batch_tensors = batch_tensors.to(device, non_blocking=True)

        # ── Forward pass ──────────────────────────────────────────────────
        with torch.no_grad():
            batch_recon = model(batch_tensors)

        # ── Move both to CPU for scoring (keeps GPU free for next batch) ──
        orig_cpu  = batch_tensors.float().cpu()
        recon_cpu = batch_recon.float().cpu()

        # ── Per-image scoring ─────────────────────────────────────────────
        for i, base_name in enumerate(batch_names):
            overall, rgb_s, ndvi_s = compute_anomaly_score(
                orig_cpu[i], recon_cpu[i])
            interp = interpret_score(overall, rgb_s, ndvi_s)
            records.append({
                "field":         base_name,
                "overall":       overall,
                "rgb":           rgb_s,
                "ndvi":          ndvi_s,
                "status":        interp["status"],
                "advice":        interp["advice"],
                "pre_symptomatic": interp["invisible_stress"],
            })

        if batch_idx % 10 == 0:
            print(f"      Batch {batch_idx+1}/{total_batches}", end="\r")

    print(f"      Batch {total_batches}/{total_batches}  ✓         ")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if not records:
        return None

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "field_health_summary.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"    ✓ Inference complete → {csv_path}")
    return csv_path