# import os
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.colors import LinearSegmentedColormap
# from PIL import Image, ImageDraw

# # --- MODEL DEFINITION ---
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
# def parse_row_col(filename):
#     parts = os.path.splitext(filename)[0].split("_")
#     return int(parts[1]), int(parts[2])

# def load_tensor(rgb_dir, ndvi_dir, filename):
#     name = os.path.splitext(filename)[0]
#     def find(directory):
#         for ext in ("", ".tif", ".tiff", ".TIF", ".TIFF"):
#             p = os.path.join(directory, name + ext)
#             if os.path.exists(p): return p
#         raise FileNotFoundError(f"{name} not found in {directory}")

#     rgb = Image.open(find(rgb_dir)).convert("RGB").resize((128, 128), Image.BILINEAR)
#     rgb_t = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1)

#     ndvi_img = Image.open(find(ndvi_dir))
#     if ndvi_img.mode in ["F", "I;16", "I"]:
#         ndvi_arr = np.array(ndvi_img.resize((128, 128), Image.BILINEAR), dtype=np.float32)
#     else:
#         ndvi_arr = np.array(ndvi_img.convert("L").resize((128, 128), Image.BILINEAR), dtype=np.float32)
#     if ndvi_arr.max() > 1.0: ndvi_arr = ndvi_arr / 255.0

#     return torch.cat([rgb_t, torch.from_numpy(ndvi_arr).unsqueeze(0)], dim=0)

# def to_rgb_display(arr):
#     rgb = arr[:3].transpose(1, 2, 0)
#     lo, hi = rgb.min(), rgb.max()
#     return (rgb - lo) / (hi - lo + 1e-8)

# def is_valid_field_patch(rgb_dir, ndvi_dir, filename, max_black_ratio=0.05):
#     try:
#         tensor = load_tensor(rgb_dir, ndvi_dir, filename)
#         rgb = tensor[:3]
#         black_pixels = torch.sum((rgb[0] == 0) & (rgb[1] == 0) & (rgb[2] == 0)).item()
#         total_pixels = rgb.shape[1] * rgb.shape[2]
#         return (black_pixels / total_pixels) <= max_black_ratio
#     except FileNotFoundError: return False

# # --- PLOTTING FUNCTIONS ---
# def plot_heatmap(df, flagged, output_dir):
#     rows, cols = [parse_row_col(f)[0] for f in df["field"]], [parse_row_col(f)[1] for f in df["field"]]
#     min_r, max_r, min_c, max_c = min(rows), max(rows), min(cols), max(cols)
#     grid_h, grid_w = max_r - min_r + 1, max_c - min_c + 1

#     grid, flag_grid = np.full((grid_h, grid_w), np.nan), np.zeros((grid_h, grid_w), dtype=bool)
#     for _, row in df.iterrows():
#         r, c = parse_row_col(row["field"])
#         grid[r - min_r, c - min_c] = row["overall"]
#         if row["field"] in flagged: flag_grid[r - min_r, c - min_c] = True

#     cmap = LinearSegmentedColormap.from_list("field", ["#2ecc71", "#f1c40f", "#e74c3c"])
#     cmap.set_bad(color="#1a1a2e")

#     fig, ax = plt.subplots(figsize=(14, 10))
#     fig.patch.set_facecolor("#1a1a2e")
#     ax.set_facecolor("#1a1a2e")
#     im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=np.nanmin(grid), vmax=np.nanmax(grid))

#     for r in range(grid_h):
#         for c in range(grid_w):
#             if flag_grid[r, c]:
#                 ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=2, edgecolor="#e67e22", facecolor="none"))

#     cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
#     cbar.set_label("Anomaly Score", color="white", fontsize=11)
#     cbar.ax.yaxis.set_tick_params(color="white")
#     plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

#     ax.legend(handles=[mpatches.Patch(edgecolor="#e67e22", facecolor="none", linewidth=2, label="Highest Stress Regions")], loc="upper right", facecolor="#0f3460", labelcolor="white")
#     ax.set_title("Field Anomaly Heatmap", color="white", fontsize=13, fontweight="bold", pad=12)
#     ax.set_xlabel("Column", color="white"), ax.set_ylabel("Row", color="white")
#     ax.tick_params(colors="white")

#     out = os.path.join(output_dir, "field_heatmap.png")
#     plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"  Heatmap saved {out}")

# def plot_flagged_patches(flagged_df, model, rgb_dir, ndvi_dir, output_dir):
#     device = next(model.parameters()).device
#     model.eval()
#     n = len(flagged_df)
#     if n == 0: return

#     fig, axes = plt.subplots(n, 5, figsize=(22, 4.5 * n))
#     fig.patch.set_facecolor("#1a1a2e")
#     if n == 1: axes = axes.reshape(1, 5)

#     for ax, title in zip(axes[0], ["Original RGB", "Reconstructed RGB", "Original NDVI", "Reconstructed NDVI", "Error Heatmap"]):
#         ax.set_title(title, color="white", fontsize=10, pad=6)

#     for i, (_, row) in enumerate(flagged_df.iterrows()):
#         fname, score, pre_s = row["field"], row["overall"], row.get("pre_symptomatic", False)
#         try: tensor = load_tensor(rgb_dir, ndvi_dir, fname)
#         except FileNotFoundError:
#             for ax in axes[i]: ax.axis("off")
#             continue

#         with torch.no_grad(): recon = model(tensor.unsqueeze(0).to(device)).squeeze(0).cpu()
#         orig_np, recon_np = tensor.numpy(), recon.numpy()
#         error = np.mean((orig_np - recon_np) ** 2, axis=0)

#         axes[i, 0].imshow(to_rgb_display(orig_np))
#         axes[i, 1].imshow(to_rgb_display(recon_np))
#         axes[i, 2].imshow(orig_np[3], cmap="RdYlGn", vmin=0, vmax=1)
#         axes[i, 3].imshow(recon_np[3], cmap="RdYlGn", vmin=0, vmax=1)
#         im = axes[i, 4].imshow(error, cmap="hot")
#         plt.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04)

#         for ax in axes[i]: ax.axis("off")
#         pre_label = "  |  Pre-symptomatic" if pre_s else ""
#         axes[i, 0].set_ylabel(f"{fname}\nScore: {score:.6f} [STRESS]{pre_label}", color="#e67e22", fontsize=9, rotation=0, labelpad=160, va="center")

#     fig.suptitle(f"Top {n} Most Anomalous Patches", color="white", fontsize=13, fontweight="bold", y=1.01)
#     plt.tight_layout()
#     out = os.path.join(output_dir, "flagged_patches.png")
#     plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"  Flagged patches saved  {out}")

# # --- MAIN RUNNER ---
# def run_visualization(csv_path, rgb_dir, ndvi_dir, model_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     df = pd.read_csv(csv_path)
#     if "invisible" in df.columns and "pre_symptomatic" not in df.columns:
#         df.rename(columns={"invisible": "pre_symptomatic"}, inplace=True)
    
#     print(f"     Filtering out pure black edge patches from visualization...")
#     valid_mask = df["field"].apply(lambda f: is_valid_field_patch(rgb_dir, ndvi_dir, f))
#     df_valid = df[valid_mask].copy()
#     print(f"     Dropped {len(df) - len(df_valid)} edge patches. {len(df_valid)} valid patches remain.")

#     # TARGET_ANOMALIES = min(100, len(df_valid))
#     TARGET_ANOMALIES = min(10, len(df_valid))
#     flagged = df_valid.nlargest(TARGET_ANOMALIES, "overall").copy()
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ComplexMultiModalAutoencoder().to(device)
#     state = torch.load(model_path, map_location=device, weights_only=False)
#     if "model_state_dict" in state: state = state["model_state_dict"]
#     model.load_state_dict(state)
#     model.eval()

#     print("\n     Generating field heatmap...")
#     plot_heatmap(df_valid, set(flagged["field"].tolist()), output_dir)
#     print("     Generating visual patch reconstructions...")
#     plot_flagged_patches(flagged, model, rgb_dir, ndvi_dir, output_dir)


# def stitch_master_map(df_all, flagged_names, rgb_dir, output_path):
#     """
#     Re-assembles the entire field and tints flagged patches red.
#     """
#     print("     Stitching big Map")
    
#     # Calculate grid size
#     rows = [parse_row_col(f)[0] for f in df_all["field"]]
#     cols = [parse_row_col(f)[1] for f in df_all["field"]]
#     max_r, max_c = max(rows), max(cols)
#     patch_size = 512 # Original slice size
    
#     # Create a blank canvas for the whole field
#     master_w = (max_c + 1) * patch_size
#     master_h = (max_r + 1) * patch_size
#     master_img = Image.new("RGB", (master_w, master_h), "black")
    
#     # Overlay for the red tint
#     overlay = Image.new("RGBA", (master_w, master_h), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)

#     for _, row in df_all.iterrows():
#         fname = row["field"]
#         r, c = parse_row_col(fname)
        
#         # Load the original high-res 512px patch
#         patch_path = os.path.join(rgb_dir, fname + ".tif")
#         if os.path.exists(patch_path):
#             p_img = Image.open(patch_path).convert("RGB")
#             x, y = c * patch_size, r * patch_size
#             master_img.paste(p_img, (x, y))
            
#             # If AI flagged it, draw a semi-transparent red box on the overlay
#             if fname in flagged_names:
#                 draw.rectangle([x, y, x + patch_size, y + patch_size], fill=(255, 0, 0, 80))

#     # Merge the red tint onto the master image
#     final_map = Image.alpha_composite(master_img.convert("RGBA"), overlay)
#     final_map.convert("RGB").save(output_path, "JPEG", quality=85)
#     print(f"     Master Map saved  {output_path}")

# # --- UPDATED RUN VISUALIZATION ---
# def run_visualization(csv_path, rgb_dir, ndvi_dir, model_path, workspace):
#     # Create a subfolder for final reports
#     report_dir = os.path.join(workspace, "final_reports")
#     os.makedirs(report_dir, exist_ok=True)

#     df = pd.read_csv(csv_path)
#     if "invisible" in df.columns and "pre_symptomatic" not in df.columns:
#         df.rename(columns={"invisible": "pre_symptomatic"}, inplace=True)
    
#     # Identify the "Bad" patches (Top 5% or Top 100)
#     # We use df_valid for calculation but we keep 'df' for the master map
#     valid_mask = df["field"].apply(lambda f: is_valid_field_patch(rgb_dir, ndvi_dir, f))
#     df_valid = df[valid_mask].copy()
    
#     # We'll flag the top 10% for the red master map tint
#     top_stressed = df_valid.nlargest(int(len(df_valid) * 0.1), "overall")
#     flagged_names = set(top_stressed["field"].tolist())

#     # 1. Generate the Heatmap
#     # (Using your plot_heatmap function from before)
#     # plot_heatmap(df_valid, flagged_names, report_dir)

#     # 2. Generate the Stitched Master Map
#     master_map_path = os.path.join(report_dir, "master_field_map_marked.jpg")
#     stitch_master_map(df, flagged_names, rgb_dir, master_map_path)

#     print(f"\nVISUALS done: {report_dir}")

# # [Model Definition and Helpers stay the same as before...]
# # (Keep ResidualBlock, ComplexMultiModalAutoencoder, parse_row_col, load_tensor, is_valid_field_patch)


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFile
import re

# Prevent errors with very large images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- MODEL DEFINITION (Same as Step 4) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x): return self.relu(self.conv(x) + x)

class ComplexMultiModalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_input = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.enc1  = nn.Sequential(nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
        self.res1  = ResidualBlock(64)
        self.enc2  = nn.Sequential(nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.res2  = ResidualBlock(128)
        self.enc3  = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.res3  = ResidualBlock(256)
        self.dec1     = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.res_dec1 = ResidualBlock(128)
        self.dec2     = nn.Sequential(nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, inplace=True))
        self.res_dec2 = ResidualBlock(64)
        self.dec3     = nn.Sequential(nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, inplace=True))
        self.res_dec3 = ResidualBlock(32)
        self.final_conv = nn.Conv2d(32, 4, 3, padding=1)

    def forward(self, x):
        x = self.res3(self.enc3(self.res2(self.enc2(self.res1(self.enc1(self.encoder_input(x)))))))
        x = self.res_dec3(self.dec3(self.res_dec2(self.dec2(self.res_dec1(self.dec1(x))))))
        x = self.final_conv(x)
        return torch.cat([torch.sigmoid(x[:, 0:3]), torch.tanh(x[:, 3:4])], dim=1)

# --- HELPERS ---
def parse_row_col(filename):
    # Extracts numbers even if name is 'patch_4_12.tif'
    nums = re.findall(r'\d+', filename)
    return int(nums[0]), int(nums[1])

def stitch_complete_map(rgb_dir, flagged_names, output_path):
    """
    Stitches EVERY patch in the directory (buildings + field) 
    and tints only the flagged field patches red.
    """
    print("     Stitching Complete")
    
    all_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.tif')]
    if not all_files: return

    # Determine canvas size
    max_r, max_c = 0, 0
    for f in all_files:
        r, c = parse_row_col(f)
        max_r, max_c = max(max_r, r), max(max_c, c)
    
    patch_size = 512
    master_w, master_h = (max_c + 1) * patch_size, (max_r + 1) * patch_size
    
    # Create Canvas
    master_img = Image.new("RGB", (master_w, master_h), (26, 26, 46)) # Dark blue background
    overlay = Image.new("RGBA", (master_w, master_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, fname in enumerate(all_files):
        if i % 500 == 0: print(f"      Restiching patch {i} of {len(all_files)}")
        
        r, c = parse_row_col(fname)
        x, y = c * patch_size, r * patch_size
        
        p_path = os.path.join(rgb_dir, fname)
        patch_img = Image.open(p_path).convert("RGB")
        master_img.paste(patch_img, (x, y))
        
        # Check if THIS specific patch name (without .tif) was flagged by AI
        pure_name = os.path.splitext(fname)[0]
        if pure_name in flagged_names:
            # Apply red tint (RGBA: Red, Green, Blue, Alpha/Opacity)
            draw.rectangle([x, y, x + patch_size, y + patch_size], fill=(255, 0, 0, 90))

    # Composite and save
    final = Image.alpha_composite(master_img.convert("RGBA"), overlay)
    final.convert("RGB").save(output_path, "JPEG", quality=80)
    print(f"     Full Master Map saved to: {output_path}")


def run_visualization(csv_path, rgb_dir, workspace):
    report_dir = os.path.join(workspace, "final_reports")
    os.makedirs(report_dir, exist_ok=True)

    # 1. Get the list of bad patches from the CSV
    df = pd.read_csv(csv_path)
    
    # We'll tint the top 10% most stressed patches red
    # You can change '0.10' to '0.05' if you want fewer red boxes
    num_to_tint = int(len(df) * 0.10)
    flagged = df.nlargest(num_to_tint, "overall")
    flagged_names = set(flagged["field"].tolist())

    print(f"     Flagged {len(flagged_names)} patches as red.")

    # 2. Run the stitcher
    master_map_path = os.path.join(report_dir, "master_marked_field.jpg")
    stitch_complete_map(rgb_dir, flagged_names, master_map_path)

    # 3. Save a small text file summary
    with open(os.path.join(report_dir, "stress_report.txt"), "w") as f:
        f.write("Fianl Report\n")
        f.write(f"Total field patches analyzed: {len(df)}\n")
        f.write(f"Patches marked as high-stress: {len(flagged_names)}\n\n")
        f.write("Top 5 Critical Patches:\n")
        for _, row in flagged.head(5).iterrows():
            f.write(f"- {row['field']} (Score: {row['overall']:.6f})\n")

    print(f"\n Report saved to '{report_dir}'")