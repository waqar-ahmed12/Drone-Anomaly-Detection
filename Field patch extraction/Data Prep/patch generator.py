import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image

#A 123MB stitched image will trigger Pillow's safety limits 
Image.MAX_IMAGE_PIXELS = None

def generate_patches(patch_size=512):
    #Initialize the GUI window and hide the main background window
    root = tk.Tk()
    root.withdraw() 

    print("select image")
    # 2. Open the file selection dialog
    file_path = filedialog.askopenfilename(
        title="Select Stitched Image",
        filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Processing: {file_path}")
    
    #a dedicated folder for the patches next to the original image
    base_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(base_dir, f"{base_name}_patches")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Load and calculate dimensions
    try:
        img = Image.open(file_path)
        width, height = img.size
        print(f"Image loaded! Dimensions: {width}px by {height}px")

        # Calculate the total number of rows and columns needed
        total_rows = (height + patch_size - 1) // patch_size
        total_cols = (width + patch_size - 1) // patch_size

        print(f"Generating {total_rows * total_cols} patches. Please wait...")

        # 5. Slice the image
        for row in range(total_rows):
            for col in range(total_cols):
                
                # Define the pixel coordinates for the bounding box
                left = col * patch_size
                upper = row * patch_size
                right = min(left + patch_size, width)
                lower = min(upper + patch_size, height)

                # Crop the patch out of the main image
                patch = img.crop((left, upper, right, lower))

                # padding the patches that are not perfectly 512 px
                if patch.size != (patch_size, patch_size):
                    padded_patch = Image.new(img.mode, (patch_size, patch_size), "black")
                    padded_patch.paste(patch, (0, 0))
                    patch = padded_patch

                #the patch in TIF format
                patch_filename = f"patch_{row}_{col}.tif"
                patch_path = os.path.join(output_dir, patch_filename)
                
                patch.save(patch_path)

        print(f"\nsaved to:\n{output_dir}")

    except Exception as e:
        print(f"error occurred: {e}")

if __name__ == "__main__":
    generate_patches()