import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def generate_patches(file_path, output_dir, patch_size=512, rotation=90):
    if not os.path.exists(file_path):
        print(f"Error: Could not find file at {file_path}")
        return

    print(f"Processing: {file_path}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        img = Image.open(file_path)
        
        if rotation != 0:
            print(f"Rotation:")
            if rotation == 90: img = img.transpose(Image.Transpose.ROTATE_90)
            elif rotation == 180: img = img.transpose(Image.Transpose.ROTATE_180)
            elif rotation == 270: img = img.transpose(Image.Transpose.ROTATE_270)
            
            ##Dont want to save 
            # preview_path = os.path.join(output_dir, f"{base_name}_rotated_preview.jpg")
            # img.convert('RGB').save(preview_path, "JPEG")

        width, height = img.size
        total_rows = (height + patch_size - 1) // patch_size
        total_cols = (width + patch_size - 1) // patch_size

        print(f"there are {total_rows * total_cols} patches")

        for row in range(total_rows):
            for col in range(total_cols):
                left = col * patch_size
                upper = row * patch_size
                right = min(left + patch_size, width)
                lower = min(upper + patch_size, height)

                patch = img.crop((left, upper, right, lower))

                if patch.size != (patch_size, patch_size):
                    padded_patch = Image.new(img.mode, (patch_size, patch_size), "black")
                    padded_patch.paste(patch, (0, 0))
                    patch = padded_patch

                patch_filename = f"patch_{row}_{col}.tif"
                patch.save(os.path.join(output_dir, patch_filename))

        print(f"Patches saved to: {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")