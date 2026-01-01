"""
Step 1: Data Acquisition
Download SpaceNet 7 AOI_11_Rotterdam dataset from AWS S3
Optimized for Google Colab Free Tier
"""

import subprocess
import os
import shutil
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = "/content/spacenet_data"
AOI = "AOI_11_Rotterdam"
MAX_IMAGES = 500  # Limit to prevent storage overflow

# ============================================
# STEP 1.1: Install AWS CLI
# ============================================
def install_dependencies():
    """Install required packages for data download"""
    print("📦 Installing AWS CLI and dependencies...")
    
    # Install awscli
    subprocess.run(["pip", "install", "awscli", "--quiet"], check=True)
    
    # Verify installation
    result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
    print(f"✅ AWS CLI installed: {result.stdout.strip()}")

# ============================================
# STEP 1.2: Download SpaceNet 7 Data
# ============================================
def download_spacenet_data():
    """
    Download AOI_11_Rotterdam from SpaceNet S3 bucket
    Uses --no-sign-request for public bucket access
    """
    print(f"\n🛰️ Downloading SpaceNet 7 - {AOI}...")
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # S3 bucket path for SpaceNet 7
    s3_base = "s3://spacenet-dataset/spacenet/SN7_buildings/train/"
    s3_path = f"{s3_base}{AOI}/"
    
    # Download images directory
    images_dir = os.path.join(DATA_DIR, AOI, "images")
    labels_dir = os.path.join(DATA_DIR, AOI, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"📂 Downloading to: {DATA_DIR}")
    
    # Download images (mosaic images for each time step)
    # SpaceNet 7 structure: AOI_XX/images/ and AOI_XX/labels/
    
    cmd_images = [
        "aws", "s3", "sync",
        f"{s3_path}images/",
        images_dir,
        "--no-sign-request",
        "--quiet"
    ]
    
    cmd_labels = [
        "aws", "s3", "sync",
        f"{s3_path}labels/",
        labels_dir,
        "--no-sign-request",
        "--quiet"
    ]
    
    print("📥 Downloading images...")
    subprocess.run(cmd_images, check=True)
    
    print("📥 Downloading labels (building masks)...")
    subprocess.run(cmd_labels, check=True)
    
    print("✅ Download complete!")
    return images_dir, labels_dir

# ============================================
# STEP 1.3: Limit Dataset Size
# ============================================
def limit_dataset(images_dir, max_images=MAX_IMAGES):
    """
    Limit the number of images to prevent storage overflow
    SpaceNet 7 has multiple time-series images per AOI
    """
    print(f"\n🔧 Limiting dataset to {max_images} images...")
    
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.endswith(('.tif', '.png', '.jpg'))
    ])
    
    total_images = len(image_files)
    print(f"📊 Found {total_images} images")
    
    if total_images > max_images:
        # Remove excess images
        images_to_remove = image_files[max_images:]
        for img in images_to_remove:
            img_path = os.path.join(images_dir, img)
            os.remove(img_path)
        print(f"🗑️ Removed {len(images_to_remove)} excess images")
    
    # Final count
    remaining = len([f for f in os.listdir(images_dir) if f.endswith(('.tif', '.png', '.jpg'))])
    print(f"✅ Dataset limited to {remaining} images")
    
    return remaining

# ============================================
# STEP 1.4: Convert TIF to PNG (if needed)
# ============================================
def convert_tif_to_png(images_dir):
    """
    Convert GeoTIFF images to PNG for easier processing
    Qwen2-VL works better with standard image formats
    """
    print("\n🔄 Converting TIF images to PNG...")
    
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        subprocess.run(["pip", "install", "Pillow", "numpy", "--quiet"], check=True)
        from PIL import Image
        import numpy as np
    
    try:
        import rasterio
    except ImportError:
        subprocess.run(["pip", "install", "rasterio", "--quiet"], check=True)
        import rasterio
    
    png_dir = os.path.join(os.path.dirname(images_dir), "images_png")
    os.makedirs(png_dir, exist_ok=True)
    
    tif_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    
    converted = 0
    for tif_file in tif_files:
        try:
            tif_path = os.path.join(images_dir, tif_file)
            png_path = os.path.join(png_dir, tif_file.replace('.tif', '.png'))
            
            # Read GeoTIFF
            with rasterio.open(tif_path) as src:
                # Read RGB bands (SpaceNet uses bands 1,2,3 for RGB)
                if src.count >= 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                    
                    # Stack and normalize
                    rgb = np.dstack((r, g, b))
                    
                    # Normalize to 0-255 range
                    rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # Save as PNG
                    img = Image.fromarray(rgb_norm)
                    img.save(png_path)
                    converted += 1
                    
        except Exception as e:
            print(f"⚠️ Error converting {tif_file}: {e}")
            continue
    
    print(f"✅ Converted {converted} images to PNG")
    return png_dir

# ============================================
# STEP 1.5: Dataset Statistics
# ============================================
def print_dataset_stats(data_dir):
    """Print dataset statistics"""
    print("\n📊 Dataset Statistics:")
    print("=" * 50)
    
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        
        # Count files by extension
        ext_count = {}
        for f in files:
            ext = os.path.splitext(f)[1]
            ext_count[ext] = ext_count.get(ext, 0) + 1
        
        for ext, count in ext_count.items():
            print(f"{indent}  └── {count} {ext} files")
    
    # Calculate total size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(data_dir)
        for filename in filenames
    ) / (1024 * 1024 * 1024)  # Convert to GB
    
    print(f"\n💾 Total dataset size: {total_size:.2f} GB")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    print("=" * 60)
    print("🚀 SpaceNet 7 Data Acquisition Script")
    print("=" * 60)
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Download data
    images_dir, labels_dir = download_spacenet_data()
    
    # Step 3: Limit dataset
    limit_dataset(images_dir, MAX_IMAGES)
    
    # Step 4: Convert to PNG
    png_dir = convert_tif_to_png(images_dir)
    
    # Step 5: Print statistics
    print_dataset_stats(DATA_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Data acquisition complete!")
    print(f"📂 Images (PNG): {png_dir}")
    print(f"📂 Labels: {labels_dir}")
    print("=" * 60)
    
    return png_dir, labels_dir

if __name__ == "__main__":
    png_dir, labels_dir = main()
