import os
import sys

# Add the project root folder to sys.path so config can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import config

def run_eda():
    print("Starting Exploratory Data Analysis (EDA)...")
    
    split_dir = config.DATASET_SPLIT_DIR
    classes = config.CLASSES
    
    if not os.path.exists(split_dir):
        print(f"Error: Split dataset directory not found at {split_dir}. Please run data_split first.")
        return

    # 1. Dataset Statistics & Class Distribution
    stats = []
    corrupted_files = []
    dimensions = set()
    pixel_means = {cls: [] for cls in classes}
    pixel_stds = {cls: [] for cls in classes}
    all_sample_images = {}

    for split in ["train", "val", "test"]:
        for cls in classes:
            cls_dir = os.path.join(split_dir, split, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Directory not found: {cls_dir}")
                continue
                
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            stats.append({
                "Split": split,
                "Class": cls,
                "Count": len(files)
            })
            
            # Read sample images and check dimensions/pixel stats for training set
            if split == "train" and len(files) > 0:
                # Save the first image as a sample for visualization
                sample_path = os.path.join(cls_dir, files[0])
                all_sample_images[cls] = sample_path
                
                # Check all files in train for dimensions, pixel distribution, and corruption
                for f in files:
                    f_path = os.path.join(cls_dir, f)
                    try:
                        with Image.open(f_path) as img:
                            w, h = img.size
                            mode = img.mode
                            dimensions.add((w, h, mode))
                            
                            # Convert to numpy array for pixel distribution analysis
                            arr = np.array(img)
                            pixel_means[cls].append(np.mean(arr))
                            pixel_stds[cls].append(np.std(arr))
                    except Exception as e:
                        print(f"Corrupted image found: {f_path}. Error: {e}")
                        corrupted_files.append(f_path)

    df_stats = pd.DataFrame(stats)
    print("Dataset split statistics:\n")
    print(df_stats.to_string(index=False))
    
    # Check dimensions
    print(f"Unique image dimensions found in training set (width, height, mode): {list(dimensions)}")
    if len(corrupted_files) == 0:
        print("Sanity check passed: 0 corrupted or missing files detected.")
    else:
        print(f"Warning: Detected {len(corrupted_files)} corrupted files: {corrupted_files}")

    # Plot 1: Class Distribution per split
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_stats, x="Class", y="Count", hue="Split", palette="muted")
    plt.title("Class Distribution across Train, Val, and Test Splits", fontsize=14, fontweight='bold')
    plt.xlabel("Defect Class", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    dist_plot_path = os.path.join(config.OUTPUT_DIR, "class_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Class distribution plot saved to: {dist_plot_path}")

    # Plot 2: Sample Images Grid (6 classes)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, cls in enumerate(classes):
        if cls in all_sample_images:
            img_path = all_sample_images[cls]
            img = Image.open(img_path)
            axes[i].imshow(img, cmap='gray' if img.mode == 'L' else None)
            axes[i].set_title(f"Class: {cls.replace('_', ' ').capitalize()}", fontsize=12, fontweight='bold')
        axes[i].axis("off")
    plt.suptitle("Sample Steel Surface Defect Images (NEU-CLS)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    grid_plot_path = os.path.join(config.OUTPUT_DIR, "sample_images.png")
    plt.savefig(grid_plot_path, dpi=300)
    plt.close()
    print(f"Sample image grid saved to: {grid_plot_path}")

    # Plot 3: Pixel Value Distribution
    plt.figure(figsize=(10, 6))
    for cls in classes:
        means = pixel_means[cls]
        if len(means) > 0:
            sns.kdeplot(means, label=cls, fill=True, alpha=0.1)
    plt.title("Pixel Intensity Distribution Mean across Classes (Train Set)", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Pixel Intensity (0-255)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Classes")
    plt.tight_layout()
    pixel_plot_path = os.path.join(config.OUTPUT_DIR, "pixel_distribution.png")
    plt.savefig(pixel_plot_path, dpi=300)
    plt.close()
    print(f"Pixel value distribution plot saved to: {pixel_plot_path}")

    # Print summary statistics per class
    print("Summary Statistics of Pixel Intensity per Class (Training Set):")
    for cls in classes:
        means = pixel_means[cls]
        stds = pixel_stds[cls]
        if len(means) > 0:
            print(f"Class '{cls}' - Mean Pixel Value: {np.mean(means):.2f}, Std Dev: {np.mean(stds):.2f}")

    print("EDA completed successfully!")

if __name__ == "__main__":
    run_eda()
