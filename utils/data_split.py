import os
import shutil
import random
import config

def split_dataset():
    raw_dir = config.DATASET_RAW_DIR
    split_dir = config.DATASET_SPLIT_DIR
    train_ratio = config.TRAIN_RATIO
    val_ratio = config.VAL_RATIO
    random_seed = config.RANDOM_SEED
    classes = config.CLASSES

    random.seed(random_seed)

    print("Starting dataset split process...")
    print(f"Ratios: Train={train_ratio}, Val={val_ratio}, Test={config.TEST_RATIO}")

    if not os.path.exists(raw_dir):
        print(f"Error: Raw dataset directory not found: {raw_dir}")
        return

    # Dictionary to collect all image paths per class
    class_images = {cls: [] for cls in classes}

    # Search in both train and validation directories of raw dataset
    search_dirs = [
        os.path.join(raw_dir, "train", "images"),
        os.path.join(raw_dir, "validation", "images")
    ]

    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue

        for cls in classes:
            cls_dir = os.path.join(s_dir, cls)
            if not os.path.exists(cls_dir):
                # Fallback for minor naming variations (e.g. rolled-in_scale vs rolled-in-scale)
                alt_cls = cls.replace("_", "-")
                cls_dir_alt = os.path.join(s_dir, alt_cls)
                if os.path.exists(cls_dir_alt):
                    cls_dir = cls_dir_alt
                else:
                    print(f"Warning: Class directory not found: {cls_dir}")
                    continue

            # Gather all image files
            files = [
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            class_images[cls].extend(files)

    # Print summary
    total_images = sum(len(paths) for paths in class_images.values())
    print(f"Total images gathered across all classes: {total_images}")
    for cls, paths in class_images.items():
        print(f"Class '{cls}': {len(paths)} images")

    # Clear split directory if it exists
    if os.path.exists(split_dir):
        print(f"Cleaning existing split directory: {split_dir}")
        shutil.rmtree(split_dir)

    # Create new directories for train, val, and test splits
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, split, cls), exist_ok=True)

    # Perform stratified split and copy files
    for cls, paths in class_images.items():
        random.shuffle(paths)
        
        n_total = len(paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = paths[:n_train]
        val_files = paths[n_train:n_train + n_val]
        test_files = paths[n_train + n_val:]

        print(f"Splitting class '{cls}': Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

        # Helper to copy files
        def copy_files(file_list, target_split):
            for file_path in file_list:
                dest_dir = os.path.join(split_dir, target_split, cls)
                shutil.copy2(file_path, os.path.join(dest_dir, os.path.basename(file_path)))

        copy_files(train_files, "train")
        copy_files(val_files, "val")
        copy_files(test_files, "test")

    print("Dataset split process completed successfully!")

if __name__ == "__main__":
    split_dataset()
