import os
import glob
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from preprocessing import preprocess_pil


# ------------- CONFIGURATION -------------
RAW_DIR       = "data/raw"       # data/raw/normal, data/raw/ag
PROCESSED_DIR = "data/processed" # will contain train/val/test/normal and /ag
SPLITS        = {"train": 0.6, "val": 0.25, "test": 0.15}
N_AUG         = 10              # number of augmentations per train image
SEED          = 42

# 1) First, define a transform that resizes the cleaned PIL image to 224×224
resize_to_224 = transforms.Compose([
    transforms.Resize((224, 224)),
])

# 2) Then define your augmentation pipeline (applied only on train)
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop((224,224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
])

# ------------- HELPERS -------------
def split_filenames(filenames, ratios):
    """
    Given a list of filenames, return a dict with three lists: train/val/test,
    ensuring at least one file per split if possible.
    """
    random.shuffle(filenames)
    n = len(filenames)
    train_cnt = max(1, int(ratios["train"] * n))
    val_cnt   = max(1, int(ratios["val"] * n))
    test_cnt  = n - train_cnt - val_cnt

    # If test is zero but you have >=3 samples, force at least one
    if test_cnt < 1 and n >= 3:
        test_cnt = 1
        if train_cnt > val_cnt:
            train_cnt -= 1
        else:
            val_cnt -= 1

    train_files = filenames[:train_cnt]
    val_files   = filenames[train_cnt:train_cnt+val_cnt]
    test_files  = filenames[train_cnt+val_cnt:train_cnt+val_cnt+test_cnt]
    return {"train": train_files, "val": val_files, "test": test_files}

# ------------- MAIN SCRIPT -------------
def create_augmented_dataset():
    random.seed(SEED)

    # Create output directories
    for split in SPLITS:
        for label in ("normal", "ag"):
            os.makedirs(os.path.join(PROCESSED_DIR, split, label), exist_ok=True)

    # Process each class folder
    for label in ("normal", "ag"):
        src_folder = os.path.join(RAW_DIR, label)
        all_files = [f for f in os.listdir(src_folder)
                     if f.lower().endswith((".bmp", ".jpg", ".png"))]
        splits = split_filenames(all_files, SPLITS)

        for split, files in splits.items():
            dst_folder = os.path.join(PROCESSED_DIR, split, label)
            for fname in tqdm(files, desc=f"Processing {label} [{split}]"):
                src_path = os.path.join(src_folder, fname)

                # 1) Load raw and run full preprocessing (OpenCV pipeline)
                img_raw = Image.open(src_path).convert("RGB")
                img_clean = preprocess_pil(img_raw)     # PIL image, cleaned

                # 2) Resize the cleaned image to 224×224
                img_resized = resize_to_224(img_clean)

                base, ext = os.path.splitext(fname)
                out_orig = os.path.join(dst_folder, f"{base}_orig{ext}")
                img_resized.save(out_orig)

                # 3) If this is train, generate augmentations from that cleaned+resized base
                if split == "train":
                    for i in range(N_AUG):
                        img_aug = augmentations(img_resized)
                        img_aug.save(os.path.join(dst_folder, f"{base}_aug{i}{ext}"))

if __name__ == "__main__":
    create_augmented_dataset()
