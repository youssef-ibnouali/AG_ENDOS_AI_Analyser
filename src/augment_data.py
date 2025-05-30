import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Config
RAW_DIR       = "data/raw"       # data/raw/ag and data/raw/normal
PROCESSED_DIR = "data/processed" # will contain train/val/test splits
SPLITS        = {"train":0.7, "val":0.15, "test":0.15}
N_AUG         = 10
SEED          = 42

# Define augmentations and preprocess transforms
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop((224,224), scale=(0.8,1.0), ratio=(0.9,1.1)),
])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

def split_filenames(filenames, ratios):
    random.shuffle(filenames)
    n = len(filenames)
    train_end = int(ratios["train"] * n)
    val_end   = train_end + int(ratios["val"] * n)
    return {
        "train": filenames[:train_end],
        "val":   filenames[train_end:val_end],
        "test":  filenames[val_end:]
    }

def create_augmented_dataset():
    random.seed(SEED)

    # Prepare output directories
    for split in SPLITS:
        for label in ("ag", "normal"):
            os.makedirs(os.path.join(PROCESSED_DIR, split, label), exist_ok=True)

    # Process each class folder
    for label in ("ag", "normal"):
        src_folder = os.path.join(RAW_DIR, label)
        all_files = [f for f in os.listdir(src_folder)
                     if f.lower().endswith(('.bmp','.jpg','.png'))]
        splits = split_filenames(all_files, SPLITS)

        for split, files in splits.items():
            dst_folder = os.path.join(PROCESSED_DIR, split, label)
            for fname in tqdm(files, desc=f"Augmenting {label} [{split}]"):
                path = os.path.join(src_folder, fname)
                img = Image.open(path).convert('RGB')
                # Preprocess
                img_proc = preprocess(img)
                base, ext = os.path.splitext(fname)
                # Save original
                img_proc.save(os.path.join(dst_folder, f"{base}_orig{ext}"))
                # Create and save augmentations
                for i in range(N_AUG):
                    aug = augmentations(img_proc)
                    aug.save(os.path.join(dst_folder, f"{base}_aug{i}{ext}"))

if __name__ == "__main__":
    create_augmented_dataset()
