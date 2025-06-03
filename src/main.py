import os
import glob
from train import main as train_main
from evaluate import main as eval_main
from augment_data import create_augmented_dataset

def clear(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return
    files = glob.glob(os.path.join(checkpoint_dir, "*"))
    for f in files: os.remove(f)

if __name__ == "__main__":
    clear("data/processed/train/normal")
    clear("data/processed/train/ag")
    clear("data/processed/val/normal")
    clear("data/processed/val/ag")
    clear("data/processed/test/normal")
    clear("data/processed/test/ag")

    print("=== PREPROCESSING & DATA AUGMENTATION ===")
    create_augmented_dataset()

    clear("checkpoints")
    print("=== TRAINING ===")
    train_main()

    print("\n=== EVALUATION ===")
    eval_main()