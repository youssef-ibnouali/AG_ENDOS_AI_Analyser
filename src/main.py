import os
import glob
from train import main as train_main
from evaluate import main as eval_main

def clear_checkpoints(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return
    files = glob.glob(os.path.join(checkpoint_dir, "*"))
    for f in files: os.remove(f)

if __name__ == "__main__":
    clear_checkpoints("checkpoints")

    print("=== TRAINING ===")
    train_main()

    print("\n=== EVALUATION ===")
    eval_main()
