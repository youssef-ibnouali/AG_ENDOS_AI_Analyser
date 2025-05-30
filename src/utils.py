import os
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from dataset import AGDataset
from preprocessing import preprocess_pil
from PIL import Image
import numpy as np

def show_preprocessing_comparison(dataset, batch_size=4):
    """
    For batch_size random samples, show (original | preprocessed) pairs.
    """
    # pick a few random indices
    indices = np.random.choice(len(dataset), batch_size, replace=False)

    fig, axes = plt.subplots(batch_size, 2, figsize=(6, 3 * batch_size))
    for row, idx in enumerate(indices):
        # 1) load raw image via PIL
        path, label = dataset.samples[idx]
        orig = Image.open(path).convert('RGB')

        # 2) apply your preprocess pipeline
        proc_pil = preprocess_pil(orig)

        # 3) convert both to numpy for plotting
        orig_np = np.array(orig)
        proc_np = np.array(proc_pil)

        # plot
        axes[row, 0].imshow(orig_np)
        axes[row, 0].set_title(f"Raw (label={label})")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(proc_np)
        axes[row, 1].set_title("Preprocessed")
        axes[row, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    ds = AGDataset(root_dir='data')
    show_preprocessing_comparison(ds, batch_size=4)
