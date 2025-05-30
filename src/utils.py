import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import AGDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from PIL import Image

def unnormalize(tensor, mean, std):
    """Undo Normalize so that tensor is back in [0,1] for display."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0.0, 1.0)

def show_preprocessing_comparison(dataset, batch_size=4):
    """
    Show raw vs. (preprocessed + normalized) inputs.
    """
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    fig, axes = plt.subplots(batch_size, 2, figsize=(6, 3 * batch_size))

    # Unpack the mean/std you used in your Dataset
    mean = dataset.transform.transforms[-1].mean   # [0.485,0.456,0.406]
    std  = dataset.transform.transforms[-1].std    # [0.229,0.224,0.225]

    for row, idx in enumerate(indices):
        # 1- Raw image
        path, label = dataset.samples[idx]
        orig = Image.open(path).convert('RGB')

        # 2- Fully transformed tensor (includes preprocess_pil + Normalize)
        img_tensor, _ = dataset[idx]    # this calls preprocess_pil + ToTensor + Normalize

        # 3- Unnormalize for display
        img_disp = unnormalize(img_tensor.clone(), mean, std)
        #img_np   = img_tensor.permute(1,2,0).numpy()
        img_np   = img_disp.permute(1,2,0).numpy()

        # Plot
        axes[row, 0].imshow(orig)

        if label == 0: label = "normal stomach"
        else: label = "inflamed gastric mucosa"

        axes[row, 0].set_title(label)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(img_np)
        axes[row, 1].set_title("Preprocessed")
        axes[row, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ds = AGDataset(root_dir='data')
    show_preprocessing_comparison(ds, batch_size=4)
