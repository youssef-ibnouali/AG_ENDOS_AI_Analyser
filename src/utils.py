import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from dataset import AGDataset

def show_batch(dataset, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    imgs, labels = next(iter(loader))
    grid = torchvision.utils.make_grid(imgs, nrow=batch_size)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0).numpy())
    plt.title(labels.tolist())
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    ds = AGDataset(root_dir='data')
    show_batch(ds)
