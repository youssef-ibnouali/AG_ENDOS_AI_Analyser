import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from preprocessing import preprocess_pil

class AGDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        # construit la liste (path, label)
        subdirs = ['normal', 'ag']
        self.samples = []
        for label_idx, sub in enumerate(subdirs):
            folder = os.path.join(root_dir, 'raw', sub)
            for fname in os.listdir(folder):
                if fname.lower().endswith('.bmp') or fname.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(folder, fname), label_idx))

        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = preprocess_pil(img)
        img = self.transform(img)
        return img, label
