import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, transform=None):
        self.raw = raw_dataset
        self.transform = transform
    def __len__(self): return len(self.raw)
    def __getitem__(self, idx):
        img, label = self.raw[idx]
        if self.transform:
            img = self.transform(img)
        return img, label