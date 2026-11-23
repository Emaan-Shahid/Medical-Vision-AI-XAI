import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    """Training: gentle random changes. Validation/test: no randomness."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BrainMRIDataset(Dataset):
    """Reads one picture at a time and returns (image tensor, class number)."""

    def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform=None):
        self.paths = df['path'].tolist()
        self.labels = [class_to_idx[l] for l in df['label']]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def make_loaders(splits_df, class_to_idx, batch_size=32, num_workers=2):
    """Build train / validation / test loaders from the saved splits table."""
    loaders = {}
    for group in ['train', 'val', 'test']:
        part = splits_df[splits_df.group == group]
        ds = BrainMRIDataset(part, class_to_idx, get_transforms(train=(group == 'train')))
        loaders[group] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(group == 'train'),
                                    num_workers=num_workers, pin_memory=True)
    return loaders
    
