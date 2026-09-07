import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


#We leave train and val transform inside train so we can customize


class UnifiedPedestrianDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.image_paths = self.df['image_path'].values
        self.attributes = self.df.iloc[:, 2:].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (128, 256), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        raw_labels = self.attributes[idx]
        mask = (raw_labels != -1.0).astype(np.float32)
        clean_labels = np.where(raw_labels == -1.0, 0.0, raw_labels)
        
        return image, torch.tensor(clean_labels), torch.tensor(mask)