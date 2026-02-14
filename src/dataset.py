import torch
from torch.utils.data import Dataset 
import numpy as np
import pickle
import os
from glob import glob

class AnimalNumpyDataset(Dataset):
    def __init__(self , root , train=True, transform=None):
        self.root = root 
        self.train = train
        self.transform = transform
        self.class_to_label = {
            'cow': 0,
            'dog': 1,
            'duck': 2,
            'fish': 3,
            'raccoon': 4
        }
        
        if train :
            data_dir = os.path.join(root, "train")
        else :
            data_dir = os.path.join(root, "test")
        self.images = []
        self.labels = []

        npy_files = sorted(glob(os.path.join(data_dir, "*.npy")))
        
        for npy_file in npy_files:
            filename = os.path.basename(npy_file)
            class_name = filename.replace("full_numpy_bitmap_", "").replace(".npy", "")
            
            if class_name in self.class_to_label:
                data = np.load(npy_file)
                print(f"Loaded {filename}: shape {data.shape}")
                
                for img in data:
                    self.images.append(img)
                    self.labels.append(self.class_to_label[class_name])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__" :
    dataset = AnimalNumpyDataset(root = "../data_npy" , train = True)
       