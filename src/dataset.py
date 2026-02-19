import torch
from torch.utils.data import Dataset 
import numpy as np
import os
from glob import glob
import cv2

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
        
        self.data_files = []  
        self.indices = []    
        npy_files = sorted(glob(os.path.join(data_dir, "*.npy")))
        
        for file_idx, npy_file in enumerate(npy_files):
            filename = os.path.basename(npy_file)
            class_name = filename.replace("full_numpy_bitmap_", "").replace(".npy", "")
            
            if class_name in self.class_to_label:
                print(f"Loading {filename}...", end=" ")
                data = np.load(npy_file)
                self.data_files.append((data, self.class_to_label[class_name]))
                print(f"shape {data.shape}")
                for img_idx in range(len(data)):
                    self.indices.append((file_idx, img_idx))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx, img_idx = self.indices[idx]
        data, label = self.data_files[file_idx]
        
        image = data[img_idx]
        # Reshape from 784 to 28x28
        image = image.reshape(28, 28).astype(np.uint8)
        # Resize to 64x64 
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        return image, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data_npy") 
    dataset = AnimalNumpyDataset(root=data_path, train=True)
    print(f"Total samples: {len(dataset)}")
    if len(dataset) > 0:
        sample_img, sample_label = dataset[0]
        print(f"Tensor shape of one sample: {sample_img.shape}")
        print(f"Label of the first sample: {sample_label}")
    else:
        print("Warning: Không tìm thấy file dữ liệu nào! Hãy kiểm tra lại thư mục.")
    
    


