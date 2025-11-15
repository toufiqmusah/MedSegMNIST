import os
import numpy as np
import gdown
from torch.utils.data import Dataset

__all__ = ["MedSegMNIST3D", "BrainTumor"]

DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".medsegmnist")

class MedSegMNIST3D(Dataset):
    flag = None 
    available_sizes = []  
    
    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        download=False,
        root=DEFAULT_ROOT,
        size=None,
        mmap_mode=None,
    ):
        """
        Args:
            split (str): 'train', 'val' or 'test'
            transform (callable, optional): Transform for images
            target_transform (callable, optional): Transform for masks
            download (bool): Download if not exists
            root (str): Root directory for datasets
            size (int, optional): Image size (128, 224 ...) . If None, uses default
            mmap_mode (str, optional): numpy mmap mode for memory efficiency
        """

        if size is None:
            self.size = self.available_sizes[0]  
            self.size_flag = ""
        else:
            if size not in self.available_sizes:
                raise ValueError(f"Size {size} not available. Choose from: {self.available_sizes}")
            
            self.size = size
            self.size_flag = f"_{size}"
        
        if root is not None:
            self.root = root
            os.makedirs(self.root, exist_ok=True)
        else:
            raise RuntimeError("Failed to setup root directory. Please specify a valid root path.")
        
        if download:
            self.download()
        
        npz_path = os.path.join(self.root, f"{self.flag}-{self.size_flag}.npz")
        if not os.path.exists(npz_path):
            raise RuntimeError(
                f"Dataset not found at {npz_path}. Set download=True to download it."
            )
        
        # Load data
        npz_file = np.load(npz_path, mmap_mode=mmap_mode)
        
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if split in ["train", "val", "test"]:
            self.images = npz_file[f"{split}_mris"]
            self.masks = npz_file[f"{split}_masks"]
        else:
            raise ValueError(f"Invalid split: {split}. Choose from: train, val, test")
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        """
        Returns:
            image: 3D volume array
            mask: 3D segmentation mask
        """
        image = self.images[index]
        mask = self.masks[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        
        return image, mask
    
    def __repr__(self):
        head = f"Dataset {self.__class__.__name__} (size={self.size})"
        body = [
            f"Number of samples: {len(self)}",
            f"Root: {self.root}",
            f"Split: {self.split}",
            f"Image shape: {self.images[0].shape}",
        ]
        lines = [head] + ["  " + line for line in body]
        return "\n".join(lines)
    
    def download(self):
        """Download dataset from Google Drive"""
        if not hasattr(self, 'gdrive_ids'):
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} must define 'gdrive_ids' dict"
            )
        
        if self.size not in self.gdrive_ids:
            raise ValueError(f"No download link for size {self.size}. Available: {list(self.gdrive_ids.keys())}")
        
        filename = f"{self.flag}{self.size_flag}.npz"
        filepath = os.path.join(self.root, filename)
        
        if os.path.exists(filepath):
            print(f"Dataset already exists at {filepath}")
            return
        
        print(f"Downloading {filename} to {self.root}...")
        file_id = self.gdrive_ids[self.size]
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, filepath, quiet=False)
            print("Download complete!")
        except Exception as e:
            raise RuntimeError(
                f"Download failed: {e}\n"
                f"Please manually download from: {url}\n"
                f"And place it at: {filepath}"
            )
    
    def get_data(self):
        """Return raw numpy arrays (for non-DataLoader usage)"""
        return self.images, self.masks


class BrainTumor(MedSegMNIST3D):
    """BraTS Brain Tumor Segmentation Dataset (T2-Flair + Whole Tumor Segmentation)"""
    
    flag = "braintumor"
    available_sizes = [128, 224]
    
    gdrive_ids = {
        128: "1Fm2YSTWF72KD0YtV-Nv8f9EvliLqF1SC",
        # 224: "1def456_YOUR_224_FILE_ID",
    }

"""
if __name__ == "__main__":
    # Example 1: Simple data loading (no DataLoader)
    dataset = BrainTumor(split="train", size=224, download=True)
    images, masks = dataset.get_data()
    print(f"Loaded {len(images)} samples")
    
    # Example 2: Use with PyTorch DataLoader
    from torch.utils.data import DataLoader
    train_dataset = BrainTumor(split="train", size=224, download=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    for batch_images, batch_masks in train_loader:
        print(f"Batch shape: {batch_images.shape}")
        break
    
    # Example 3: Memory-efficient loading for large datasets
    dataset_mmap = BrainTumor(
        split="train", 
        size=224, 
        download=False,
        mmap_mode='r'  # Read from disk instead of loading to RAM
    )
    
    # Example 4: Pretty print
    print(dataset)
"""