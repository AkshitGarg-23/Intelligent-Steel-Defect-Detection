import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import config

class DefectDataset(Dataset):
    """
    Custom PyTorch Dataset for loading steel surface defect images.
    """
    def __init__(self, split_dir, split, transform=None):
        self.split_dir = os.path.join(split_dir, split)
        self.transform = transform
        self.classes = config.CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split folder does not exist: {self.split_dir}")
            
        # Collect all image files
        for cls_name in self.classes:
            cls_folder = os.path.join(self.split_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
                
            for filename in os.listdir(cls_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(cls_folder, filename))
                    self.labels.append(self.class_to_idx[cls_name])
                    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to 3-channel RGB (NEU images are gray)
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = transforms.ToTensor()(img_rgb)
            
        return img_tensor, label

def get_transforms(split):
    """
    Returns image augmentations and normalization transforms based on split.
    """
    normalize = transforms.Normalize(
        mean=config.NORM_MEAN,
        std=config.NORM_STD
    )
    
    if split == "train":
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomRotation(config.ROTATE_DEGREES),
            transforms.RandomHorizontalFlip(p=config.FLIP_PROB),
            transforms.ColorJitter(
                brightness=config.BRIGHTNESS,
                contrast=config.CONTRAST
            ),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation/Test splits use deterministic preprocessing
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            normalize
        ])
