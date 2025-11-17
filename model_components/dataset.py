from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple, List
import torch

class SRGANDataset(Dataset):
    """
    Custom PyTorch Dataset for SRGAN.
    It loads HR images, performs a random 96x96 crop,
    and generates the corresponding LR image by bicubic downsampling.
    
    According to the paper's data processing:
    - 4x upscaling factor
    - Random 96x96 HR crops
    - Bicubic downsampling to get LR (24x24)
    - LR images normalized to [0, 1]
    - HR images normalized to [-1, 1]
    """
    def __init__(self, image_files: List[str], hr_crop_size: int = 96, upscale_factor: int = 4) -> None:
        super(SRGANDataset, self).__init__()
        
        self.image_files = image_files
        self.hr_crop_size = hr_crop_size
        self.upscale_factor = upscale_factor
        self.lr_crop_size = hr_crop_size // upscale_factor

        # Random crop 
        self.hr_crop = transforms.RandomCrop(self.hr_crop_size)
        
        # ToTensor and Normalize for HR 
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # to [-1, 1]
        ])
        
        # Downsample by a factor of 4 using bicubic interpolation, To [0, 1]
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor() # to [0, 1]
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_image_path = self.image_files[index]
        hr_image = Image.open(hr_image_path).convert("RGB")
        
        hr_cropped = self.hr_crop(hr_image)
        
        lr_image = self.lr_transform(hr_cropped)
        
        hr_image = self.hr_transform(hr_cropped)
        
        return lr_image, hr_image

    def __len__(self) -> int:
        return len(self.image_files)