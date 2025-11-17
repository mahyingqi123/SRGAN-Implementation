from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torchmetrics import StructuralSimilarityIndexMeasure
from pathlib import Path
from PIL import Image
import math
import json
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Import your generator from its file
from model_components.generator import Generator

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse.item())

def rgb_to_y_channel(img: torch.Tensor) -> torch.Tensor:
    if img.shape[1] != 3:
        raise ValueError("Input image must be RGB with 3 channels.")
    
    # Weights for RGB to Y conversion
    weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(img.device)
    return torch.sum(img * weights, dim=1, keepdim=True)

def preprocess_for_metrics(
    sr_img: torch.Tensor, 
    hr_img: torch.Tensor, 
    border_pixels: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]: 

    # Convert to Y-channel
    sr_y = rgb_to_y_channel(sr_img)
    hr_y = rgb_to_y_channel(hr_img)
    
    # Remove 4-pixel border 
    hr_y = hr_y[..., border_pixels:-border_pixels, border_pixels:-border_pixels]
    sr_y = sr_y[..., border_pixels:-border_pixels, border_pixels:-border_pixels]
    
    return sr_y, hr_y

def run_test():
    # Load configuration from JSON
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Test parameters from config
    MODEL_PATH = config.get('test_model_path', 'models/best_model.pth')
    TEST_DIR = config.get('test_data_path', 'test_images')
    OUTPUT_DIR = config.get('test_output_dir', 'results')
    UPSCALE_FACTOR = config.get('upscale_factor', 4)
    
    print("=" * 60)
    print("SRGAN Testing Configuration")
    print("=" * 60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Test Directory: {TEST_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Upscale Factor: {UPSCALE_FACTOR}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = Generator().to(device)
    
    print(f"\nLoading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']} epochs")
        if 'g_loss' in checkpoint:
            print(f"Final generator loss: {checkpoint['g_loss']:.4f}")
    else:
        generator.load_state_dict(checkpoint) # In case the file is just the state_dict
    
    generator.eval()
    print("Generator loaded successfully and set to eval mode.")
    
    # Setup Metrics 
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_psnr = 0
    total_ssim = 0
    
    hr_to_tensor = transforms.ToTensor() 
    
    lr_transform = transforms.Compose([
        transforms.ToTensor() # To [0, 1]
    ])

    # Process Test Dataset
    test_path = Path(TEST_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Support multiple image formats
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(sorted(list(test_path.glob(ext))))
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {TEST_DIR}")
        return

    print(f"\nTesting on {len(image_files)} images...")
    
    with torch.no_grad():
        for hr_image_path in image_files:
            print(f"Processing {hr_image_path.name}...")
            
            # Load HR image
            hr_image_pil = Image.open(hr_image_path).convert("RGB")
            
            # Create LR Image
            w, h = hr_image_pil.size
            lr_size = (w // UPSCALE_FACTOR, h // UPSCALE_FACTOR)
            lr_image_pil = hr_image_pil.resize(lr_size, Image.BICUBIC)

            # Create Bicubic Upscaled Image
            bicubic_image_pil = lr_image_pil.resize((w, h), Image.BICUBIC)
            
            # Prepare Tensors
            lr_tensor = lr_transform(lr_image_pil).unsqueeze(0).to(device)
            hr_tensor = hr_to_tensor(hr_image_pil).unsqueeze(0).to(device) 
            bicubic_tensor = hr_to_tensor(bicubic_image_pil).unsqueeze(0).to(device) 

            # Run Generator
            sr_tensor_tanh = generator(lr_tensor)
            
            # Un-normalize SR tensor 
            sr_tensor = (sr_tensor_tanh + 1.0) / 2.0
            
            # Calculate Metrics
            sr_y, hr_y = preprocess_for_metrics(sr_tensor, hr_tensor, UPSCALE_FACTOR)
            
            current_psnr = calculate_psnr(sr_y, hr_y)
            current_ssim = ssim(sr_y, hr_y).item()
            
            total_psnr += current_psnr
            total_ssim += current_ssim

            # Save Output Images
            base_name = hr_image_path.stem
            save_image(sr_tensor, output_path / f"{base_name}_SRGAN.png")
            save_image(bicubic_tensor, output_path / f"{base_name}_Bicubic.png")
            save_image(hr_tensor, output_path / f"{base_name}_Original.png")
            
            print(f"  PSNR: {current_psnr:.4f} dB | SSIM: {current_ssim:.4f}")

    # Report Averages
    avg_psnr = total_psnr / len(image_files)
    avg_ssim = total_ssim / len(image_files)
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Dataset: {TEST_DIR}")
    print(f"Total Images: {len(image_files)}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Generated images saved to '{OUTPUT_DIR}'")
    print("=" * 60)

if __name__ == "__main__":
    run_test()