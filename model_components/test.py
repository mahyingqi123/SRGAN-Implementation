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
import argparse
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Import your generator from its file
from model_components.generator import Generator

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculates PSNR on [0, 1] range tensors."""
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse.item())

def rgb_to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """
    Converts an RGB image tensor to the Y-channel (luminance)
    using the standard formula (Y = 0.299R + 0.587G + 0.114B).
    """
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
    """
    Prepares images for metric calculation, as per the paper:
    1. Converts to Y-channel.
    2. Removes a 4-pixel border from each side.
    """
    # Convert to Y-channel
    sr_y = rgb_to_y_channel(sr_img)
    hr_y = rgb_to_y_channel(hr_img)
    
    # Remove 4-pixel border 
    hr_y = hr_y[..., border_pixels:-border_pixels, border_pixels:-border_pixels]
    sr_y = sr_y[..., border_pixels:-border_pixels, border_pixels:-border_pixels]
    
    return sr_y, hr_y

def run_test():
    # --- 1. Setup Arguments ---
    parser = argparse.ArgumentParser(description="Test SRGAN Generator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained generator checkpoint.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the test dataset (e.g., Set14, BSD100).")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save generated images.")
    parser.add_argument("--upscale_factor", type=int, default=4, help="Upscaling factor (must match model).")
    args = parser.parse_args()

    # --- 2. Setup Model and Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Generator
    generator = Generator().to(device)
    
    # Load weights
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint) # In case the file is just the state_dict
    
    # "During test time we turn batch-normalization update off" [cite: 245]
    generator.eval()
    
    # --- 3. Setup Metrics ---
    # SSIM metric, expects [0, 1] range, data_range=1.0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_psnr = 0
    total_ssim = 0
    
    # --- 4. Setup Image Transforms ---
    hr_to_tensor = transforms.ToTensor() # Converts PIL [0, 255] to Tensor [0, 1]
    
    # LR transform (for generator input)
    # The generator expects [0, 1] range, not [-1, 1] as per our dataset.py
    lr_transform = transforms.Compose([
        transforms.ToTensor() # To [0, 1]
    ])

    # --- 5. Process Test Directory ---
    test_path = Path(args.test_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(list(test_path.glob("*.png")))
    image_files.extend(sorted(list(test_path.glob("*.jpg"))))
    image_files.extend(sorted(list(test_path.glob("*.bmp"))))

    print(f"Testing on {len(image_files)} images...")
    
    with torch.no_grad():
        for hr_image_path in image_files:
            print(f"Processing {hr_image_path.name}...")
            
            # Load HR image
            hr_image_pil = Image.open(hr_image_path).convert("RGB")
            
            # --- Create LR Image (Bicubic) ---
            # "I_LR is obtained by... a downsampling operation" [cite: 117]
            # We match the training pipeline's bicubic downsampling
            w, h = hr_image_pil.size
            lr_size = (w // args.upscale_factor, h // args.upscale_factor)
            lr_image_pil = hr_image_pil.resize(lr_size, Image.BICUBIC)

            # --- Create Bicubic Upscaled Image (for comparison) ---
            bicubic_image_pil = lr_image_pil.resize((w, h), Image.BICUBIC)
            
            # --- Prepare Tensors ---
            lr_tensor = lr_transform(lr_image_pil).unsqueeze(0).to(device)
            hr_tensor = hr_to_tensor(hr_image_pil).unsqueeze(0).to(device) # [0, 1]
            bicubic_tensor = hr_to_tensor(bicubic_image_pil).unsqueeze(0).to(device) # [0, 1]

            # --- Run Generator ---
            # Generator outputs [-1, 1] due to Tanh
            sr_tensor_tanh = generator(lr_tensor)
            
            # Un-normalize SR tensor from [-1, 1] to [0, 1] for metrics/saving
            sr_tensor = (sr_tensor_tanh + 1.0) / 2.0
            
            # --- Calculate Metrics (on Y-channel, borders cropped) ---
            sr_y, hr_y = preprocess_for_metrics(sr_tensor, hr_tensor, args.upscale_factor)
            
            current_psnr = calculate_psnr(sr_y, hr_y)
            current_ssim = ssim(sr_y, hr_y).item()
            
            total_psnr += current_psnr
            total_ssim += current_ssim

            # --- Save Output Images ---
            # This provides the visual comparison like in the paper
            base_name = hr_image_path.stem
            save_image(sr_tensor, output_path / f"{base_name}_SRGAN.png")
            save_image(bicubic_tensor, output_path / f"{base_name}_Bicubic.png")
            save_image(hr_tensor, output_path / f"{base_name}_Original.png")

    # --- 6. Report Averages ---
    avg_psnr = total_psnr / len(image_files)
    avg_ssim = total_ssim / len(image_files)
    
    print("\n--- Test Results ---")
    print(f"Dataset: {args.test_dir}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Generated images saved to '{args.output_dir}'")

if __name__ == "__main__":
    run_test()