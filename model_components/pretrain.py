import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from tqdm import tqdm
import math
import torch.nn.functional as F
import logging
from datetime import datetime
from typing import Tuple, Optional

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_components.generator import Generator
from model_components.dataset import SRGANDataset

def validate_psnr(
    generator: Generator, 
    dataloader: DataLoader, 
    device: torch.device
) -> float:
    
    generator.eval()
    total_psnr = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            gen_hr = generator(lr_imgs)
            
            # Un-normalize images from [-1, 1] to [0, 1] to calculate PSNR
            gen_hr = (gen_hr + 1.0) / 2.0
            hr_imgs = (hr_imgs + 1.0) / 2.0
            
            mse = F.mse_loss(gen_hr, hr_imgs)
            
            if mse.item() > 0:
                psnr = 10 * math.log10(1.0 / mse.item())
                total_psnr += psnr
            else:
                total_psnr += 100.0 # Perfect match
                
    generator.train()
    return total_psnr / len(dataloader)

def save_checkpoint(
    generator: Generator,
    optimizer_G: Optional[torch.optim.Optimizer] = None,
    save_path: str = "models/pretrain_checkpoint.pth",
    epoch: Optional[int] = None,
    mse_loss: Optional[float] = None,
    val_psnr: Optional[float] = None,
    best_val_psnr: Optional[float] = None
) -> None:
    """Save pretraining checkpoint."""
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
    }
    
    if optimizer_G is not None:
        checkpoint['optimizer_G_state_dict'] = optimizer_G.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if mse_loss is not None:
        checkpoint['mse_loss'] = mse_loss
    if val_psnr is not None:
        checkpoint['val_psnr'] = val_psnr
    if best_val_psnr is not None:
        checkpoint['best_val_psnr'] = best_val_psnr
    
    torch.save(checkpoint, save_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Checkpoint saved to {save_path}")

def load_checkpoint(
    generator: Generator,
    optimizer_G: Optional[torch.optim.Optimizer] = None,
    load_path: str = "models/pretrain_checkpoint.pth",
    device: torch.device = torch.device("cpu")
) -> Tuple[int, float]:
    """Load pretraining checkpoint."""
    logger = logging.getLogger(__name__)
    if not Path(load_path).exists():
        logger.info(f"Checkpoint not found at {load_path}. Starting from scratch.")
        return 0, 0.0
    
    checkpoint = torch.load(load_path, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    start_epoch = 0
    best_val_psnr = 0.0
    
    if optimizer_G is not None and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    if 'best_val_psnr' in checkpoint:
        best_val_psnr = checkpoint['best_val_psnr']
    
    logger.info(f"Loaded checkpoint from {load_path}. Resuming from epoch {start_epoch + 1}.")
    logger.info(f"Best validation PSNR from checkpoint: {best_val_psnr:.4f} dB")
    return start_epoch, best_val_psnr

def main_pretrain():
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pretrain_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    


    # Load Config
    with open('config.json', 'r') as f:
        config = json.load(f)

    res_blocks = config['num_res_blocks']
    logger.info(f"Using {res_blocks} residual blocks")

    generator = Generator(num_res_blocks=res_blocks).to(device)
    
    criterion = nn.MSELoss().to(device)
    
    DATA_DIR = config['train_data_path']
    VAL_SPLIT_SIZE = config['val_split_size']
    RANDOM_STATE = config['random_state']
    BATCH_SIZE = config['batch_size']
    NUM_WORKERS = config['num_workers']
    NUM_EPOCHS = config.get('pretrain_epochs', 100) 
    PRETRAIN_SAVE_PATH = "models/srresnet_mse.pth"
    LEARNING_RATE = config['learning_rate']
    RESUME_PRETRAIN = config.get('resume_pretrain', False)
    PRETRAIN_CHECKPOINT_PATH = config.get('pretrain_checkpoint_path', "models/pretrain_checkpoint.pth")
    # DataLoaders
    logger.info("Loading data for pre-training...")
    data_dir = Path(DATA_DIR)
    image_files = sorted(list(data_dir.glob("*.png")))
    logger.info(f"Found {len(image_files)} image files")
    train_files, val_files = train_test_split(
        image_files, test_size=VAL_SPLIT_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    # Load Checkpoint or Start Fresh
    if RESUME_PRETRAIN:
        logger.info(f"Resuming pretraining from checkpoint: {PRETRAIN_CHECKPOINT_PATH}")
        start_epoch, best_val_psnr = load_checkpoint(
            generator, optimizer_G, PRETRAIN_CHECKPOINT_PATH, device
        )
    else:
        start_epoch = 0
        best_val_psnr = 0.0
    
    train_dataset = SRGANDataset(train_files)
    val_dataset = SRGANDataset(val_files)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    logger.info(f"DataLoaders created successfully. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    logger.info("Starting SRResNet (MSE) pre-training...")
    logger.info(f"Training configuration: Epochs={NUM_EPOCHS}, Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")
    if RESUME_PRETRAIN:
        logger.info(f"Resuming from epoch {start_epoch + 1}. Best validation PSNR so far: {best_val_psnr:.4f} dB")
    
    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        generator.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)
        
        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer_G.zero_grad()
            
            gen_hr = generator(lr_imgs)
            
            # Calculate MSE loss
            loss = criterion(gen_hr, hr_imgs)
            
            loss.backward()
            optimizer_G.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'MSE_Loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        
        
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} complete.")
        logger.info(f"Average MSE Loss: {avg_loss:.4f}")
        
        val_psnr = validate_psnr(generator, val_loader, device)
        logger.info(f"Validation PSNR: {val_psnr:.4f} dB") # Commented out for faster training
        
        # Save Best Model (based on PSNR)
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            logger.info(f"New best model found! Saving to '{PRETRAIN_SAVE_PATH}' (PSNR: {val_psnr:.4f} dB)")
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(generator.state_dict(), PRETRAIN_SAVE_PATH)
        
        # Save checkpoint every 5 epochs
        if epoch % (NUM_EPOCHS//10) == 0:
            logger.info(f"Saving checkpoint for epoch {epoch}...")
            save_checkpoint(
                generator=generator,
                optimizer_G=optimizer_G,
                save_path=f"models/pretrain_checkpoint_epoch_{epoch}.pth",
                epoch=epoch,
                mse_loss=avg_loss,
                val_psnr=val_psnr,
                best_val_psnr=best_val_psnr
            )
            
    logger.info(f"Pre-training finished. Best model saved to '{PRETRAIN_SAVE_PATH}' with {best_val_psnr:.4f} dB.")

if __name__ == "__main__":
    main_pretrain()