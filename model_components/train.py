import torch
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import math
import torch.nn.functional as F
import logging
from datetime import datetime
import sys

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_components.generator import Generator
from model_components.discriminator import Discriminator
from model_components.loss import SRGANLoss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model_components.dataset import SRGANDataset
import json

def train_step(
    generator: Generator,
    discriminator: Discriminator,
    loss_fn: SRGANLoss,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    lr_imgs: torch.Tensor,
    hr_imgs: torch.Tensor
) -> Tuple[float, float]:
    
    # Train Generator
    optimizer_G.zero_grad()
    
    # Generate SR Images
    gen_hr = generator(lr_imgs)
    
    # Discriminator Classification
    validity_fake = discriminator(gen_hr)
    
    # Calculate Perceptual Loss
    g_loss = loss_fn(validity_fake, gen_hr, hr_imgs)
    
    g_loss.backward()
    optimizer_G.step()
    
    # Train Discriminator
    optimizer_D.zero_grad()
    
    # Ground Truth Loss
    validity_real = discriminator(hr_imgs)
    d_real_loss = loss_fn.bce_loss(validity_real, torch.ones_like(validity_real))
    
    # Generated Loss
    validity_fake_detached = discriminator(gen_hr.detach())
    d_fake_loss = loss_fn.bce_loss(validity_fake_detached, torch.zeros_like(validity_fake_detached))
    
    # Average Discriminator Loss
    d_loss = (d_real_loss + d_fake_loss) / 2
    
    d_loss.backward()
    optimizer_D.step()
    
    return g_loss.item(), d_loss.item()


def save_model(
    generator: Generator,
    discriminator: Discriminator,
    optimizer_G: Optional[torch.optim.Optimizer] = None,
    optimizer_D: Optional[torch.optim.Optimizer] = None,
    save_path: str = "models/srgan_checkpoint.pth",
    epoch: Optional[int] = None,
    g_loss: Optional[float] = None,
    d_loss: Optional[float] = None,
    val_content_loss: Optional[float] = None  
) -> None:
    
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }
    
    if optimizer_G is not None:
        checkpoint['optimizer_G_state_dict'] = optimizer_G.state_dict()
    if optimizer_D is not None:
        checkpoint['optimizer_D_state_dict'] = optimizer_D.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if g_loss is not None:
        checkpoint['g_loss'] = g_loss
    if d_loss is not None:
        checkpoint['d_loss'] = d_loss
    if val_content_loss is not None:                    
        checkpoint['best_val_content_loss'] = val_content_loss 
    
    
    torch.save(checkpoint, save_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Model saved to {save_path}")

def load_model(
    generator: Generator,
    discriminator: Discriminator,
    optimizer_G: Optional[torch.optim.Optimizer] = None,
    optimizer_D: Optional[torch.optim.Optimizer] = None,
    load_path: str = "models/srgan_checkpoint.pth",
    device: torch.device = torch.device("cpu")
) -> Tuple[int, float]:
    
    logger = logging.getLogger(__name__)
    if not Path(load_path).exists():
        logger.info(f"Checkpoint not found at {load_path}. Starting from scratch.")
        return 0, float('inf')

    checkpoint = torch.load(load_path, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    start_epoch = 0
    best_val_content_loss = float('inf') 

    if optimizer_G is not None and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    if optimizer_D is not None and 'optimizer_D_state_dict' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    if 'best_val_content_loss' in checkpoint:  
        best_val_content_loss = checkpoint['best_val_content_loss']  

    logger.info(f"Loaded checkpoint from {load_path}. Resuming from epoch {start_epoch + 1}.")
    return start_epoch, best_val_content_loss  


def validate(
    generator: Generator,
    dataloader: DataLoader,
    loss_fn: SRGANLoss,  
    device: torch.device
) -> Tuple[float, float]:
    generator.eval() 
    total_psnr = 0
    total_content_loss = 0  
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            gen_hr = generator(lr_imgs)
            
            # Content Loss Calculation
            content_loss = loss_fn.content_loss(gen_hr, hr_imgs)  
            total_content_loss += content_loss.item()             
            
            # PSNR Calculation
            # Un-normalize images from [-1, 1] to [0, 1] to calculate PSNR
            gen_hr_unnorm = (gen_hr + 1.0) / 2.0
            hr_imgs_unnorm = (hr_imgs + 1.0) / 2.0
            
            # Calculate MSE 
            mse = F.mse_loss(gen_hr_unnorm, hr_imgs_unnorm)
            
            # Calculate PSNR 
            if mse.item() > 0:
                psnr = 10 * math.log10(1.0 / mse.item())
                total_psnr += psnr
            else:
                total_psnr += 100.0 
                
    generator.train() 
    
    avg_psnr = total_psnr / len(dataloader)
    avg_content_loss = total_content_loss / len(dataloader)  
    
    return avg_psnr, avg_content_loss  


def main():
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
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
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = SRGANLoss().to(device)

    # Parameters
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    DATA_DIR = config['train_data_path']
    VAL_SPLIT_SIZE = config['val_split_size']
    RANDOM_STATE = config['random_state']
    BATCH_SIZE = config['batch_size']
    NUM_WORKERS = config['num_workers']
    NUM_EPOCHS = config['num_epochs']
    RESUME_TRAINING = config['resume_training']
    CHECKPOINT_PATH = config['checkpoint_path']
    BEST_MODEL_PATH = "models/best_model.pth"
    LEARNING_RATE = config['learning_rate']
    PRETRAINED_GENERATOR_PATH = Path("models/srresnet_mse.pth")
    
    # Load pretrained SRResNet generator if available and not resuming training
    if not RESUME_TRAINING and PRETRAINED_GENERATOR_PATH.exists():
        logger.info(f"Loading pretrained SRResNet generator from {PRETRAINED_GENERATOR_PATH}...")
        try:
            pretrained_state = torch.load(PRETRAINED_GENERATOR_PATH, map_location=device)
            generator.load_state_dict(pretrained_state)
            logger.info("Pretrained SRResNet generator loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load pretrained generator: {e}. Starting with random weights.")

    # Get and split image files
    logger.info("Scanning for image files...")
    data_dir = Path(DATA_DIR)
    image_files = sorted(list(data_dir.glob("*.png")))
    logger.info(f"Found {len(image_files)} total images.")

    # training and validation sets (80:20)
    train_files, val_files = train_test_split(
        image_files,
        test_size=VAL_SPLIT_SIZE,
        random_state=RANDOM_STATE
    )
    logger.info(f"Training images: {len(train_files)}, Validation images: {len(val_files)}")

    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    # Learning Rate Schedulers (MultiStepLR for staged decay)
    # Decay by 0.1 at specified milestones (e.g., [100, 150] means decay at epoch 100 and 150)
    LR_DECAY_MILESTONES = config.get('lr_decay_milestones', [100])
    LR_DECAY_GAMMA = config.get('lr_decay_gamma', 0.1)
    
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_G, 
        milestones=LR_DECAY_MILESTONES, 
        gamma=LR_DECAY_GAMMA
    )
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_D, 
        milestones=LR_DECAY_MILESTONES, 
        gamma=LR_DECAY_GAMMA
    )
    
    logger.info(f"Learning rate schedulers initialized: milestones={LR_DECAY_MILESTONES}, gamma={LR_DECAY_GAMMA}")

    # Datasets
    train_dataset = SRGANDataset(train_files)
    val_dataset = SRGANDataset(val_files)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # Validate one image at a time
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    logger.info(f"DataLoaders created successfully. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Load Checkpoint or Start Fresh
    if RESUME_TRAINING:
        logger.info(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        start_epoch, best_val_content_loss = load_model(  
            generator, discriminator, optimizer_G, optimizer_D, CHECKPOINT_PATH, device
        )
        # Step schedulers to correct epoch after loading
        for _ in range(start_epoch):
            scheduler_G.step()
            scheduler_D.step()
    else:
        start_epoch = 0
        best_val_content_loss = float('inf') 

    # Early Stopping Parameters
    EARLY_STOPPING_PATIENCE = config.get('early_stopping_patience', 10)
    EARLY_STOPPING_MIN_DELTA = config.get('early_stopping_min_delta', 0.0) 
    USE_EARLY_STOPPING = config.get('use_early_stopping', False) 
    
    epochs_without_improvement = 0

    # Main Training Loop
    logger.info(f"Starting training from epoch {start_epoch + 1}...")
    logger.info(f"Training configuration: Epochs={NUM_EPOCHS}, Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")
    if USE_EARLY_STOPPING:
        logger.info(f"Early stopping enabled: patience={EARLY_STOPPING_PATIENCE}, min_delta={EARLY_STOPPING_MIN_DELTA}")
    
    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)
        
        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            g_loss, d_loss = train_step(
                generator, discriminator, criterion, 
                optimizer_G, optimizer_D, 
                lr_imgs, hr_imgs
            )
            
            epoch_g_loss += g_loss
            epoch_d_loss += d_loss
            
            progress_bar.set_postfix({
                'G_Loss': f"{g_loss:.4f}", 
                'D_Loss': f"{d_loss:.4f}"
            })
        
        # End of Epoch
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        # Get current learning rates
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} complete.")
        logger.info(f"Average Generator Loss: {avg_g_loss:.4f}")
        logger.info(f"Average Discriminator Loss: {avg_d_loss:.4f}")
        logger.info(f"Learning Rates: G={current_lr_G:.6f}, D={current_lr_D:.6f}")
        
        # Validation
        logger.info("Running validation...")
        val_psnr, val_content_loss = validate(generator, val_loader, criterion, device)  
        logger.info(f"Validation PSNR: {val_psnr:.4f} dB")
        logger.info(f"Validation Content Loss: {val_content_loss:.4f}")  
        
        improvement = best_val_content_loss - val_content_loss  
        has_improved = improvement > EARLY_STOPPING_MIN_DELTA
        
        # Save Best Model
        if has_improved:
            best_val_content_loss = val_content_loss  
            epochs_without_improvement = 0
            logger.info(f"New best model found at epoch {epoch}! (Improvement: {improvement:.4f}) Saving to '{BEST_MODEL_PATH}'")
            save_model(
                generator=generator,
                discriminator=discriminator,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                save_path=BEST_MODEL_PATH,
                epoch=epoch,
                g_loss=avg_g_loss,
                d_loss=avg_d_loss,
                val_content_loss=best_val_content_loss  
            )
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s). Best Content Loss: {best_val_content_loss:.4f}")  
        
        # Save a checkpoint every 5 epochs
        if epoch % 5 == 0:
            logger.info(f"Saving checkpoint for epoch {epoch}...")
            save_model(
                generator=generator,
                discriminator=discriminator,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                save_path=f"models/checkpoint_epoch_{epoch}.pth",
                epoch=epoch,
                g_loss=avg_g_loss,
                d_loss=avg_d_loss,
                val_content_loss=val_content_loss 
            )
        
        # Step the learning rate schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        if USE_EARLY_STOPPING and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            logger.info(f"Best Validation Content Loss achieved: {best_val_content_loss:.4f} at epoch {epoch - epochs_without_improvement}")  
            break

    logger.info("Training finished.")
    logger.info(f"Best Validation Content Loss achieved: {best_val_content_loss:.4f} at '{BEST_MODEL_PATH}'")  

if __name__ == "__main__":
    main()