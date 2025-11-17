#!/bin/bash

# Script to run complete SRGAN training pipeline:
# 1. Pretrain at lr=0.0001 for 100 epochs
# 2. Train at lr=0.0001 for 10 epochs
# 3. Train at lr=0.00001 for 10 epochs

set -e  # Exit on error

echo "=========================================="
echo "SRGAN Training Pipeline"
echo "=========================================="

# Backup original config
echo "Backing up original config.json..."
cp config.json config.json.backup

# Function to update config using Python
update_config() {
    python3 << EOF
import json

with open('config.json', 'r') as f:
    config = json.load(f)

$1

with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)
EOF
}

# Step 1: Pretrain at lr=0.0001 for 50 epochs
echo ""
echo "=========================================="
echo "Step 1: Pretraining (lr=0.0001, 100 epochs)"
echo "=========================================="
update_config "config['pretrain_epochs'] = 100"
echo "Updated config: pretrain_epochs=100"
echo "Running pretrain..."
python3 -m model_components.pretrain

# Load pretrained generator into a checkpoint for training
echo ""
echo "Loading pretrained generator for training..."
python3 << 'LOADSCRIPT'
import torch
from pathlib import Path
from model_components.generator import Generator
from model_components.discriminator import Discriminator

pretrain_path = Path("models/srresnet_mse.pth")
checkpoint_path = Path("models/srgan_checkpoint.pth")

if pretrain_path.exists():
    # Load pretrained generator
    pretrain_state = torch.load(pretrain_path, map_location='cpu')
    
    # Create fresh discriminator to get its initial state
    discriminator = Discriminator()
    
    # Create checkpoint with pretrained generator and fresh discriminator
    checkpoint = {
        'generator_state_dict': pretrain_state,
        'discriminator_state_dict': discriminator.state_dict(),
        'epoch': 0,
        'best_val_content_loss': float('inf')
    }
    
    # Save as checkpoint
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Pretrained generator loaded into {checkpoint_path}")
    print("Fresh discriminator initialized")
else:
    print(f"Warning: Pretrained model not found at {pretrain_path}")
LOADSCRIPT

# Step 2: Train at lr=0.0001 for 10 epochs
echo ""
echo "=========================================="
echo "Step 2: Training Phase 1 (lr=0.0001, 10 epochs)"
echo "=========================================="
update_config "config['learning_rate'] = 0.0001
config['num_epochs'] = 10
config['resume_training'] = True
config['checkpoint_path'] = 'models/srgan_checkpoint.pth'"
echo "Updated config: learning_rate=0.0001, num_epochs=10, resume_training=True"
echo "Running train..."
python3 -m model_components.train

# Update checkpoint to best model for next phase
if [ -f "models/best_model.pth" ]; then
    echo "Copying best model to checkpoint for next phase..."
    cp models/best_model.pth models/srgan_checkpoint.pth
fi

# Step 3: Train at lr=0.00001 for 10 epochs
echo ""
echo "=========================================="
echo "Step 3: Training Phase 2 (lr=0.00001, 10 epochs)"
echo "=========================================="
update_config "config['learning_rate'] = 0.00001
config['num_epochs'] = 10
config['resume_training'] = True
config['checkpoint_path'] = 'models/srgan_checkpoint.pth'"
echo "Updated config: learning_rate=0.00001, num_epochs=10, resume_training=True"
echo "Running train..."
python3 -m model_components.train

# Restore original config
echo ""
echo "Restoring original config.json..."
mv config.json.backup config.json

echo ""
echo "=========================================="
echo "Training Pipeline Complete!"
echo "=========================================="
echo "Best model saved at: models/best_model.pth"
echo "Checkpoints saved in: models/"

