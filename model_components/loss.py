import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import json

class SRGANLoss(nn.Module):
    def __init__(self, use_vgg54: bool = True) -> None:
        super(SRGANLoss, self).__init__()

        with open('config.json', 'r') as f:
            config = json.load(f)
            
        self.use_pixel_loss = config['use_pixel_loss']
        
        vgg = models.vgg19(pretrained=True)
        
        if use_vgg54:
            self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:36])
        else:
            self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:9])
            
        self.feature_extractor.eval() 
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss() 

    def pixel_loss(self, sr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
        """
        This pixel loss is used to avoid the collapse of the generator.
        As during training without this loss, the generator will learn to output unrecognizable images.
        It could be the sheer number of epochs needed to train the generator to output realistic images.
        The original paper ran for 1million iterations for pretraining, and 200k iterations for training.
        But with only 300-500 epochs, this loss calculation is better.
        Arguably there's little difference between using the perceptual loss and mse loss after
        adding this pixel loss. But the pixel loss is more stable and faster to train.
        """
        if self.use_pixel_loss:
            return self.mse_loss(sr_img, hr_img)
        else:
            return 0.0

    def content_loss(self, sr_img: torch.Tensor, hr_img: torch.Tensor) -> torch.Tensor:
        # VGG expects images in [0, 1] range, convert to [0, 1]
        sr_img_normalized = (sr_img + 1.0) / 2.0
        hr_img_normalized = (hr_img + 1.0) / 2.0
        
        # Clip to ensure values are in [0, 1] range
        sr_img_normalized = torch.clamp(sr_img_normalized, 0, 1)
        hr_img_normalized = torch.clamp(hr_img_normalized, 0, 1)
        
        # Extract features and compute MSE loss
        sr_features = self.feature_extractor(sr_img_normalized)
        hr_features = self.feature_extractor(hr_img_normalized)
        
        # Divide by 12.75 to normalize (as in the paper)
        return self.mse_loss(sr_features / 12.75, hr_features / 12.75)

    def adversarial_loss(self, discriminator_preds: torch.Tensor) -> torch.Tensor:

        return self.bce_loss(discriminator_preds, torch.ones_like(discriminator_preds))

    def forward(
        self,
        discriminator_preds: torch.Tensor,
        sr_img: torch.Tensor,
        hr_img: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the final hybrid generator loss.
        
        This combines Pixel Loss (for stability), Content Loss (for perception),
        and Adversarial Loss (for realism), based on the robust 
        implementation from the Lornatang/SRGAN-PyTorch repo.
        """
        
        # Weights based on Lornatang's config 
        pixel_weight = 1.0
        content_weight = 1.0
        adversarial_weight = 1e-3 
        
        pixel = self.pixel_loss(sr_img, hr_img)
        
        content = self.content_loss(sr_img, hr_img)
        
        adversarial = self.adversarial_loss(discriminator_preds)
        
        # Combine all three
        total_loss = (pixel_weight * pixel) + (content_weight * content) + (adversarial_weight * adversarial)
        
        return total_loss