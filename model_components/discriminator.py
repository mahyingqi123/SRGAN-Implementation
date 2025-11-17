import torch
import torch.nn as nn
from typing import List

class Discriminator(nn.Module):
    """
    Discriminator Architecture:
    - VGG-style blocks (increasing filters: 64->512)
    - Alternating strided convolutions (stride=2) for downsampling
    - LeakyReLU (alpha=0.2)
    - Dense layers at the end
    """
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        
        def discriminator_block(
            in_filters: int,
            out_filters: int,
            stride: int = 1,
            batch_norm: bool = True
        ) -> List[nn.Module]:
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Input size: 3x96x96
        # Output size: 512x6x6
        self.model = nn.Sequential(
            *discriminator_block(3, 64, stride=1, batch_norm=False),   # k3n64s1
            *discriminator_block(64, 64, stride=2),                    # k3n64s2
            *discriminator_block(64, 128, stride=1),                   # k3n128s1
            *discriminator_block(128, 128, stride=2),                  # k3n128s2
            *discriminator_block(128, 256, stride=1),                  # k3n256s1
            *discriminator_block(256, 256, stride=2),                  # k3n256s2
            *discriminator_block(256, 512, stride=1),                  # k3n512s1
            *discriminator_block(512, 512, stride=2),                  # k3n512s2
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18432, 1024), # 512x6x6 = 18432
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.model(img)
        validity = self.classifier(features)
        return validity

