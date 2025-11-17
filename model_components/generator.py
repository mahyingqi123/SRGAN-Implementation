import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Conv -> BN -> PReLU -> Conv -> BN -> Element-wise Sum
    """
    def __init__(self, channels: int = 64) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False) # k3n64s1
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False) # k3n64s1
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual # Element-wise sum with skip connection

class UpsampleBlock(nn.Module):
    """
    Upsampling using Sub-Pixel Convolution (PixelShuffle) to increase resolution by 2x.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1) # k3n256s1
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    """
    SRResNet Architecture:
    - Initial conv with PReLU activation
    - 16 Residual Blocks
    - Middle conv, BN and Element-wise sum with output of first PReLU layer
    - 2 Upsampling Blocks for 4x upscaling 
    - Final conv with Tanh activation
    """
    def __init__(self, num_res_blocks: int = 16) -> None:
        super(Generator, self).__init__()
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4) # k9n64s1
        self.prelu = nn.PReLU()
        
        # 16 Residual Blocks (B=16), configurable in config.json
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Middle Convolution 
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False) # k3n64s1
        self.bn_mid = nn.BatchNorm2d(64)
        
        # Upsampling Blocks
        self.upsample1 = UpsampleBlock(64, 64)
        self.upsample2 = UpsampleBlock(64, 64)
        
        # Output Convolution
        self.conv_out = nn.Conv2d(64, 3, kernel_size=9, padding=4) # k9n3s1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.prelu(self.conv1(x))
        
        out_res = self.res_blocks(out1)
        out_res = self.bn_mid(self.conv_mid(out_res))
        out_res = out1 + out_res  # Element-wise Sum with residual connection
        
        out = self.upsample1(out_res)
        out = self.upsample2(out)
        out = self.conv_out(out)
        return torch.tanh(out) # Use Tanh to scale output to [-1, 1]