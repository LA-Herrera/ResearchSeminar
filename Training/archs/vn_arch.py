import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, num_channels: int = 64) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=1)
        self.deconv = nn.Conv2d(num_channels * 2, num_channels * 2, kernel_size=3, padding=1, groups=num_channels * 2)
        self.simple_gate = SimpleGate()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_channels, num_channels, kernel_size=1))
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

        self.layer_norm2 = nn.LayerNorm(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels * 2, kernel_size=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros((1, num_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, num_channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer = x.permute(0, 2, 3, 1)
        layer_norm = self.layer_norm1(layer)
        layer_norm = layer_norm.permute(0, 3, 1, 2)
        x_conv1 = self.conv1(layer_norm)
        x_deconv = self.deconv(x_conv1)
        simple_gate = self.simple_gate(x_deconv)
        sca = simple_gate * self.attention(simple_gate)
        x_conv2 = self.conv2(sca)

        x_block = x + x_conv2 * self.beta

        layer = x_block.permute(0, 2, 3, 1)
        layer_norm = self.layer_norm2(layer)
        layer_norm = layer_norm.permute(0, 3, 1, 2)
        x_conv1 = self.conv3(layer_norm)
        simple_gate = self.simple_gate(x_conv1)
        x_conv2 = self.conv4(simple_gate)

        x = x + x_conv2 * self.gamma

        return x
    
@ARCH_REGISTRY.register()
class VN(nn.Module):
    def __init__(self, depth: int = 4, scale_factor: int = 4, num_channels: int = 64) -> None:
        super(VN, self).__init__()
        self.scale_factor = scale_factor

        self.first_layer = nn.Conv2d(9, num_channels, kernel_size=3, padding=1)
        self.nafblocks = nn.Sequential(*[NAFBlock(num_channels) for _ in range(depth)])
        self.last_layer = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x_bc = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')
        x_bl = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x_nn = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

        x_concat = torch.cat((x_nn, x_bl, x_bc), dim=1)

        x = self.first_layer(x_concat)
        x = self.nafblocks(x)
        x = self.last_layer(x)

        x = x + x_bc

        return x
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.normal_(self.last_layer.weight, mean=0.0, std=1e-3)
    
    def set_scale_factor(self, scale_factor: int = 4) -> None:
        self.scale_factor = scale_factor
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)