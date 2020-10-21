import torch
import torch.nn as nn
import math
from nets import base

class Net(nn.Module):
    def __init__(self, n_colors=3, n_feats=64, n_resblocks=16, res_scale=0.1, scale=2):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(n_colors, n_feats, kernel_size=3, stride=1, padding=1, bias=False)

        self.downscale = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True)
        )
        
        residual = [
            base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        self.residual = nn.Sequential(*residual)

        self.conv_mid = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale = nn.Sequential(
            nn.Conv2d(n_feats, (scale**2)*n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(scale),
        )

        self.conv_output = nn.Conv2d(n_feats, n_colors, kernel_size=3, stride=1, padding=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_input(x)
        skip = out
        out = self.downscale(out)
        out = self.residual(out)
        out = self.conv_mid(out)
        out = self.upscale(out)
        out += skip
        out = self.conv_output(out)
        return out
