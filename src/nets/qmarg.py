import torch
import torch.nn as nn
import math
from nets import base

class Net(nn.Module):
    def __init__(self, in_channel, in_glo_channel, out_channel, n_colors=3, n_feats=64, n_resblocks=16, res_scale=0.1, scale=2):
        super(Net, self).__init__()

        gfc = 32
        self.gfc = gfc
        self.gfs = 7
        biasSet = True
        self.conv_global = nn.Sequential( #input: 112*112
            nn.Conv2d(in_glo_channel, gfc, kernel_size=4, stride=2, padding=1, bias=biasSet), #56
            nn.ReLU(True),
            nn.Conv2d(gfc, gfc, kernel_size=3, stride=1, padding=1, bias=biasSet),
            nn.ReLU(True),
            nn.Conv2d(gfc, gfc*2, kernel_size=4, stride=2, padding=1, bias=biasSet), #28
            nn.ReLU(True),
            nn.Conv2d(gfc*2, gfc*2, kernel_size=3, stride=1, padding=1, bias=biasSet),
            nn.ReLU(True),
            nn.Conv2d(gfc*2, gfc*4, kernel_size=4, stride=2, padding=1, bias=biasSet), #14
            nn.ReLU(True),
            nn.Conv2d(gfc*4, gfc*4, kernel_size=3, stride=1, padding=1, bias=biasSet),
            nn.ReLU(True),
            nn.Conv2d(gfc*4, gfc*4, kernel_size=4, stride=2, padding=1, bias=biasSet), #7
            nn.ReLU(True),
            nn.Conv2d(gfc*4, gfc*4, kernel_size=3, stride=1, padding=1, bias=biasSet),
            nn.ReLU(True),
        )
        self.conv_global_l = nn.Sequential(
            nn.Linear(self.gfs**2*gfc*4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64),
        )

        self.conv_input = nn.Conv2d(in_channel, n_feats, kernel_size=3, stride=1, padding=1, bias=False)

        self.downscale = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True)
        )
        
        residual1 = [
            base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks//2)
        ]
        self.residual1 = nn.Sequential(*residual1)

        self.conv_res_mid = nn.Conv2d(n_feats*2, n_feats, kernel_size=3, stride=1, padding=1)
        self.relu_mid = nn.ReLU(True)

        residual2 = [
            base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks//2)
        ]
        self.residual2 = nn.Sequential(*residual2)

        self.conv_mid = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale = nn.Sequential(
            nn.Conv2d(n_feats, (scale**2)*n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(scale),
        )

        self.conv_output = nn.Conv2d(n_feats, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, source):
        x, resized = source[0], source[1]
        assert resized.shape[2] == 112, 'Wrong input size {}, should be 112.'.format(resized.shape[2])
        assert resized.shape[3] == 112, 'Wrong input size {}, should be 112.'.format(resized.shape[3])
        gout = self.conv_global(resized)
        gout = gout.view(-1, self.gfs**2*self.gfc*4) #TODO
        gout = self.conv_global_l(gout)

        
        out = self.conv_input(x)
        out = self.downscale(out)
        
        skip1 = out
        out = self.residual1(out)
        out += skip1

        gout = gout.repeat(out.shape[2], out.shape[3], 1, 1)
        gout = gout.transpose(0,2).transpose(1,3)

        out = torch.cat((out, gout), 1)
        out = self.conv_res_mid(out)
        out = self.relu_mid(out)

        skip2 = out
        out = self.residual2(out)
        out += skip2

        out = self.conv_mid(out)
        out = self.upscale(out)
        out = self.conv_output(out)
        return out
