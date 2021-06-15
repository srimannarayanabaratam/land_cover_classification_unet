""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch

from .unet_parts import *

torch.autograd.set_detect_anomaly(True)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ## Half Breadth Versions
        self.inc = DoubleConv(n_channels, 64//2)
        self.down1 = Down(64//2, 128//2)
        self.down2 = Down(128//2, 256//2)
        self.down3 = Down(256//2, 512//2)
        factor = 2 if bilinear else 1
        self.down4 = Down(512//2, 1024//2 // factor)
        self.up1 = Up(1024//2, 512 //2// factor, bilinear)
        self.up2 = Up(512//2, 256//2 // factor, bilinear)
        self.up3 = Up(256//2, 128//2 // factor, bilinear)
        self.up4 = Up(128//2, 64//2, bilinear)
        self.outc = OutConv(64//2, n_classes)

        # self.inc = DoubleConv(n_channels, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 5122// factor, bilinear)
        # self.up2 = Up(512, 256// factor, bilinear)
        # self.up3 = Up(256, 128// factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # GPUtil.showUtilization()
        # print("Base")
        assert torch.isfinite(x).all()
        x1 = self.inc(x)
        assert torch.isfinite(x1).all()
        # GPUtil.showUtilization()
        # print("After x1")
        x2 = self.down1(x1)
        assert torch.isfinite(x2).all()
        # GPUtil.showUtilization()
        # print("After x2")
        x3 = self.down2(x2)
        assert torch.isfinite(x3).all()
        # GPUtil.showUtilization()
        # print("After x3")
        x4 = self.down3(x3)
        assert torch.isfinite(x4).all()
        # GPUtil.showUtilization()
        # print("After x4")
        x5 = self.down4(x4)
        assert torch.isfinite(x5).all()
        # GPUtil.showUtilization()
        # print("After x5")
        x = self.up1(x5, x4)
        assert torch.isfinite(x).all()
        # GPUtil.showUtilization()
        # print("After xup1")
        # del x5
        # del x4
        # torch.cuda.empty_cache()
        # GPUtil.showUtilization()
        # print("After x5,x4 delete")
        x = self.up2(x, x3)
        assert torch.isfinite(x).all()
        # GPUtil.showUtilization()
        # print("After xup2")
        # del x3
        # torch.cuda.empty_cache()
        # GPUtil.showUtilization()
        # print("After x3 delete")
        x = self.up3(x, x2)
        assert torch.isfinite(x).all()
        # GPUtil.showUtilization()
        # print("After xup3")
        # del x2
        # torch.cuda.empty_cache()
        # GPUtil.showUtilization()
        # print("After x2 delete")
        x = self.up4(x, x1)
        assert torch.isfinite(x).all()
        # GPUtil.showUtilization()
        # print("After xup4")
        # del x1
        # torch.cuda.empty_cache()
        # GPUtil.showUtilization()
        # print("After x1 delete")
        logits = self.outc(x)
        assert torch.isfinite(logits).all()
        return logits
