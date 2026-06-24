"""From-scratch Attention U-Net (Oktay et al. 2018) + Dice+BCE loss.

Built entirely from torch.nn primitives — no segmentation_models_pytorch. This is
the same architecture defined in ml/notebooks/phase0_colab.ipynb, kept here so the
local pipeline (train/evaluate/export) uses the hand-built model too.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(3x3 conv -> BatchNorm -> ReLU) x2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Additive attention gate: reweight the encoder skip `x` by an attention map
    computed from `x` and the decoder gating signal `g`."""
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.theta_x = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.phi_g = nn.Conv2d(g_ch, inter_ch, 1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)

    def forward(self, g, x):
        alpha = torch.sigmoid(self.psi(F.relu(self.theta_x(x) + self.phi_g(g))))
        return x * alpha


class UpBlock(nn.Module):
    """Upsample -> attention-gate the skip -> concat -> DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.att = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=out_ch // 2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    """U-Net hand-built from conv primitives, with an attention gate on every skip."""
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.e1, self.e2 = DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1])
        self.e3, self.e4 = DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c[3], c[4])
        self.u4, self.u3 = UpBlock(c[4], c[3], c[3]), UpBlock(c[3], c[2], c[2])
        self.u2, self.u1 = UpBlock(c[2], c[1], c[1]), UpBlock(c[1], c[0], c[0])
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        x = self.u4(b, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        return self.head(x)           # raw logits [B, out_ch, H, W]


def dice_bce_loss(logits, target, eps=1.0):
    """Dice + BCE, both hand-written."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = (p + target).sum((1, 2, 3)) + eps
    return bce + (1 - (num / den).mean())
