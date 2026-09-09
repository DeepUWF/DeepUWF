import torch,os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(ch, ch),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class ResNetFPNRecover(nn.Module):
    """
    KPA attacker used during adversarial training.

    Input:
        encrypted image / mix_img, shape [B, 3, H, W], range [0, 1]
    Output:
        recovered plaintext image, shape [B, 3, H, W], range [0, 1]

    This architecture is intentionally different from the U-Net used for final KPA evaluation.
    """

    def __init__(self, out_channels=3, pretrained=True, base_channels=128):
        super().__init__()

        try:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
        except Exception:
            backbone = models.resnet34(pretrained=pretrained)
        #resnet34_weight_path = "/path/to/hpc/workspace/torch-cache/hub/checkpoints/resnet34-b627a593.pth"
        #backbone = models.resnet34(weights=None)
        # if not pretrained:
        #     if not os.path.isfile(resnet34_weight_path):
        #         raise FileNotFoundError(

        #             f"ResNet34 pretrained weight not found: {resnet34_weight_path}"

        #         )
        #     state_dict = torch.load(resnet34_weight_path, map_location="cpu")
        #     msg = backbone.load_state_dict(state_dict, strict=True)
        #     print(f"Loaded local ResNet34 weights from {resnet34_weight_path}")
        #     print(f"Load message: {msg}")

        # ResNet stem and stages
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )  # 1/2

        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32

        # lateral projections
        self.lat1 = nn.Conv2d(64, base_channels, kernel_size=1)
        self.lat2 = nn.Conv2d(128, base_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(256, base_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(512, base_channels, kernel_size=1)

        # smooth blocks
        self.smooth4 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.smooth3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.smooth2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.smooth1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

        # decoder
        self.dec3 = ConvBNReLU(base_channels, base_channels)
        self.dec2 = ConvBNReLU(base_channels, base_channels)
        self.dec1 = ConvBNReLU(base_channels, base_channels)
        self.dec0 = ConvBNReLU(base_channels, base_channels // 2)

        self.out = nn.Sequential(
            ConvBNReLU(base_channels // 2, base_channels // 2),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        # encoder
        c0 = self.stem(x)              # 1/2
        c1 = self.layer1(self.maxpool(c0))  # 1/4
        c2 = self.layer2(c1)           # 1/8
        c3 = self.layer3(c2)           # 1/16
        c4 = self.layer4(c3)           # 1/32

        # FPN top-down
        p4 = self.smooth4(self.lat4(c4))

        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)

        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)

        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.smooth1(p1)

        # decode to original size
        x = self.dec3(F.interpolate(p1, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.dec2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.dec1(x)
        x = self.dec0(x)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.out(x)

        return x