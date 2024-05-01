from torchvision import models
from torch import nn


class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet18 = models.resnet18(weights=weights)
        self.resnet18.fc = nn.Identity()

    def forward(self, x):
        return self.resnet18(x)


class ResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34, self).__init__()
        weights = None
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        self.resnet34 = models.resnet34(weights=weights)
        self.resnet34.fc = nn.Identity()

    def forward(self, x):
        return self.resnet34(x)


class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        weights = None
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.resnet50 = models.resnet50(weights=weights)
        self.resnet50.fc = nn.Identity()

    def forward(self, x):
        return self.resnet50(x)
