import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, efficientnet_b4

class AGClassifierResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # 1) Load pretrained ResNet18 (or set pretrained=False to train from scratch)
        self.backbone = models.resnet18(pretrained=pretrained)
        # 2) Replace the final fully‐connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class AGClassifierResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class AGClassifierDenseNet121(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.densenet121(pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class AGClassifierEfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=pretrained)
       
