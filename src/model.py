import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, efficientnet_b4
import torch.nn.functional as F


class AGClassifierResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
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
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
       
    def forward(self, x):
        return self.backbone(x)
    
class AGClassifierEfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = efficientnet_b4(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)





# DenseNet + Attention (Squeeze-and-Excitation Block)
 
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class AGClassifierDenseNetSE(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # 1) Backbone DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        # 2) adding a SEBlock before classification
        self.seblock = SEBlock(self.backbone.features[-1].num_features)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone.features(x) 
        features = self.seblock(features)     # apply Attention
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1,1)).view(features.size(0), -1)
        logits = self.backbone.classifier(out)
        return logits
