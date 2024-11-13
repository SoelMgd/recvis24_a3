import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)




class CustomEfficientNetB3(nn.Module):
    def __init__(self, num_classes=500, fine_tune=True):
        super(CustomEfficientNetB3, self).__init__()
        # Charger le modèle EfficientNet-B3 pré-entraîné
        self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Fine-tuning : geler toutes les couches sauf les dernières
        if fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.features[-1].parameters():
                param.requires_grad = True
        
        # Remplacer la couche de sortie pour 500 classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)



class CustomVGG16(nn.Module):
    def __init__(self, num_classes=500, fine_tune=True):
        super(CustomVGG16, self).__init__()
        # Charger le modèle VGG-16 pré-entraîné
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Fine-tuning : geler toutes les couches sauf la dernière couche
        if fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True

        # Remplacer la dernière couche pour 500 classes
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
