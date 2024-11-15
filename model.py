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
    

class CustomEfficientNetB4(nn.Module):
    def __init__(self, num_classes=500, fine_tune=True):
        super(CustomEfficientNetB4, self).__init__()
        # Charger le modèle EfficientNet-B4 pré-entraîné
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

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


class CustomEfficientNetM(nn.Module):
    def __init__(self, num_classes=500, model_name="efficientnet_v2_m"):
        super(CustomEfficientNetM, self).__init__()
        if model_name == "efficientnet_v2_m":
            self.model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Modèle non pris en charge")

        # Remplacer la couche de sortie pour 500 classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class CustomResNet50(nn.Module):
    def __init__(self, num_classes=500):
        super(CustomResNet50, self).__init__()
        
        # Charger le modèle ResNet-50 pré-entraîné
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remplacer la dernière couche fully connected pour adapter le modèle aux 500 classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def get_features(self):
        # Retourne les couches convolutionnelles sans la dernière couche fully connected
        return nn.Sequential(*list(self.model.children())[:-2], nn.AdaptiveAvgPool2d((1, 1)))


class EnsembleModel(nn.Module):
    def __init__(self, num_classes=500):
        super(EnsembleModel, self).__init__()

        # Charger EfficientNet-B3
        self.model1 = CustomEfficientNetB3(num_classes=500, fine_tune=False)
        self.model1.load_state_dict(torch.load("/kaggle/working/recvis24_a3/saved_models/model_unknown.pth"))
        self.model1 = self.model1.model.features  # Accéder aux caractéristiques convolutionnelles directement


        # Charger ResNet-50
        self.model2 = CustomResNet50(num_classes=500)
        self.model2.load_state_dict(torch.load("/kaggle/working/recvis24_a3/saved_models/ResNet50.pth"))
        self.model2 = self.model2.get_features()


        # Geler les paramètres des modèles de base
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False

        # Créer la nouvelle couche fully connected pour la sortie combinée
        self.fc = nn.Linear(1280 + 2048, num_classes)  # Adapter les dimensions en fonction des modèles

    def forward(self, x):
        # Extraire les caractéristiques de chaque modèle de base
        feat1 = self.model1(x)
        feat2 = self.model2(x)

        # Réduire les dimensions avec Global Average Pooling pour EfficientNet-B3
        feat1 = feat1.mean([2, 3])  # Réduit [16, 1536, 7, 7] à [16, 1536]

        # Aplatir les caractéristiques de ResNet-50
        feat2 = feat2.view(feat2.size(0), -1)  # Réduit [16, 2048, 1, 1] à [16, 2048]

        # Vérifier les dimensions
        print("EfficientNet-B3 features shape:", feat1.shape)  # Devrait être [16, 1536]
        print("ResNet-50 features shape:", feat2.shape)        # Devrait être [16, 2048]

        # Concaténer les caractéristiques
        combined_features = torch.cat([feat1, feat2], dim=1)
        print("Combined features shape:", combined_features.shape)  # Devrait être [16, 3584]

        # Passer par la dernière couche pour la classification finale
        out = self.fc(combined_features)
        return out
