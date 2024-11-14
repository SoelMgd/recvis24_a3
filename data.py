import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance


class ThickenLines:
    """Classe de transformation pour épaissir les lignes d'un croquis."""
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        # Convertir l'image en niveaux de gris pour appliquer la dilatation
        img = np.array(img.convert("L"))
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        # Reconvertir en RGB après la dilatation pour maintenir la cohérence des canaux
        img = np.stack([img]*3, axis=-1)  # Convertir de 1 canal à 3 canaux
        return Image.fromarray(img, mode="RGB")


class InvertColor:
    def __call__(self, img):
        return ImageOps.invert(img.convert("RGB"))

class LightenImage:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img.convert("RGB"))
        return enhancer.enhance(self.factor)


# Définition des transformations pour l'entraînement avec data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# Définition des transformations pour l'entraînement avec data augmentation
train_transforms_old = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomApply([InvertColor()], p=0.33),  # Inversion des couleurs avec probabilité de 33%
    transforms.RandomApply([ThickenLines(kernel_size=3)], p=0.4),  # Épaississement avec probabilité de 50%
    transforms.RandomApply([LightenImage(factor=1.5)], p=0.4),  # Éclaircissement avec probabilité de 50%
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])




test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
