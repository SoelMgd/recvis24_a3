"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms
from model import CustomEfficientNetB3, CustomVGG16 


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "efficientnet_b3":
            print("Loading EfficientNet-B3 model")
            return CustomEfficientNetB3(num_classes=500, fine_tune=False)
        if self.model_name == "vgg16":
            print("Loading vgg16 model")
            return CustomVGG16(num_classes=500, fine_tune=False)
        
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "efficientnet_b3":
            return data_transforms
        if self.model_name == "vgg16":
            return data_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform