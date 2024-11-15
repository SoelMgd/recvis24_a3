"""Python file to instantite the model and the transform that goes with it."""

from data import train_transforms, test_transforms
from model import CustomEfficientNetB3, CustomVGG16, CustomEfficientNetM, CustomResNet50, CustomEfficientNetB4


class ModelFactory:
    def __init__(self, model_name: str, train_mode=False):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform(train_mode)

    def init_model(self):
        if self.model_name == "efficientnet_b3":
            print("Loading EfficientNet-B3 model")
            return CustomEfficientNetB3(num_classes=500, fine_tune=False)
        if self.model_name == "efficientnet_b4":
            print("Loading EfficientNet-B4 model")
            return CustomEfficientNetB4(num_classes=500, fine_tune=False)
        if self.model_name == "vgg16":
            print("Loading vgg16 model")
            return CustomVGG16(num_classes=500, fine_tune=False)
        if self.model_name == "efficientnet_v2_m":
            print("Loading efficientnet_v2_m model")
            return CustomEfficientNetM(num_classes=500)
        if self.model_name == "resnet50":
            print("Loading resnet50 model")
            return CustomResNet50(num_classes=500)
        
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self, train_mode=False):
        if train_mode:
            return train_transforms
        if not train_mode:
            return test_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform