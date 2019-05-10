from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from model.densenet import DenseNet121,DenseNet169,DenseNet201,DenseNet161
from model.resnext import ResNeXt29_2x64d,ResNeXt29_32x4d,ResNeXt29_4x64d,ResNeXt29_8x64d
from model.vgg import VGG

def qmodel(name, device):
    if name == "resnet18":
        return ResNet18().to(device)
    elif name == "resnet34":
        return ResNet34().to(device)
    elif name == "resnet50":
        return ResNet50().to(device)
    elif name == "resnet101":
        return ResNet101().to(device)
    elif name == "resnet152":
        return ResNet152().to(device)
    elif name == "vgg11":
        return VGG("VGG11").to(device)
    elif name == "vgg13":
        return VGG("VGG13").to(device)
    elif name == "vgg16":
        return VGG("VGG16").to(device)
    elif name == "vgg19":
        return VGG("VGG19").to(device)
    elif name == "densenet121":
        return DenseNet121().to(device)
    elif name == "densenet169":
        return DenseNet169().to(device)
    elif name == "densenet201":
        return DenseNet201().to(device)
    elif name == "resnext":
        return ResNeXt29_8x64d().to(device)
