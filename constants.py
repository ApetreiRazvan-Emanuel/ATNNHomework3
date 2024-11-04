from torchvision import datasets, transforms
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import timm
import models


dataset_classes = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100
}

normalization_values = {
    "MNIST": ((0.1307,), (0.3081,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
}

transform_methods = {
    "Resize": transforms.Resize,
    "RandomCrop": transforms.RandomCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "RandomVerticalFlip": transforms.RandomVerticalFlip,
    "RandomAffine": transforms.RandomAffine,
    "RandomRotation": transforms.RandomRotation,
    "RandomErasing": transforms.RandomErasing,
    "ColorJitter": transforms.ColorJitter,
    "GaussianBlur": transforms.GaussianBlur
}

optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "RMSprop": optim.RMSprop
}

schedulers = {
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "StepLR": lr_scheduler.StepLR
}

loss_functions = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "L1Loss": nn.L1Loss,
    "NLLLoss": nn.NLLLoss
}

available_models = {
    "resnet18_cifar10": lambda: timm.create_model('resnet18', pretrained=False, num_classes=10),
    "resnet18_cifar100": lambda: timm.create_model('resnet18', pretrained=False, num_classes=100),
    "PreActResNet18-CIFAR10": lambda: models.PreActResNet18_C10(10),
    "PreActResNet18-CIFAR100": lambda: models.PreActResNet18_C10(100),
    "mlp": lambda: models.MLP(),
    "LeNet": lambda: models.LeNet()
}
