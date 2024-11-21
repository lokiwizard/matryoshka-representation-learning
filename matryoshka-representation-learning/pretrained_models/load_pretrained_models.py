# load resnet18, resnet34, resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def load_resnet18(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    return model

def load_resnet34(pretrained=True):
    model = models.resnet34(pretrained=pretrained)
    return model

def load_resnet50(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    return model


def load_models(model_name, pretrained=True):

    assert model_name in ['resnet18', 'resnet34', 'resnet50'], 'model_name not found'
    if model_name == 'resnet18':
        model = load_resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = load_resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = load_resnet50(pretrained=pretrained)
    else:
        raise ValueError('model_name not found')
    return model