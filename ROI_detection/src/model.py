# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

import torch
import torch.nn as nn

def get_densenet_model(model_name):
    if model_name == 'densenet121':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        in_features = 1024
    elif model_name == 'densenet169':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet169', pretrained=True)
        in_features = 1664
    elif model_name == 'densenet201':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=True)
        in_features = 1920
    elif model_name == 'densenet161':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet161', pretrained=True)
        in_features = 2208

    model.classifier = nn.Linear(in_features=in_features, out_features=2, bias=True)
    return model
