#loads the pre-trained model (like r3d_18), modifies last layer

import torch
import torch.nn as nn
import torchvision

def build_model_r3d_18():
    #load teh pre-trained 3D ResNet-18 model
    model = torchvision.models.video.r3d_18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False # freeze all layers

    in_features = model.fc.in_features # usually 512
    model.fc = nn.Linear(in_features, 2) # 2 output class for crash or no_crash (changing the final classfication layer)
    return model

def build_model_mc3():
    model = torchvision.models.video.mc3_18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model