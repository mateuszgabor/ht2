import torch
from torch import nn
from torchvision import models
from ht2 import ht2
from hosvd1 import hosvd1
from svd_decomposition import svd_decomposition
import matlab.engine


if __name__ == "__main__":
    energy = 0.78
    net = models.vgg16_bn(True)
    eng = matlab.engine.start_matlab()

    for key, layer in net.features._modules.items():
        if key == "0":
            decomposed = hosvd1(layer, eng, energy)
            net.features._modules[key] = decomposed

        if isinstance(layer, nn.modules.conv.Conv2d) and key != "0":
            decomposed = ht2(layer, eng, energy)
            net.features._modules[key] = decomposed
    
    for key, layer in net.classifier._modules.items():
        if isinstance(layer, nn.modules.linear.Linear):
            decomposed = svd_decomposition(layer, energy)
            net.classifier._modules[key] = decomposed
    
    checkpoint = {"state_dict": net.state_dict()}
    torch.save(checkpoint, "vgg16_ht2.pth")
        
