import torch
from torch import nn
from torchvision import models
from ht2 import ht2_compression
import matlab


if __name__ == "__main__":
    energy = 0.78
    net = models.vgg16_bn(True)
    eng = matlab.engine.start_matlab()

    for key, layer in net.features._modules.items():
        if key == "0":
            pass

        if isinstance(layer, nn.modules.conv.Conv2d) and key != "0":
            compressed = ht2_compression(layer, eng, energy)
            net.features._modules[key] = compressed

    checkpoint = {"state_dict": net.state_dict()}
    torch.save(checkpoint, "vgg16_ht2")
