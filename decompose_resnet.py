import torch
from torchvision import models
from ht2 import ht2
from hosvd1 import hosvd1
from svd_decomposition import svd_decomposition


if __name__ == "__main__":
    energy = 0.8
    net = models.resnet50(True)
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]

    decomposed = hosvd1(net.conv1, energy)
    net.conv1 = decomposed

    for layer in layers:
        for block in layer:
            conv1 = block.conv1
            decomposed_conv1 = hosvd1(conv1, energy)
            block.conv1 = decomposed_conv1
            conv2 = block.conv2
            decomposed_conv2 = ht2(conv2, energy)
            block.conv2 = decomposed_conv2
            conv3 = block.conv3
            decomposed_conv3 = hosvd1(conv3, energy)
            block.conv3 = decomposed_conv3
            if block.downsample:
                conv = block.downsample[0]
                decomposed_conv = hosvd1(conv, energy)
                block.downsample[0] = decomposed_conv

    decomposed = svd_decomposition(net.fc, energy)
    net.fc = decomposed

    print(net)
    checkpoint = {"model": net, "state_dict": net.state_dict()}
    torch.save(checkpoint, "resnet50_ht2.pth")
