import torch.nn as nn
from torchvision.models.googlenet import BasicConv2d

from decompositions import conv_svd, fc_svd, ht2


def _compress_conv_layer(layer, energy, is_first_layer=False):
    if is_first_layer or layer.kernel_size == (1, 1):
        return conv_svd(layer, energy)
    return ht2(layer, energy)


def compress_resnet18(args, net):
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]
    net.conv1 = _compress_conv_layer(net.conv1, args.e, is_first_layer=True)

    for layer in layers:
        for block in layer:
            block.conv1 = _compress_conv_layer(block.conv1, args.e)
            block.conv2 = _compress_conv_layer(block.conv2, args.e)
            if block.downsample:
                block.downsample[0] = _compress_conv_layer(
                    block.downsample[0], args.e, is_first_layer=True
                )

    net.fc = fc_svd(net.fc, args.e)


def compress_resnet50(args, net):
    layers = [net.layer1, net.layer2, net.layer3, net.layer4]
    net.conv1 = _compress_conv_layer(net.conv1, args.e, is_first_layer=True)

    for layer in layers:
        for block in layer:
            block.conv1 = _compress_conv_layer(block.conv1, args.e, is_first_layer=True)
            block.conv2 = _compress_conv_layer(block.conv2, args.e)
            block.conv3 = _compress_conv_layer(block.conv3, args.e, is_first_layer=True)
            if block.downsample:
                block.downsample[0] = _compress_conv_layer(
                    block.downsample[0], args.e, is_first_layer=True
                )

    net.fc = fc_svd(net.fc, args.e)


def compress_resnet(args, net):
    match args.net:
        case "imagenet_resnet18":
            compress_resnet18(args, net)
        case "imagenet_resnet50":
            compress_resnet50(args, net)
        case _:
            raise NotImplementedError(f"Network {args.net} currently is not supported")


def compress_densenet(args, net):
    denses = [net.dense1, net.dense2, net.dense3]
    transitions = [net.trans1, net.trans2]

    net.conv1 = _compress_conv_layer(net.conv1, args.e, is_first_layer=True)

    for dense in denses:
        for layer in dense:
            layer.conv1 = _compress_conv_layer(layer.conv1, args.e)

    for trans in transitions:
        trans.conv1 = _compress_conv_layer(trans.conv1, args.e, is_first_layer=True)

    net.fc = fc_svd(net.fc, args.e)


def _compress_sequential_model(net, energy):
    for key, layer in net.features._modules.items():
        if isinstance(layer, nn.modules.conv.Conv2d):
            net.features._modules[key] = _compress_conv_layer(
                layer, energy, is_first_layer=(key == "0")
            )

    for key, layer in net.classifier._modules.items():
        if isinstance(layer, nn.modules.linear.Linear):
            net.classifier._modules[key] = fc_svd(layer, energy)


def compress_vgg(args, net):
    _compress_sequential_model(net, args.e)


def compress_alexnet(args, net):
    _compress_sequential_model(net, args.e)


def _compress_googlenet_branch(branch, energy):
    if isinstance(branch, nn.Sequential):
        for layer in branch:
            if isinstance(layer, BasicConv2d):
                layer.conv = _compress_conv_layer(
                    layer.conv,
                    energy,
                    is_first_layer=(layer.conv.kernel_size == (1, 1)),
                )
    elif isinstance(branch, BasicConv2d):
        branch.conv = _compress_conv_layer(
            branch.conv, energy, is_first_layer=(branch.conv.kernel_size == (1, 1))
        )


def compress_googlenet(args, net):
    inceptions = [
        net.inception3a,
        net.inception3b,
        net.inception4a,
        net.inception4b,
        net.inception4c,
        net.inception4d,
        net.inception4e,
        net.inception5a,
        net.inception5b,
    ]

    net.conv1.conv = _compress_conv_layer(net.conv1.conv, args.e, is_first_layer=True)
    net.conv2.conv = _compress_conv_layer(net.conv2.conv, args.e, is_first_layer=True)
    net.conv3.conv = _compress_conv_layer(net.conv3.conv, args.e)

    for inception in inceptions:
        for branch in [
            inception.branch1,
            inception.branch2,
            inception.branch3,
            inception.branch4,
        ]:
            _compress_googlenet_branch(branch, args.e)

    net.fc = fc_svd(net.fc, args.e)


def compress(args, net):
    match args.net:
        case "imagenet_resnet18":
            compress_resnet18(args, net)
        case "imagenet_resnet50":
            compress_resnet50(args, net)
        case "imagenet_googlenet":
            compress_googlenet(args, net)
        case "imagenet_vgg16":
            compress_vgg(args, net)
        case "imagenet_alexnet":
            compress_alexnet(args, net)
        case "cifar10_densenet40":
            compress_densenet(args, net)
        case _:
            raise NotImplementedError(
                f"Compression for network {args.net} is not supported"
            )
