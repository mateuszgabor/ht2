import argparse
from pathlib import Path

import torch
import torchvision.models as models

from compression import compress
from utils import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument(
        "-weights",
        type=str,
        required=False,
        help="the weights file of the baseline network",
    )
    parser.add_argument(
        "-e", type=float, default=False, help="Energy level of compression"
    )
    args = parser.parse_args()
    p = Path(__file__)

    match args.net:
        case "imagenet_resnet50":
            net = models.resnet50(True)
        case "imagenet_resnet18":
            net = models.resnet18(True)
        case "imagenet_googlenet":
            net = models.googlenet(True)
        case "imagenet_vgg16":
            net = models.vgg16_bn(True)
        case "imagenet_alexnet":
            net = load_model(args.weights)
        case "cifar10_densenet40":
            net = load_model(args.weights)
        case _:
            raise NotImplementedError(
                f"The network {args.net} is currently not supported yet"
            )

    compress(args, net)
    print(net)

    Path(f"{p.parent}/compressed_weights/").mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": net, "state_dict": net.state_dict()}
    tmp = str(args.e)
    torch.save(
        checkpoint,
        f"{p.parent}/compressed_weights/ht2_{args.net}_{tmp.replace('.', '_')}.pth",
    )
