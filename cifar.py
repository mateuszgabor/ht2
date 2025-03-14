import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import AverageMeter, accuracy, get_network, load_model


def train(trainloader, model, criterion, optimizer, epoch, scheduler, mode):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = tqdm(trainloader, desc=f"Epoch {epoch} [TRAIN]", leave=False)
    for inputs, targets in pbar:
        inputs, targets = (
            inputs.cuda(non_blocking=True),
            targets.cuda(non_blocking=True),
        )
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, _ = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if mode == "fine_tune":
            scheduler.step()

        pbar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.4f}"})

    print(f"TRAIN, Epoch: {epoch}, Avg. loss: {losses.avg:.4f}, Top-1: {top1.avg:.4f}")


def test(testloader, model, criterion, epoch):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    pbar = tqdm(testloader, desc=f"Epoch {epoch} [TEST]", leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = (
                inputs.cuda(non_blocking=True),
                targets.cuda(non_blocking=True),
            )

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, _ = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))

            pbar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{top1.avg:.4f}"})

    print(f"TEST, Epoch: {epoch}, Avg. loss: {losses.avg:.4f}, Top-1: {top1.avg:.4f}")

    return top1.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument("-mode", type=str, required=True, default="train")
    parser.add_argument("-weights", type=str, required=False)
    parser.add_argument("-lr", type=float, default=0.1, required=False)
    parser.add_argument("-epochs", type=int, default=200, required=False)
    parser.add_argument("-wd", type=float, default=1e-4, required=False)
    parser.add_argument("-b", type=int, default=128, required=False)
    parser.add_argument("-momentum", type=float, default=0.9, required=False)
    args = parser.parse_args()

    p = Path(__file__)
    if args.mode == "train":
        weight_path = f"{p.parent}/weights"
        end = f"{args.net}.pth"
    elif args.mode == "fine_tune":
        weight_path = f"{p.parent}/fine_tuned"
        end = args.weights
        end = end.split("/")[-1]
    else:
        raise ValueError(f"Wrong mode: {args.mode}")

    Path(weight_path).mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        model = get_network(args.net)
    elif args.mode == "fine_tune":
        model = load_model(args.weights)
    else:
        raise ValueError(f"Wrong mode: {args.mode}")

    model.cuda()
    dataset = datasets.CIFAR10
    mean = [0.49139968, 0.48215841, 0.44653091]
    stdv = [0.24703223, 0.24348513, 0.26158784]
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ]
    )
    trainset = dataset(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = data.DataLoader(
        trainset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True
    )

    testset = dataset(
        root="./data", train=False, download=False, transform=transform_test
    )
    testloader = data.DataLoader(
        testset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True,
    )
    if args.mode == "train":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 225], gamma=0.1
        )
    elif args.mode == "fine_tune":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(trainloader),
            epochs=args.epochs,
        )
    else:
        raise ValueError("Wrong mode")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print()
        train(trainloader, model, criterion, optimizer, epoch, scheduler, args.mode)
        best_top1 = test(testloader, model, criterion, epoch)

        if best_acc < best_top1:
            print(f"Saving model file to {weight_path}/{end}")
            checkpoint = {"model": model, "state_dict": model.state_dict()}
            torch.save(checkpoint, f"{weight_path}/{end}")
            best_acc = best_top1
            continue

        if args.mode == "train":
            scheduler.step()


if __name__ == "__main__":
    main()
