import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms as T
from tqdm import tqdm

from utils import AverageMeter, accuracy, get_network, load_model

best_acc1 = 0

WORLD_SIZE = torch.cuda.device_count()
DIST_URL = "tcp://127.0.0.1:5000"
DIST_BACKEND = "nccl"

__imagenet_pca = {
    "eigval": torch.Tensor([0.2175, 0.0188, 0.0045]),
    "eigvec": torch.Tensor(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    ),
}


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )
        return img.add(rgb.view(3, 1, 1).expand_as(img))


def train(trainloader, model, criterion, optimizer, scheduler, epoch, gpu, scaler):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with tqdm(trainloader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            optimizer.zero_grad(set_to_none=True)
            tepoch.set_description(f"TRAIN Epoch {epoch}")
            inputs, targets = inputs.cuda(gpu), targets.cuda(gpu)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tepoch.set_postfix(loss=loss.item(), accuracy=acc1)

    print(
        f"TRAIN, Epoch: {epoch}, Avg. loss: {losses.avg:.4f}, Top-1: {top1.avg:.4f}, Top-5: {top5.avg:.4f}"
    )


def test(testloader, model, criterion, epoch, gpu):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                inputs, targets = inputs.cuda(gpu), targets.cuda(gpu)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                tepoch.set_postfix(loss=loss.item(), accuracy=acc1)

    print(
        f"TEST, Epoch: {epoch}, Avg. loss: {losses.avg:.4f}, Top-1: {top1.avg:.4f}, Top-5: {top5.avg:.4f}"
    )

    return top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", type=float, default=0.01, required=True)
    parser.add_argument("-mode", type=str, required=True, default="train")
    parser.add_argument("-epochs", type=int, default=90, required=True)
    parser.add_argument("-wd", type=float, default=1e-3, required=True)
    parser.add_argument("-b", type=int, default=128, required=True)
    parser.add_argument("-train", type=str, required=True, help="Train dataset folder")
    parser.add_argument("-val", type=str, required=True, help="Test dataset folder")
    parser.add_argument("-workers", type=int, default=4)
    parser.add_argument("-weights", type=str, required=True)
    parser.add_argument("-momentum", type=float, default=0.9, required=False)
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

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

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    rank = gpu
    dist.init_process_group(
        backend=DIST_BACKEND, init_method=DIST_URL, world_size=WORLD_SIZE, rank=rank
    )

    if args.mode == "train":
        model = get_network(args.net)
    else:
        model = load_model(args.weights)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        args.b = int(args.b / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )

    cudnn.benchmark = True

    traindir = args.train
    valdir = args.val
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.net == "imagenet_alexnet":
        train_dataset = datasets.ImageFolder(
            traindir,
            T.Compose(
                [
                    T.RandomResizedCrop(227),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    Lighting(0.1, __imagenet_pca["eigval"], __imagenet_pca["eigvec"]),
                    normalize,
                ]
            ),
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
        )

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                T.Compose(
                    [
                        T.Resize(256),
                        T.CenterCrop(227),
                        T.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.b,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            T.Compose(
                [
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            ),
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.b,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
        )

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                T.Compose(
                    [
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.b,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    if args.mode == "train" and args.net == "imagenet_alexnet":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.1, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
        )

    scaler = torch.amp.GradScaler()

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, criterion, optimizer, scheduler, epoch, gpu, scaler)
        acc1, _ = test(val_loader, model, criterion, epoch, gpu)
        if best_acc1 < acc1 and rank == 0:
            checkpoint = {
                "model": model.module,
                "epoch": epoch,
                "state_dict": model.module.state_dict(),
                "best_top1": acc1,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"{weight_path}/{end}")
            best_acc1 = acc1


if __name__ == "__main__":
    main()
