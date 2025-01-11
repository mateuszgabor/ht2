import tensorly as tl
import torch


def get_singular_values(X):
    _, R = tl.qr(X.T)
    R = R[0 : R.shape[1], :]
    _, S, _ = torch.linalg.svd(R.T, False)
    return S


def load_model(filepath):
    try:
        checkpoint = torch.load(filepath)
        model = checkpoint["model"]
        model.load_state_dict(checkpoint["state_dict"])
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {filepath}")
    except KeyError as e:
        raise KeyError(f"Invalid checkpoint format: missing key {e}")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_network(net_name):
    if net_name == "cifar10_densenet40":
        from models import densenet40

        net = densenet40()
    elif net_name == "imagenet_alexnet":
        from models import alexnet

        net = alexnet()
    else:
        raise NotImplementedError(
            f"The network {net_name} is currently not supported yet"
        )
    return net
