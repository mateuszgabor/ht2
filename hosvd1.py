import tensorly as tl
import torch
import torch.nn as nn
from ht2_decomposition import get_singular_values
from tensorly.decomposition import partial_tucker


def estimate_rank(weights, energy_threshold):
    rank = None
    unfold = tl.unfold(weights, 0)

    s = get_singular_values(
        unfold,
    )
    total_sum = torch.sum(s**2)
    s_sum = 0
    count = 0
    for i in s:
        s_sum += i**2
        count += 1
        energy = s_sum / total_sum
        if energy > energy_threshold:
            rank = count

    return rank


def hosvd1(layer, energy_threshold):
    tl.set_backend("pytorch")
    is_bias = torch.is_tensor(layer.bias)
    weights = layer.weight.detach()
    R = estimate_rank(weights, energy_threshold)
    out, _ = partial_tucker(weights, R, [0])
    T, factors = out
    U = factors[0]

    first = T
    second = U

    inChannels = layer.in_channels
    outChannels = layer.out_channels

    firstWeights = first if first.ndim > 2 else first.unsqueeze(2).unsqueeze(3)
    secondWeights = second.unsqueeze(2).unsqueeze(3)

    first_layer = nn.Conv2d(
        in_channels=inChannels,
        out_channels=R,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    second_layer = nn.Conv2d(
        in_channels=R,
        out_channels=outChannels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=is_bias,
    )

    first_layer.weight.data = firstWeights
    second_layer.weight.data = secondWeights
    if is_bias:
        second_layer.bias.data = layer.bias.data

    return nn.Sequential(first_layer, second_layer)
