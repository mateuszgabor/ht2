import matlab
import numpy as np
import torch
from torch import nn


def estimate_ranks(weights, energy_threshold, eng):
    to_matlab = weights.tolist()
    mat_weights = matlab.double(to_matlab)

    unfold_0_mat = eng.unfold(weights, 0)
    unfold_0 = np.asarray(unfold_0_mat, dtype=np.float32)

    unfold_1_mat = eng.unfold(weights, 2)
    unfold_1 = np.asarray(unfold_1_mat, dtype=np.float32)

    unfold_2_mat = eng.canonical_matricization(mat_weights, 1, 2)
    unfold_2 = np.asarray(unfold_2_mat, dtype=np.float32)

    _, s, _ = np.linalg.svd(unfold_0)
    sum_0 = 0
    total_sum_0 = np.sum(s ** 2)
    count_0 = 0

    for i in s:
        sum_0 += i ** 2
        count_0 += 1
        energy = sum_0 / total_sum_0
        if energy > energy_threshold:
            break

    R1 = count_0
    _, s, _ = np.linalg.svd(unfold_1)
    sum_1 = 0
    total_sum_1 = np.sum(s ** 2)
    count_1 = 0

    for i in s:
        sum_1 += i ** 2
        count_1 += 1
        energy = sum_1 / total_sum_1

        if energy > energy_threshold:
            break

    R3 = count_1

    _, s, _ = np.linalg.svd(unfold_2)
    sum_2 = 0
    total_sum_2 = np.sum(s ** 2)
    count_2 = 0

    for i in s:
        sum_2 += i ** 2
        count_2 += 1
        energy = sum_2 / total_sum_2
        if energy > energy_threshold:
            break

    R13 = count_2

    return [R1, R3, R13]


def new_ht2_energy_decomposition(layer, eng, energy_threshold):
    is_bias = torch.is_tensor(layer.bias)
    weights = layer.weight.cpu().data.numpy()
    weights = np.moveaxis(weights, 1, 2)
    weights = np.swapaxes(weights, 1, 3)

    mat_weights = matlab.double(weights.tolist())
    R1, R3, R13 = estimate_ranks(weights, energy_threshold, eng)

    factors = eng.conv_new_ht2_decomposition(mat_weights, R1, R3, R13)

    fourth = np.asarray(factors[0], dtype=np.float32)
    first = np.asarray(factors[1], dtype=np.float32)
    third = np.asarray(factors[2], dtype=np.float32)
    second = np.asarray(factors[3], dtype=np.float32)

    first_weights = np.expand_dims(np.swapaxes(first, 0, 1), axis=(2, 3))
    second_weights = np.expand_dims(np.moveaxis(second, -1, 0), axis=3)
    third_weights = np.expand_dims(np.swapaxes(third, 1, 2), axis=2)
    fourth_weights = np.expand_dims(fourth, axis=(2, 3))

    first_layer = nn.Conv2d(
        in_channels=layer.in_channels,
        out_channels=R3,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    second_layer = nn.Conv2d(
        in_channels=R3,
        out_channels=R13,
        kernel_size=(layer.kernel_size[0], 1),
        stride=(layer.stride[0], 1),
        padding=(layer.padding[0], 0),
        dilation=layer.dilation,
        bias=False,
    )

    third_layer = nn.Conv2d(
        in_channels=R13,
        out_channels=R1,
        kernel_size=(1, layer.kernel_size[1]),
        stride=(1, layer.stride[1]),
        padding=(0, layer.padding[1]),
        dilation=layer.dilation,
        bias=False,
    )

    fourth_layer = nn.Conv2d(
        in_channels=R3,
        out_channels=layer.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=is_bias,
    )

    first_layer.weight.data = torch.from_numpy(first_weights)
    second_layer.weight.data = torch.from_numpy(second_weights)
    third_layer.weight.data = torch.from_numpy(third_weights)
    fourth_layer.weight.data = torch.from_numpy(fourth_weights)

    if is_bias:
        fourth_layer.bias.data = layer.bias.data

    return nn.Sequential(*[first_layer, second_layer, third_layer, fourth_layer])
