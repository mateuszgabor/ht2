import torch
import torch.nn as nn
from ht2_decomposition import get_singular_values


def estimate_rank(weights, energy_threshold):
    rank = None
    s = get_singular_values(weights)
    total_sum = torch.sum(s**2)
    s_sum = 0
    count = 0

    for i in s:
        s_sum += i**2
        count += 1
        energy = s_sum / total_sum
        if energy > energy_threshold:
            rank = count
            break

    return rank


def svd_decomposition(layer, energy_threshold):
    weights = layer.weight.detach()
    is_bias = torch.is_tensor(layer.bias)
    rank = estimate_rank(weights, energy_threshold)

    U, S, Vt = torch.linalg.svd(weights, full_matrices=False)
    w1 = U[:, 0:rank]
    w0 = torch.diag(S[0:rank]) @ Vt[0:rank, :]

    first_layer = nn.Linear(w0.shape[1], w0.shape[0], bias=False)
    second_layer = nn.Linear(w1.shape[1], w1.shape[0], bias=is_bias)

    first_layer.weight.data = w0
    second_layer.weight.data = w1

    if is_bias:
        second_layer.bias.data = layer.bias.data

    return nn.Sequential(first_layer, second_layer)
