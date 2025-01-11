<!-- omit in toc -->
# Compressing convolutional neural networks with hierarchical Tucker-2 decomposition


Official PyTorch implementation of the paper [*Compressing convolutional neural networks with hierarchical Tucker-2 decomposition*](https://www.sciencedirect.com/science/article/pii/S156849462200905X) (Applied Soft Computing), by [Mateusz Gabor](https://scholar.google.com/citations?user=z5JaKvYAAAAJ&hl=pl&oi=ao) and [Rafał Zdunek](https://scholar.google.com/citations?user=z6o7_iQAAAAJ&hl=pl&oi=ao).

<p align="center" width="150%">
    <img src="assets\HT2.png" width="90%" height="75%">
</p>

<!-- omit in toc -->
## Table of Contents
- [Abstract](#abstract)
- [Setup](#setup)
- [Baselines](#baselines)
- [Compression](#compression)
- [Fine-tuning](#fine-tuning)
- [Information](#information)
- [Citation](#citation)

## Abstract

Convolutional neural networks (CNNs) play a crucial role and achieve top results in computer vision tasks but at the cost of high computational cost and storage complexity. One way to solve this problem is the approximation of the convolution kernel using tensor decomposition methods. In this way, the original kernel is replaced with a sequence of kernels in a lower-dimensional space. This study proposes a novel CNN compression technique based on the hierarchical Tucker-2 (HT-2) tensor decomposition and makes an important contribution to the field of neural network compression based on low-rank approximations. We demonstrate the effectiveness of our approach on many CNN architectures on CIFAR-10 and ImageNet datasets. The obtained results show a significant reduction in parameters and FLOPS with a minor drop in classification accuracy. Compared to different state-of-the-art compression methods, including pruning and matrix/tensor decomposition, the HT-2, as a new alternative, outperforms most of the cited methods.

## Setup
Before running the code, ensure you have installed Python and Conda. 

Download this repo and create a Conda environment using the provided `environment.yml` file.


```
git clone https://github.com/mateuszgabor/ht2 && cd ht2 
conda env create --prefix ./myenv -f environment.yml
conda activate myenv/
```

Next, download the ImageNet dataset from http://www.image-net.org/.

## Baselines

To train `Densenet-40` on CIFAR-10, run the following command:

```
python cifar.py -net cifar10_densenet40 -mode train -lr 0.1 -epochs 300 -wd 1e-4 -b 64 -momentum 0.9
```

To train `Alexnet` on ImageNet, run the following command:

```
python imagenet.py -net imagenet_alexnet -mode train -train <train_path> -val <val_path> -lr 0.05 -epochs 100 -wd 5e-4 -b 128 -workers 4 
```
where
- `<train_path>` is the path to the training dataset,
- `<val_path>` is the path to the validation dataset.

The weights of rest models: ResNet-18, ResNet-50, VGG-16, and GoogleNet are taken from pretrained Torchvision models.

## Compression

To compress a network, run the following command:

```
python compress_network.py -net <net_name> -weights <weights_path> -e <energy_level>
```
where
- `<net_name>` is the name of the network to compress from the list: 
    - `imagenet_resnet18`
    - `imagenet_resnet34`
    - `imagenet_resnet50`
    - `imagenet_googlenet`
    - `imagenet_vgg16`
    - `imagenet_alexnet`
    - `cifar10_densenet40`
- `<weights_path>` (optional) is the path to the weights of the network (required for `imagenet_alexnet` and `cifar10_densenet40`),
- `<energy_level>` is the energy level of compression.

The compressed weights are saved in the `decomposed_weights` folder.

The following command will compress ResNet-18 with energy level 0.9.

```
python compress_network.py -net imagenet_resnet18 -e 0.9
```

## Fine-tuning

To fine-tune Densenet-40 on CIFAR-10, run the following command:

```
python cifar.py -net cifar10_densenet40 -weights <weights_path> -mode fine_tune -train <train_path> -val <val_path> -lr <lr> -epochs <epochs> -wd <weight_decay> -b <batch_size> -momentum <momentum>
```
where
- `<weights_path>` is the path to the weights of the network,
- `<train_path>` is the path to the training dataset,
- `<val_path>` is the path to the validation dataset,
- `<lr>` is the learning rate,
- `<epochs>` is the number of epochs,
- `<weight_decay>` is the weight decay,
- `<batch_size>` is the batch size,
- `<momentum>` is the momentum.

The following command is an example of how to fine-tune Densenet-40.

```
python cifar.py -net cifar10_densenet40 -weights compressed_weights/ht2_densenet40_0.9.pth -mode fine_tune -lr 0.01 -epochs 300 -wd 3e-4 -b 64 -momentum 0.9
``` 

To fine-tune network on ImageNet, run the following command:

```
python imagenet.py -net <net_name> -weights <weights_path> -mode fine_tune -lr <lr> -epochs <epochs> -wd <weight_decay> -b <batch_size> -workers <workers>
```
where
- `<net_name>` is the name of the network to fine-tune from the list:
    - `imagenet_resnet18`
    - `imagenet_resnet34`
    - `imagenet_resnet50`
    - `imagenet_googlenet`
    - `imagenet_vgg16`
    - `imagenet_alexnet`
- `<weights_path>` is the path to the weights of the network,
- `<lr>` is the learning rate,
- `<epochs>` is the number of epochs,
- `<weight_decay>` is the weight decay,
- `<batch_size>` is the batch size,
- `<workers>` is the number of workers.


```
python imagenet.py -net imagenet_resnet18 -weights compressed_weights/ht2_resnet18_0.9.pth -mode fine_tune -lr 0.01 -epochs 20 -wd 1e-4 -b 128 -workers 4
```

## Information

In the first version of this work, the hierarchical Tucker-2 decomposiiton (only the deocmposition code) was implemented in Matlab and used in Python using the [Matlab Engine for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html). The current version of the code is implemented is fully implemented in Python. Hence, the results can be slightly different than in the paper (numerical errors etc.).

## Citation
If you find this code useful for your research, please cite our paper:

```bibtex
@article{gabor2023compressing,
  title={Compressing convolutional neural networks with hierarchical Tucker-2 decomposition},
  author={Gabor, Mateusz and Zdunek, Rafa{\l}},
  journal={Applied Soft Computing},
  volume={132},
  pages={109856},
  year={2023},
  publisher={Elsevier}
}
```
