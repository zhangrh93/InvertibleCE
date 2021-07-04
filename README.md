# InvertibleCE: Invertible Concept-based Explanation (ICE)
Code for our paper [Invertible Concept-based Explanations for CNN Models with Non-negative Concept Activation Vectors](https://arxiv.org/abs/2006.15417) published in AAAI 2021

## Introduction

It's a powerful CNN explanation framework. It learns domain related concepts based on given datasets and provide both global (class level) and local (instance level) explanations. Learned concepts could be easily understanded by human.


## Demo
Two Colab jupyter notebook demos are available. In [ImageNet.ipynb](https://colab.research.google.com/github/zhangrh93/InvertibleCE/blob/main/demo/ImageNet.ipynb), you can have fun with different dog concepts and explanations from torchvision pretrained models.

If you want a guide for your own model and dataset, have a try with [MNIST.ipynb](https://colab.research.google.com/github/zhangrh93/InvertibleCE/blob/main/demo/MNIST.ipynb).


## Usage

**You need to install graphviz for explanation visualization. It could not be installed with pip or conda.**

It's a pytorch based implement. All dependent packages are included in requirements.txt

    pip install -r requirements.txt

## Explanation examples

![Husky explanation](https://github.com/zhangrh93/InvertibleCE/blob/main/Examples/248%20Eskimo%20dog%2C%20husky.jpg)
