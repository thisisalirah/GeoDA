# Geometric Decision-based Attack (GeoDA)

This repository contains the official PyTorch implementation of GeoDA algorithm described in [1]. GeoDA is a Black-box attack to generate adversarial example for image classifiers. 

## A few examples on the performance of the GeoDA for different norms

![Demo](https://user-images.githubusercontent.com/36679506/75689719-aa821b00-5c6f-11ea-9b6b-b78ff3ed871b.jpg)

## Requirements

To execute the code, please make sure that the following packages are installed:

- [Foolbox](https://foolbox.readthedocs.io/en/stable/user/installation.html)
- [NumPy](https://docs.scipy.org/doc/numpy-1.15.1/user/install.html)
- [PyTorch and Torchvision](https://pytorch.org/) (install with CUDA if available)
- [matplotlib](https://matplotlib.org/users/installing.html)



## Contents

### GeoDA.py

This function implements the GeoDA algorithm.





### utils.py

Includes general functions

### data/

Contains some examples for the demos. 



## Reference

[1] Ali Rahmati, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard, and Huaiyu Dai,
*GeoDA: a geometric framework for black-box adversarial attacks*. in CVF/IEEE Computer Vision and Pattern Recognition (CVPR'20), 2020. [[arXiv pre-print]](http://arxiv.org/abs/2003.06468)

