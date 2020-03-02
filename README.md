# Geometric Decision-based Attack (GeoDA)

This repository contains the official PyTorch implementation of SparseFool algorithm described in [1]. GeoDA is a Black-box attack to generate adversarial example for image classifiers which is presented in the following paper:


<img src = Demo.jpg:large' width='800px' />
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

[1] Ali Rahmati, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard, and Huaiyu Dai:
*A geometric framework for black-box adversarial attacks*. In Computer Vision and Pattern Recognition (CVPR ’20), IEEE, 2020.


