# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:17:56 2020

@author: evava
"""
import sys
import numpy as np
from torch import nn
from collections import OrderedDict
from nowcasting.config import cfg
# import cv2
import os.path as osp
import os
import torch
if cfg.GLOBAL.DEVICE == torch.device("cuda:0"):
    import cupy 

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))

def dBZ_to_rainfall(dbz, a=200, b=1.6):
    return ((10**(dbz/10))/a)**(1/b)

def rainfall_to_dBZ(rainfall, a=200, b=1.6):
    if cfg.GLOBAL.DEVICE == torch.device("cuda"):
        return (10*(cupy.log10((a*(rainfall**(b)))))) 
    if cfg.GLOBAL.DEVICE == torch.device("cpu"):
        return (10*(np.log10((a*(rainfall**(b))))))

def pixel_to_dBZ(pixel):
    return ((pixel*0.5)-32)

def dBZ_to_pixel(dbz):
    return ((dbz+32)*2)

def pixel_to_rainfall(pixel):
    dbz = pixel_to_dBZ(pixel)
    rainfall = dBZ_to_rainfall(dbz)
    return rainfall

def rainfall_to_pixel(rainfall):
    dbz = rainfall_to_dBZ(rainfall)
    pixel = dBZ_to_pixel(dbz)
    return pixel


# DL 27-05 wordt niet gebruikt
# def count_pixels(name=None):
#     png_dir = cfg.HKO_PNG_PATH
#     mask_dir = cfg.HKO_MASK_PATH
#     counts = np.zeros(256, dtype=np.float128)
#     for root, dirs, files in os.walk(png_dir):
#         for file_name in files:
#             if not file_name.endswith('.png'):
#                 continue
#             tmp_dir = '/'.join(root.split('/')[-3:])
#             png_path = osp.join(png_dir, tmp_dir, file_name)
#             mask_path = osp.join(mask_dir, tmp_dir, file_name.split('.')[0]+'.mask')
#             label, count = np.unique(cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2GRAY)[read_mask_file(mask_path)], return_counts=True)
#             counts[label] += count
#     if name is not None:
#         np.save(name, counts)
#     return counts

# def pixel_to_dBZ(img):
#     """

#     Parameters
#     ----------
#     img : np.ndarray or float

#     Returns
#     -------

#     """
#     return img * 70.0 - 10.0

# def dBZ_to_pixel(dBZ_img):
#     """

#     Parameters
#     ----------
#     dBZ_img : np.ndarray

#     Returns
#     -------

#     """
#     return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)

# DL 07-06, minp aangepast van 0.4 naar 1e-4. 
# def pixel_to_rainfall(img, minp=1e-4, maxp=50):
#     rainfall_intensity = 10**(np.log10(minp) + img/255 * (np.log10(maxp)-np.log10(minp))) 
#     return rainfall_intensity

# # DL 07-06, minp aangepast van 0.4 naar 1e-4. 
# def rainfall_to_pixel(p, minp=1e-4, maxp=50):
#     pixel_vals = (np.log10(p) - np.log10(minp)) / (np.log10(maxp) - np.log10(minp)) * 255
#     return pixel_vals

# def pixel_to_rainfall(img, a=58.53, b=1.56):
#     """Convert the pixel values to real rainfall intensity

#     Parameters
#     ----------
#     img : np.ndarray
#     a : float32, optional
#     b : float32, optional

#     Returns
#     -------
#     rainfall_intensity : np.ndarray
#     """
#     dBZ = pixel_to_dBZ(img)
#     dBR = (dBZ - 10.0 * np.log10(a)) / b
#     rainfall_intensity = np.power(10, dBR / 10.0)
#     return rainfall_intensity


# def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
#     """Convert the rainfall intensity to pixel values

#     Parameters
#     ----------
#     rainfall_intensity : np.ndarray
#     a : float32, optional
#     b : float32, optional

#     Returns
#     -------
#     pixel_vals : np.ndarray
#     """
#     dBR = np.log10(rainfall_intensity) * 10.0
#     # dBZ = 10b log(R) +10log(a)
#     dBZ = dBR * b + 10.0 * np.log10(a)
#     pixel_vals = (dBZ + 10.0) / 70.0
#     return pixel_vals

# def dBZ_to_rainfall(dBZ, a=58.53, b=1.56):
#     return np.power(10, (dBZ - 10 * np.log10(a))/(10*b))

# def rainfall_to_dBZ(rainfall, a=58.53, b=1.56):
#     return 10*np.log10(a) + 10*b*np.log10(rainfall)

