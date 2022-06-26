# -*- coding=utf-8 -*-
# @Time: 2021/10/5 20:54
# @Author: N
# @Software: PyCharm
from PIL import Image
import torch
from matplotlib import pyplot as plt


def show_img(img_or_tensor_or_arr):
    plt.figure()
    if isinstance(img_or_tensor_or_arr, Image.Image):
        plt.imshow(img_or_tensor_or_arr)
    elif len(img_or_tensor_or_arr.shape) == 3:
        plt.imshow(img_or_tensor_or_arr.numpy().transpose(1, 2, 0) if isinstance(img_or_tensor_or_arr,
                                                                                 torch.Tensor) else img_or_tensor_or_arr)
    else:
        plt.imshow(img_or_tensor_or_arr.numpy() if isinstance(img_or_tensor_or_arr,
                                                              torch.Tensor) else img_or_tensor_or_arr)
    plt.show()


def img_tensor2arr(tensor):
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)
