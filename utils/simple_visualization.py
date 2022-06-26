# -*- coding=utf-8 -*-
# @Time: 2021/10/5 20:54
# @Author: N
# @Software: PyCharm
from matplotlib import pyplot as plt


def show_img(image_tensor):
    plt.figure()
    if len(image_tensor.shape) == 3:
        plt.imshow(image_tensor.numpy().transpose(1, 2, 0))
    else:
        plt.imshow(image_tensor.numpy())
    plt.show()