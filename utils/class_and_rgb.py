# -*- coding=utf-8 -*-
# @Time: 2021/8/6 16:36
# @Author: N
# @Software: PyCharm

from pathlib import Path
import csv
import numpy as np
import torch


def get_class2name_lut_and_class2rgb_lut(dir_name):
    with open(Path(dir_name) / 'class_dict.csv') as f:
        csv_reader = csv.reader(f)
        class2name_lut, class2rgb_lut = [], []
        next(csv_reader)
        for row_list in csv_reader:
            class2name_lut.append(row_list[0])
            class2rgb_lut.append([int(row_list[i]) for i in range(1, 4)])
        return class2name_lut, class2rgb_lut


def rgb2class(img_array, class2rgb_lut: list):
    if isinstance(class2rgb_lut[0], int):
        for i in range(len(class2rgb_lut)):
            img_array[img_array == class2rgb_lut[i]] = i
        return img_array
    t = []
    for class_value in class2rgb_lut:
        t.append((img_array == class_value).all(axis = -1))
    return np.stack(t, axis = -1).argmax(axis = -1)


def class2rgb(t, class2rgb_lut):
    # t shall be a tensor with tensor.size(h,w) with element of class number
    rgb = [torch.zeros_like(t) for _ in range(3)]
    for i in range(len(class2rgb_lut)):
        temp = t == i
        for j in range(3): rgb[j] += temp * class2rgb_lut[i][j]
    return torch.stack(rgb, dim = 1).long() if t.dim() == 3 else torch.stack(rgb, dim = 0).long()
