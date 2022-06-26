# -*- coding=utf-8 -*-
# @Time: 2021/8/6 17:05
# @Author: N
# @Software: PyCharm
import math
from typing import Tuple, Any

import numpy as np
import os
import torch
import utils
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, dir_name, aug = None, dilation_rate = 0, normalization = False, mean = None, std = None,
                 train_val_or_test = 'train'):
        if train_val_or_test not in ('train', 'val', 'test'): raise Exception(
            'unexpected format of train_val_or_test in my_dataset constructor')
        super(MyDataset, self).__init__()
        _, self.class2rgb_lut = utils.get_class2name_lut_and_class2rgb_lut(dir_name)
        self.num_classes = len(self.class2rgb_lut)
        input_dir, label_dir = dir_name + '/' + train_val_or_test + '/', dir_name + '/' + train_val_or_test + '_labels' + '/'
        dir_list = os.listdir(input_dir)
        self.data = []
        for file in dir_list:
            name, ext = os.path.splitext(file)
            if ext != '.json': self.data.append([input_dir + file, label_dir + name + '_L' + ext])
        if normalization and (not mean or not std): mean, std = utils.get_mean_and_std(self.data)
        self.len, self.data_len, self.aug, self.normalize, self.denormalize = int(
            len(self.data) * (dilation_rate)), len(self.data), aug, transforms.Normalize(mean, std,
                                                                                         inplace = True) if normalization else transforms.Normalize(
            0, 1, inplace = True), lambda x: (x * torch.FloatTensor(std) + torch.FloatTensor(mean))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        is_ngp, index = (index / self.data_len) >= 1, index % len(self.data)
        # ngp is the acronym of "new game plus"
        try:
            if self.aug:
                input, label = self.aug(np.array(Image.open(self.data[index][0])),
                                        np.array(Image.open(self.data[index][1])), is_ngp)
            else:
                input, label = np.array(Image.open(self.data[index][0])), np.array(Image.open(self.data[index][1]))
        except:
            raise Exception('lacking labels to match some certain input images')
        return self.normalize(torch.FloatTensor(input).permute((2, 0, 1)) / 255), torch.LongTensor(
            utils.rgb2class(label, self.class2rgb_lut))


class BaseDataset(Dataset):
    def __init__(self, input_path_list: list, input_path2label_path: callable, input_open_method: callable,
                 label_open_method: callable, aug_method: callable, transformer4input_arr: callable,
                 transformer4label_arr: callable, aug_dilation_rate = 1):
        # class2rgb_lut[class_index]=[r,g,b]
        super(BaseDataset, self).__init__()
        self.input_path_list = input_path_list
        self.get_label_path = input_path2label_path
        self.aug_dilation_rate = aug_dilation_rate
        self.aug_method = aug_method
        self.input_open_method = input_open_method
        self.label_open_method = label_open_method
        self.transformer4input_arr = transformer4input_arr
        self.transformer4label_arr = transformer4label_arr
        self.len_before_aug = len(input_path_list)
        self.len_after_aug = math.floor(self.len_before_aug * aug_dilation_rate)

    def __len__(self):
        return self.len_after_aug

    def __getitem__(self, index) -> Tuple[Any, Any]:
        is_ngp, index = (index / self.len_before_aug) >= 1, index % self.len_before_aug
        input_path = self.input_path_list[index]
        input_arr, label_arr = self.aug_method(np.array(self.input_open_method(input_path)),
                                               np.array(self.label_open_method(self.get_label_path(input_path))),)
        return self.transformer4input_arr(input_arr), self.transformer4label_arr(label_arr)


class Dataset4GeneratingPredictsWithKeepingAspectRatioResize(BaseDataset):

    def __init__(self, get_label_boundary, *args, **kwargs):
        super(Dataset4GeneratingPredictsWithKeepingAspectRatioResize, self).__init__(*args, **kwargs)
        self.get_label_boundary = get_label_boundary

    def __getitem__(self, index) -> Tuple[Any, Any]:
        input_path = self.input_path_list[index]
        input_arr = np.array(self.input_open_method(input_path))
        boundary = self.get_label_boundary(input_arr)
        augmented_input_arr, augmented_label_arr = self.aug_method(input_arr, np.array(
            self.label_open_method(self.get_label_path(input_path))))
        return (self.transformer4input_arr(augmented_input_arr), os.path.split(input_path)[1],
                torch.LongTensor(input_arr.shape[:2]), boundary), self.transformer4label_arr(augmented_label_arr)


class Dataset4GeneratingPredicts(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(Dataset4GeneratingPredicts, self).__init__(*args, **kwargs)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        index = (index / self.len_before_aug) >= 1, index % self.len_before_aug
        input_path = self.input_path_list[index]
        input_arr = np.array(self.input_open_method(input_path))
        augmented_input_arr, augmented_label_arr = self.aug_method(input_arr, np.array(
            self.label_open_method(self.get_label_path(input_path))))
        return (self.transformer4input_arr(augmented_input_arr), os.path.split(input_path)[1],
                torch.LongTensor(input_arr.shape[:2])), self.transformer4label_arr(augmented_label_arr)
