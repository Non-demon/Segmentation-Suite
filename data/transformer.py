# -*- coding=utf-8 -*-
# @Time: 2021/10/12 10:58
# @Author: N
# @Software: PyCharm
import cv2
import utils
from torchvision import transforms
import torch
import numpy as np


class TransformerOfInputArr4BeitPretraining:
    def __init__(self, window_size = (14, 14), num_masking_patches = 75, max_mask_patches_per_block = None,
                 min_mask_patches_per_block = 16, img_size4vae = (112, 112), mean = None, std = None,
                 input_path_list = None, input_open_method = cv2.imread):

        if not mean or not std:
            if input_path_list:
                mean, std = utils.get_mean_and_std(input_path_list, input_open_method)
            else:
                raise Exception("require std and mean or input_path_list but get non")
        normalize = transforms.Normalize(mean, std)
        arr2tensor = lambda arr: torch.FloatTensor(arr.transpose((2, 0, 1)) / 255)

        def transformer4img(arr):
            return normalize(arr2tensor(arr))

        def map_pixels(x: torch.Tensor) -> torch.Tensor:
            logit_laplace_eps: float = 0.1
            if x.dtype != torch.float:
                raise ValueError('expected input to have type float')
            return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

        def transformer4visual_token(arr):
            return map_pixels(arr2tensor(arr))

        self.transformer4img = transformer4img
        self.transformer4visual_token = transformer4visual_token

        self.masked_position_generator = utils.MaskingGenerator(
            input_size = window_size, num_masking_patches = num_masking_patches,
            max_num_patches = max_mask_patches_per_block,
            min_num_patches = min_mask_patches_per_block)
        self.img_size4vae = img_size4vae

    def __call__(self, input_arr):
        arr4vae = cv2.pyrDown(input_arr, dstsize = self.img_size4vae)
        return self.transformer4img(input_arr), self.transformer4visual_token(
            arr4vae), self.masked_position_generator()


class TransformerOfInputArr4BeitTraining:
    def __init__(self, window_size = (14, 14), num_masking_patches = 75, max_mask_patches_per_block = None,
                 min_mask_patches_per_block = 16, img_size4vae = (112, 112), mean = None, std = None,
                 input_path_list = None, input_open_method = cv2.imread):

        if not mean or not std:
            if input_path_list:
                mean, std = utils.get_mean_and_std(input_path_list, input_open_method)
            else:
                raise Exception("require std and mean or input_path_list but get non")
        normalize = transforms.Normalize(mean, std)
        arr2tensor = lambda arr: torch.FloatTensor(arr.transpose((2, 0, 1)) / 255)

        def transformer4img(arr):
            return normalize(arr2tensor(arr))

        self.transformer4img = transformer4img

    def __call__(self, input_arr):
        return self.transformer4img(input_arr)


def transformer4label_arr_of_drive(class2rgb_lut):
    def transformer4label_arr(label_arr):
        return torch.Tensor(utils.rgb2class(label_arr, class2rgb_lut)).long()

    return transformer4label_arr
