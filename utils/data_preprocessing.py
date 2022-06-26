# -*- coding=utf-8 -*-
# @Time: 2021/8/6 16:58
# @Author: N
# @Software: PyCharm

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def get_mean_and_std_from_json(dir_path):
    if os.path.exists(Path(dir_path) / 'mean_and_std.json'):
        json_dic = json.load(open(Path(dir_path) / 'mean_and_std.json', 'r'))
        return json_dic['mean'], json_dic['std']
    return None


def get_mean_and_std(input_path_list: list, input_open_method: callable, resize_method = None,
                     size_with_max_3dim = None):
    if not input_path_list: raise ValueError('container of data paths shall not be empty')
    if not (isinstance(input_path_list[0], Path) or isinstance(input_path_list[0], str)): raise ValueError(
        'element type of input_path_list shall be Path or str')
    dir_path = os.path.split(input_path_list[0])[0]
    mean_and_std = get_mean_and_std_from_json(dir_path)
    if mean_and_std:
        return mean_and_std
    print('Needing some time to calculate mean and std, just wait up.', flush = True)
    proc_bar = tqdm(desc = 'calculation', total = len(input_path_list), ncols = 0, unit = 'img', delay = 0.1)
    shape = size_with_max_3dim if size_with_max_3dim else np.array(input_open_method(input_path_list[0])).shape
    pixels_per_img = shape[0] * shape[1]
    num_images = len(input_path_list)
    if len(shape) == 3:
        img_mean_sum, residual_square_sum = [0 for _ in range(3)], [0 for _ in range(3)]
        for path in input_path_list:
            input_img = np.array(input_open_method(path))
            input_img = resize_method(input_img) if resize_method else input_img
            input_img = input_img.transpose((2, 0, 1)) / 255
            for i in range(len(shape)):
                scratch_mean = input_img[i].sum() / pixels_per_img
                img_mean_sum[i] += scratch_mean
                residual_square_sum[i] += (
                    ((input_img[i] - np.ones(shape[:2]).astype(float) * scratch_mean) ** 2).sum())
            proc_bar.update(1)
        mean, std = [img_mean_sum[i] / num_images for i in range(3)], [
            (residual_square_sum[i] / ((pixels_per_img * num_images - 1))) ** (1 / 2) for i in range(3)]
    elif len(shape) == 2:
        img_mean_sum, residual_square_sum = 0, 0
        for path in input_path_list:
            input_img = np.array(input_open_method(path))
            input_img = resize_method(input_img) if resize_method else input_img
            input_img = input_img / 255
            scratch_mean = input_img.sum() / pixels_per_img
            img_mean_sum += scratch_mean
            residual_square_sum += (
                ((input_img - np.ones(shape).astype(float) * scratch_mean) ** 2).sum())
            proc_bar.update(1)
        mean, std = [img_mean_sum / num_images], [
            (residual_square_sum / ((pixels_per_img * num_images - 1))) ** (1 / 2)]
    else:
        raise ValueError('incorrect input image format')
    json.dump({'mean': mean, 'std': std}, open(Path(dir_path) / 'mean_and_std.json', 'w'))
    time.sleep(0.1)
    print(
        'Calculation complete, the result stored in mean_and_std.json under data dir is as following: \nmean: {}, std: {}\nOn with the show.'.format(
            [float(f'{elem:.3f}') for elem in mean], [float(f'{elem:.3f}') for elem in std]), flush = True)
    return mean, std


def resize_keep_aspectratio(img_arr, dst_size, is_label: bool = False):
    src_h, src_w = img_arr.shape[:2]
    dst_h, dst_w = dst_size
    interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR

    # to judge do with height-wise or width-wise
    h = dst_w * (float(src_h) / src_w)
    w = dst_h * (float(src_w) / src_h)

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv2.resize(img_arr, (dst_w, int(h)), interpolation = interpolation)
    else:
        image_dst = cv2.resize(img_arr, (int(w), dst_h), interpolation = interpolation)

    h_, w_ = image_dst.shape[:2]

    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_ + 1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_ + 1) / 2)

    value = [0, 0, 0]
    borderType = cv2.BORDER_CONSTANT
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)

    return image_dst


def get_label_boundary(img_arr, dst_size):
    if not dst_size:
        raise Exception('dst_size should be indicated')
    src_h, src_w = img_arr.shape[:2]
    dst_h, dst_w = dst_size

    # to judge do with height-wise or width-wise
    h = dst_w * (float(src_h) / src_w)
    w = dst_h * (float(src_w) / src_h)

    h = int(h)
    w = int(w)

    if h <= dst_h:
        h_, w_ = int(h), dst_w
    else:
        h_, w_ = dst_h, int(w)

    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_ + 1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_ + 1) / 2)

    h_up_boundary = top
    h_down_boundary = h_ + top
    w_left_boundary = left
    w_right_boundary = left + w_

    return torch.LongTensor([h_up_boundary, h_down_boundary, w_left_boundary, w_right_boundary])
