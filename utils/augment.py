# -*- coding=utf-8 -*-
# @Time: 2021/8/6 16:56
# @Author: N
# @Software: PyCharm
import math
import numbers

import cv2
import numpy as np
import random
import utils


class img_augment:

    def __init__(self, height, width, crop_height = 0, crop_width = 0, random_crop_ratio_low_boundary = 0.6,
                 random_crop_ratio_upper_boundary = 1, zoom_in_and_out = True, h_flip = False, v_flip = False,
                 brightness = 0, rotation = 90, keep_aspect_ratio = True):
        self.crop_height, self.crop_width, self.h_flip, self.v_flip, self.brightness, self.rotation = crop_height, crop_width, h_flip, v_flip, brightness, rotation
        self.height, self.width = height, width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.random_crop_ratio_low_boundary = random_crop_ratio_low_boundary
        self.random_crop_ratio_upper_boundary = random_crop_ratio_upper_boundary
        self.zoom_in_and_out = zoom_in_and_out

    def __call__(self, input_arr: np.array, label_arr: np.array):
        zoom_in = None
        crop_height = self.crop_height
        crop_width = self.crop_width
        orig_height = input_arr.shape[0]
        orig_width = input_arr.shape[1]
        if input_arr.shape[0] != label_arr.shape[0] or input_arr.shape[1] != label_arr.shape[1]:
            input_arr = cv2.resize(input_arr, [label_arr.shape[1], label_arr.shape[0]],
                                   interpolation = cv2.INTER_NEAREST)

        if crop_height > input_arr.shape[0] or crop_width > input_arr.shape[1]:
            raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (
                crop_height, crop_width, input_arr.shape[0], input_arr.shape[1]))

        if self.random_crop_ratio_low_boundary and self.random_crop_ratio_upper_boundary:
            if crop_width and crop_height:
                raise Exception('shall not assign both crop ratio and size')
            crop_ratio = random.uniform(self.random_crop_ratio_low_boundary, self.random_crop_ratio_upper_boundary)
            crop_height = math.floor(orig_height * crop_ratio)
            crop_width = math.floor(orig_width * crop_ratio)
            zoom_in = random.randint(0, 1) if self.zoom_in_and_out else 0

        if crop_width and crop_height:
            y, x = random.randint(0, orig_height - crop_height), random.randint(0, orig_width - crop_width)
            input_arr, label_arr = (input_arr[y:y + crop_height, x:x + crop_width, :],
                                    label_arr[y:y + crop_height, x:x + crop_width, :]) if len(
                label_arr.shape) == 3 else (
                input_arr[y:y + crop_height, x:x + crop_width, :],
                label_arr[y:y + crop_height, x:x + crop_width])
            if zoom_in:
                top = int((orig_height - crop_height) / 2)
                down = int((orig_height - crop_height + 1) / 2)
                left = int((orig_width - crop_width) / 2)
                right = int((orig_width - crop_width + 1) / 2)
                input_arr = cv2.copyMakeBorder(input_arr, top, down, left, right, cv2.BORDER_CONSTANT, None, [0, 0, 0])
                label_arr = cv2.copyMakeBorder(label_arr, top, down, left, right, cv2.BORDER_CONSTANT, None, [0, 0, 0])

        if self.height and self.width:
            input_arr = utils.resize_keep_aspectratio(input_arr, (
                self.width, self.height), False) if self.keep_aspect_ratio else cv2.resize(input_arr,
                                                                                           (self.width, self.height),
                                                                                           interpolation = cv2.INTER_LINEAR)
            label_arr = utils.resize_keep_aspectratio(label_arr, (
                self.width, self.height), True) if self.keep_aspect_ratio else cv2.resize(label_arr,
                                                                                          (self.width, self.height),
                                                                                          interpolation = cv2.INTER_NEAREST)

        if self.v_flip and random.randint(0, 1): input_arr, label_arr = cv2.flip(input_arr, 0), cv2.flip(label_arr, 0)

        if self.h_flip and random.randint(0, 1): input_arr, label_arr = cv2.flip(input_arr, 1), cv2.flip(label_arr, 1)

        if self.brightness: input_arr = cv2.LUT(input_arr, np.array(
            [k * brightness_gitter if k * brightness_gitter < 256 else 255 for k in range(256) for brightness_gitter
             in [random.uniform(1 - self.brightness, 1 + self.brightness)]]).astype(np.uint8))

        if self.rotation:
            angle = random.uniform(-self.rotation, self.rotation)
            M = cv2.getRotationMatrix2D((input_arr.shape[1] // 2, input_arr.shape[0] // 2), angle, scale = 1)
            input_arr, label_arr = cv2.warpAffine(input_arr, M, (input_arr.shape[1], input_arr.shape[0]),
                                                  flags = cv2.INTER_NEAREST), cv2.warpAffine(label_arr, M, (
                input_arr.shape[1], input_arr.shape[0]), flags = cv2.INTER_NEAREST)

        return input_arr, label_arr
