# -*- coding=utf-8 -*-
# @Time: 2021/10/16 14:01
# @Author: N
# @Software: PyCharm
import csv
import functools
import os
import threading
import time

import cv2
import tensorboardX
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import data
import utils
import models


def main():
    # define variables for the following
    # model_name = 'UACANet'
    # model_name = 'ResUNetPP'
    # model_name = 'TestNetWithInception'
    model_name = 'Res&Attn&SE'
    tensorboard_path = f'./saved/runs/{model_name}/'
    best_state_path = f'./saved/{model_name}/best_model.pth'
    savepoint_dir = f'./saved/{model_name}/'
    # loading_path = f'./saved/{model_name}/breakpoint_360.pth'
    # loading_path = best_state_path
    loading_path = f'./saved/{model_name}/val_best.pth'

    train_dir = "./dataset/CCF/train/img/"
    test_dir = "./dataset/CCF/test/img/"
    batch_size = 2
    class2rgb_lut = [0, 255]
    dst_size = (512, 512)  # (height,width)
    keep_aspect_ratio = True
    enable_ema = False

    # set current work space
    try:
        os.chdir(os.path.dirname(__file__))
    except:
        print("failing to set work space as the file parent dir")

    # set random or not
    utils.set_seed(False)

    # set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device, 'is in use')

    # dataset
    input_path_list = []
    for file in os.listdir(test_dir):
        if os.path.splitext(file)[1] != '.json':
            input_path_list.append(test_dir + file)
    input_path2label_path = lambda path: None
    label_open_method = lambda _: None
    aug_method = lambda input_arr, _: (
        utils.resize_keep_aspectratio(input_arr, dst_size, False) if keep_aspect_ratio else cv2.resize(input_arr,
                                                                                                       dst_size,
                                                                                                       interpolation = cv2.INTER_LINEAR),
        None)
    # input_open_method = lambda input_path: Image.open(input_path).convert('L')
    input_open_method = lambda input_path: Image.open(input_path)
    mean, std = utils.get_mean_and_std_from_json(train_dir)
    transformer4label_arr = lambda _: torch.LongTensor([0])
    transformer4input_arr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dataset = data.Dataset4GeneratingPredictsWithKeepingAspectRatioResize(
        functools.partial(utils.get_label_boundary, dst_size = dst_size), input_path_list,
        input_path2label_path, input_open_method, label_open_method, aug_method, transformer4input_arr,
        transformer4label_arr) if keep_aspect_ratio else data.Dataset4GeneratingPredicts(input_path_list,
                                                                                         input_path2label_path,
                                                                                         input_open_method,
                                                                                         label_open_method,
                                                                                         aug_method,
                                                                                         transformer4input_arr,
                                                                                         transformer4label_arr)

    # bcz the dataloader is to load data to the gpu indicated by the local_rank
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 2)

    # warp model with DDP, and load model to the gpu indicated by the local_rank
    # model = UNet(1, 2).to(device)
    # model = models.UACANet(pretrained = False).to(device)
    model = models.TestNet(3, 1).to(device)
    ema = utils.EMA(model)
    ema.register()

    # tensorboard writer
    # writer = tensorboardX.SummaryWriter(tensorboard_path)

    # load state
    try:
        best_state = torch.load(loading_path, map_location = torch.device('cpu'))
        model.load_state_dict(best_state['state_dict'])
        ema.load_state_dict(best_state['ema_state_dict'])
        num_epochs = best_state['epoch']
        mnt_best = best_state['monitor_best']
        print(f'loading success, mnt_best: {mnt_best}, epoch: {num_epochs}')
    except Exception as e:
        raise e

    if enable_ema:
        ema.apply_shadow()
        model.to(device)
    model.eval()

    tq = tqdm.tqdm(total = len(train_loader), ncols = 0, desc = f"train_epoch_{num_epochs:d}")
    with open(os.path.join('.', 'submit.csv'), 'w+', encoding = 'utf-8') as csv_fd:
        writer = csv.writer(csv_fd)
        writer.writerow(['filename', 'w h', 'rle编码'])
        with torch.no_grad():
            for zip, label in train_loader:
                if keep_aspect_ratio:
                    x, paths, original_sizes, boundaries = zip
                else:
                    x, paths, original_sizes = zip
                x, label = x.to(device), label.to(device)
                output = model(x)
                # output = model(x, x.argmax(1).unsqueeze(1).float(), torch.nn.BCEWithLogitsLoss())['pred']
                output = torch.cat([(1 - output), output], 1)
                tq.update(1)
                # output shape shall be (b,2,h,w)
                for i in range(len(output)):
                    # h_up_boundary, h_down_boundary, w_left_boundary, w_right_boundary
                    if keep_aspect_ratio:
                        hub, hdb, wlb, wrb = boundaries[i]
                        img_tensor = output[i][:, hub:hdb, wlb:wrb].unsqueeze(0)
                    else:
                        img_tensor = output[i].unsqueeze(0)
                    original_size = original_sizes[i].tolist()
                    img_tensor = torch.nn.functional.interpolate(img_tensor, original_size,
                                                                 mode = 'nearest').squeeze().argmax(0)
                    path = paths[i]
                    rle = utils.mask2rle(img_tensor.cpu().numpy())
                    row = [path, f'{original_size[1]} {original_size[0]}', rle]
                    writer.writerow(row)
                    # utils.show_img(utils.class2rgb(img_tensor.cpu(), [[ele] * 3 for ele in class2rgb_lut]))
        tq.close()


if __name__ == '__main__':
    # to close all processes with trigger ctrl+c
    try:
        threading.Thread(target = main).start()
    except:
        os.system(f"kill $(ps aux | grep '{os.path.split(__file__)[-1]}' | grep -v grep | awk '{{print $2}}')")
