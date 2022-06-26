# -*- coding=utf-8 -*-
# @Time: 2021/10/4 15:03
# @Author: N
# @Software: PyCharm

import argparse
import math
import os
import re
import sys
import threading

import PIL.Image
import cv2
import tensorboardX
import torch
import torch.distributed as dist
import torchvision.transforms
import tqdm
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import data
import models
import utils

best_state = None


def main():
    # set current work space
    try:
        os.chdir(os.path.dirname(__file__))
    except:
        print("failing to set work space as the file parent dir")

    # set parameters of base variable type for following steps
    split_rate = 0.2
    batch_size = 4
    clip_grad = 3.0
    tensorboard_path = './saved/runs/Beit'
    checkpoint_dir = './saved/BEiT/'
    input_dir = "./dataset/DRIVE/training/images/"
    save_interval = 500

    # set random or not
    utils.set_seed(False)

    # torch.distributed.run distribute each process a unique identity called as local_rank
    parser = argparse.ArgumentParser(description = 'please run the file with -m torch.distributed.run')
    parser.add_argument("--local_rank", default = -1, type = int)
    local_rank = parser.parse_args().local_rank

    # set the device of current process
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    # init the DDP backend
    dist.init_process_group(backend = 'NCCL' if not sys.platform.startswith('win') else 'GLOO')

    input_path_list = []
    for input_path in os.listdir(input_dir):
        if os.path.splitext(input_path)[1] == '.tif':
            input_path_list.append(input_dir + input_path)
    aug_method = utils.img_augment(height = 584, width = 565, h_flip = True, v_flip = True,
                                   brightness = 0.2, rotation = 0.2, aug_in_ngp = True)

    train_set = data.BaseDataset(input_path_list = input_path_list,
                                 input_path2label_path = lambda input_path: re.sub("_training.tif", "_manual1.gif",
                                                                                   re.sub("/images/", "/1st_manual/",
                                                                                   input_path)),
                                 transformer4input_arr = data.TransformerOfInputArr4BeitTraining(
                                     input_path_list = input_path_list),
                                 transformer4label_arr = data.transformer4label_arr_of_drive([0, 255]),
                                 aug_method = aug_method,
                                 input_open_method = cv2.imread, label_open_method = PIL.Image.open,
                                 aug_dilation_rate = 3)

    # bcz the sampler in dataloader is indicated by DDP so can only split train set to train set and validation set
    val_size = math.floor(len(train_set) * split_rate)
    train_size = len(train_set) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, lengths = (train_size, val_size))

    # bcz the dataloader is to load data to the gpu indicated by the local_rank
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 2,
                              sampler = train_sampler)
    train_loader = data.DataPrefetcher(train_loader, device)

    # warp model with DDP, and load model to the gpu indicated by the local_rank
    model = models.UperNet(2).to(device)
    # model.load_pretrained_beit('./saved/BEiT/best_model.pth')
    model = DDP(model, device_ids = [local_rank])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), **{'lr': 1.5e-3, 'betas': (0.9, 0.999), 'eps': 1e-8})

    num_epochs = 0
    mnt_best = math.inf
    scaler = GradScaler()
    writer = tensorboardX.SummaryWriter(tensorboard_path)

    state=torch.load('./saved/BEiT/best_model.pth')
    model.module.load_state_dict(state['state_dict'])

    for img,_ in val_dataset:
    # todo: visualization method in utils


if __name__ == '__main__':
    # to close all processes with trigger ctrl+c
    try:
        threading.Thread(target = main).start()
    except:
        os.system(f"kill $(ps aux | grep '{os.path.split(__file__)[-1]}' | grep -v grep | awk '{{print $2}}')")
