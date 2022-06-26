# -*- coding=utf-8 -*-
# @Time: 2021/10/16 14:01
# @Author: N
# @Software: PyCharm
import argparse
import functools
import math
import os
import random
import re
import sys
import threading
import time
import traceback

import cv2
import tensorboardX
import torch
import torch.distributed as dist
import tqdm
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms

import data
import utils
import models
from models.unet import UNet


def main():
    # define variables for the following
    # model_name = 'UACANet'
    # model_name = 'ResUNetPP'
    # model_name = 'TestNetWithInception'
    # model_name = 'Res&Attn&SE'
    model_name = 'SegCaps'
    tensorboard_path = f'./saved/runs/{model_name}/'
    best_state_path = f'./saved/{model_name}/best_model.pth'
    savepoint_dir = f'./saved/{model_name}/'
    # loading_path = f'./saved/{model_name}/breakpoint_360.pth'
    loading_path = best_state_path

    input_dir = "./dataset/CCF/train/img/"
    val_split_rate = 0
    batch_size = 9
    height = 512
    width = 512
    num_epochs = 0
    mnt_best = math.inf
    val_mnt_best = math.inf
    lr_schedule_interval = 20
    clip_grad = 1.0
    save_interval = 20
    val_interval = 50
    epochs_with_no_improve_and_no_scheduling = 0
    class2rgb_lut = [0, 255]
    scheduling_times = 0
    stop_scheduling_times = 5
    epochs_trained_this_time = 0
    # linear lr scaling rule: lr = base lr * batch size / 256
    # adam_kwargs = {'lr': 1.5e-3, 'betas': (0.9, 0.999), 'eps': 1e-8}
    adam_kwargs = {'lr': 1.5e-2, 'betas': (0.9, 0.999), 'eps': 1e-8}
    aug_dilation_rate = 5
    rotation = 20
    brightness = 0.2
    random_crop_ratio_low_boundary = 0.9
    random_crop_ratio_upper_boundary = 1
    init_lr_ratio = 0.5
    warm_up_epochs = 10
    num_workers = 0
    pin_memory = False
    zoom_in_and_out = False
    ddp = True
    h_flip = True
    v_flip = True
    keep_aspect_ratio = True
    mix_accuracy = False
    train_from_savepoint = False
    save_multi_breakpoint = False
    cover_mnt = False
    cover_val_mnt = False
    load_optimizer_state = False
    prefetching_data = False
    enable_EMA = False

    # set current work space
    try:
        os.chdir(os.path.dirname(__file__))
    except:
        print("failing to set work space as the file parent dir")

    # set random or not
    utils.set_seed(False)

    # torch.distributed.run distribute each process a unique identity called as local_rank
    parser = argparse.ArgumentParser(description = 'please run the file with -m torch.distributed.run')
    parser.add_argument("--local_rank", default = -1, type = int)
    local_rank = parser.parse_args().local_rank if ddp else 0

    # set the device of current process
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    # init the DDP backend
    if ddp:
        dist.init_process_group(backend = 'NCCL' if not sys.platform.startswith('win') else 'GLOO')

    # check savepoint dir
    if local_rank == 0 and not os.path.exists(savepoint_dir):
        os.makedirs(savepoint_dir)
    if ddp:
        torch.distributed.barrier()

    # dataset
    input_path_list = []
    for file in os.listdir(input_dir):
        if os.path.splitext(file)[1] != '.json':
            input_path_list.append(input_dir + file)
    input_path2label_path = lambda path: re.sub('img', 'mask', path) if os.path.splitext(path)[
                                                                            1] == '.png' else None
    aug_method = utils.img_augment(height = height, width = width, h_flip = h_flip, v_flip = v_flip,
                                   random_crop_ratio_low_boundary = random_crop_ratio_low_boundary,
                                   random_crop_ratio_upper_boundary = random_crop_ratio_upper_boundary,
                                   brightness = brightness, rotation = rotation, keep_aspect_ratio = keep_aspect_ratio,
                                   zoom_in_and_out = zoom_in_and_out)
    input_open_method = lambda input_path: Image.open(input_path)
    # input_open_method = lambda input_path: Image.open(input_path).convert('L')
    if keep_aspect_ratio:
        resize_method = functools.partial(utils.resize_keep_aspectratio, dsize = (height, width))
    else:
        resize_method = functools.partial(cv2.resize, dsize = (width, height), interpolation = cv2.INTER_NEAREST)
    mean, std = utils.get_mean_and_std(input_path_list, input_open_method, size_with_max_3dim = (height, width),
                                       resize_method = resize_method)

    transformer4label_arr = lambda augmented_label_arr: torch.FloatTensor(
        utils.rgb2class(augmented_label_arr, class2rgb_lut)).squeeze(dim = 0)

    transformer4input_arr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = data.BaseDataset(input_path_list, input_path2label_path, input_open_method,
                                 Image.open, aug_method,
                                 transformer4input_arr, transformer4label_arr,
                                 aug_dilation_rate = aug_dilation_rate)

    # split train set to train set and validation set
    val_size = math.floor(len(train_set) * val_split_rate)
    train_size = len(train_set) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, lengths = (train_size, val_size))

    # bcz the dataloader is to load data to the gpu indicated by the local_rank
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if ddp else None
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers,
                              pin_memory = pin_memory, sampler = train_sampler)

    train_loader = data.DataPrefetcher(train_loader, device) if prefetching_data else train_loader
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if ddp else None
    val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = pin_memory,
                            sampler = val_sampler)
    val_loader = data.DataPrefetcher(val_loader, device) if prefetching_data else val_loader

    # warp model with DDP, and load model to the gpu indicated by the local_rank
    # model = UNet(1, 2).to(device)
    # model = models.TestNet(3, 1).to(device)
    model = models.SegCaps().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) if ddp else model
    model = DDP(model, device_ids = [local_rank], find_unused_parameters = True) if ddp else model
    utils.initialize_weights(model)
    if enable_EMA:
        ema = utils.EMA(model.module)
        ema.register()
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # criterion = torch.nn.NLLLoss().to(device)
    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = utils.bce_iou_loss
    criterion = utils.margin_loss
    optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)

    # schedule the lr to half when not have loss desc after a indicated interval
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda
        epochs_or_times: 0.5 ** epochs_or_times if epochs_or_times < stop_scheduling_times else init_lr_ratio + (
            1 - init_lr_ratio) * (epochs_or_times - stop_scheduling_times) / warm_up_epochs)

    # Scaler for auto mix precision calculation
    scaler = GradScaler()

    # tensorboard writer
    writer = tensorboardX.SummaryWriter(tensorboard_path)

    # load state
    if train_from_savepoint:
        try:
            best_state = torch.load(loading_path, map_location = torch.device('cpu'))
            if ddp:
                model.module.load_state_dict(best_state['state_dict'])
            else:
                model.load_state_dict(best_state['state_dict'])
            if enable_EMA and best_state.get('ema_state_dict', None):
                ema.load_state_dict(best_state['ema_state_dict'])
                if local_rank == 0:
                    print('ema para loading success')
            num_epochs = best_state['epoch']
            val_mnt_best = best_state['val_monitor_best'] if cover_val_mnt else val_mnt_best
            if load_optimizer_state:
                optimizer.load_state_dict(best_state['optimizer'])
            if cover_mnt:
                mnt_best = best_state['monitor_best']
                scheduling_times = best_state.get('scheduling_times', 0)
            if local_rank == 0:
                print(
                    f'loading success, path to load: "{loading_path}", savepoint mnt: {best_state["monitor_best"]:.5f}, savepoint scheduling times: {best_state["scheduling_times"]}, mnt_best: {mnt_best:.5f}, scheduling_times: {scheduling_times}')
        except Exception as e:
            traceback.print_exc()
            if local_rank == 0:
                print('loading fail')
            best_state = None
    else:
        if local_rank == 0:
            print('loading is canceled manually')
        best_state = None

    get_state = lambda: {
        'arch': type(model).__name__,
        'epoch': num_epochs,
        'state_dict': model.module.state_dict() if ddp else model.state_dict(),
        'ema_state_dict': ema.state_dict() if enable_EMA else None,
        'optimizer': optimizer.state_dict(),
        'monitor_best': mnt_best,
        'val_monitor_best': val_mnt_best,
        'scheduling_times': scheduling_times
    }

    while True:
        num_epochs += 1
        epochs_trained_this_time += 1
        if epochs_trained_this_time <= warm_up_epochs:
            lr_scheduler.step(epochs_trained_this_time + stop_scheduling_times)
        loss_hist = []
        if local_rank == 0:
            tq = tqdm.tqdm(total = len(train_loader), ncols = 0, desc = f"train_epoch_{num_epochs:d}")
        for x, label in train_loader:
            # x, label = (x, label) if prefetching_data else (x.to(device), label.to(device))
            x, label = (x, label) if prefetching_data else (x.to(device), label.long().to(device))
            if mix_accuracy:
                with torch.cuda.amp.autocast():
                    logit = model(x)
                    loss = criterion(logit, label)

                    # # 4 UACANet
                    # loss = model(x, label, criterion)['loss']b

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                # for each backward params can only be unscaled one time
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                if enable_EMA:
                    ema.update()
                scaler.update()
            else:
                # logit = model(x)
                logit = model(x)
                loss = criterion(logit,label)

                # # 4 UACANet
                # loss = model(x, label, criterion)['loss']

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                if enable_EMA:
                    ema.update()
            loss_scalar = loss.item()
            loss_hist.append(loss_scalar)
            if local_rank == 0:
                tq.update(1)
                tq.set_postfix({'loss': loss_scalar})

        loss_sum = torch.FloatTensor([sum(loss_hist)]).to(device)
        num_batches = torch.LongTensor([len(loss_hist)]).to(device)
        dist.reduce(loss_sum, dst = 0, op = dist.ReduceOp.SUM, async_op = True)
        dist.reduce(num_batches, dst = 0, op = dist.ReduceOp.SUM, async_op = True)
        epoch_loss = (loss_sum / num_batches).to(device)
        dist.broadcast(epoch_loss, 0)
        epoch_loss = epoch_loss.item()

        if local_rank == 0:
            tq.set_postfix({'epoch_loss': epoch_loss})
            writer.add_scalar('epoch_loss', epoch_loss, global_step = num_epochs)
            tq.close()

        if epoch_loss < mnt_best:
            mnt_best = epoch_loss
            if local_rank == 0:
                print(f'updating mnt_best: {mnt_best}')
                best_state = get_state()
            epochs_with_no_improve_and_no_scheduling = 0
        else:
            epochs_with_no_improve_and_no_scheduling += 1
            if local_rank == 0 and scheduling_times >= stop_scheduling_times:
                if best_state:
                    torch.save(best_state, best_state_path)
                torch.save(get_state(), savepoint_dir + (
                    f'breakpoint_{num_epochs}.pth' if save_multi_breakpoint else 'breakpoint.pth'))
                raise Exception('learning stop')
            if epochs_with_no_improve_and_no_scheduling >= lr_schedule_interval:
                scheduling_times += 1
                lr_scheduler.step(scheduling_times)
                clip_grad /= 4
                lr_schedule_interval *= 2
                if local_rank == 0:
                    print(f'scheduling learning rate for the {scheduling_times}th time')
                epochs_with_no_improve_and_no_scheduling = 0

        if local_rank == 0 and epochs_trained_this_time % save_interval == 0:
            if best_state:
                torch.save(best_state, best_state_path)
            torch.save(get_state(),
                       savepoint_dir + (f'breakpoint_{num_epochs}.pth' if save_multi_breakpoint else 'breakpoint.pth'))

        if val_split_rate and epochs_trained_this_time % val_interval == 0:
            if enable_EMA:
                ema.apply_shadow()
            model.eval()
            with torch.no_grad():
                loss_hist = []
                if local_rank == 0:
                    tq = tqdm.tqdm(total = len(val_loader), ncols = 0,
                                   desc = f"val_{epochs_trained_this_time // val_interval:d}th")
                for x, label in val_loader:
                    x, label = (x, label) if prefetching_data else (x.to(device), label.to(device))
                    logit = model(x)
                    loss = criterion(logit, label)

                    # # 4 UACANet
                    # loss = model(x, label, criterion)['loss']

                    loss_scalar = loss.item()
                    loss_hist.append(loss_scalar)
                    if local_rank == 0:
                        tq.update(1)
                        tq.set_postfix({'loss': loss_scalar})

                val_loss_sum = torch.FloatTensor([sum(loss_hist)]).to(device)
                num_batches = torch.LongTensor([len(loss_hist)]).to(device)
                dist.reduce(val_loss_sum, dst = 0, op = dist.ReduceOp.SUM, async_op = True)
                dist.reduce(num_batches, dst = 0, op = dist.ReduceOp.SUM, async_op = True)
                val_loss_this_time = (val_loss_sum / num_batches).to(device)
                val_loss_this_time = val_loss_this_time.item()
                if local_rank == 0:
                    tq.set_postfix({'val_mean_loss': val_loss_this_time})
                    writer.add_scalar('val_mean_loss', val_loss_this_time,
                                      global_step = epochs_trained_this_time // val_interval)
                    tq.close()

            model.train()
            if enable_EMA:
                ema.restore()

            if local_rank == 0 and val_loss_this_time < val_mnt_best:
                torch.save(get_state(), savepoint_dir + 'val_best.pth')
                val_mnt_best = val_loss_this_time
                print(f'updating val_mnt_best: {val_mnt_best}')

            dist.barrier()


if __name__ == '__main__':
    # to close all processes with trigger ctrl+c
    try:
        threading.Thread(target = main).start()
    except Exception as e:
        traceback.print_exc()
        os.system(f"kill $(ps aux | grep '{os.path.split(__file__)[-1]}' | grep -v grep | awk '{{print $2}}')")
