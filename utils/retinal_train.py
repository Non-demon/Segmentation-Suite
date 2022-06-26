# -*- coding=utf-8 -*-
# @Time: 2021/10/4 15:03
# @Author: N
# @Software: PyCharm
import argparse
import io
import math
import os
import random
import re
import sys
import threading
from typing import Tuple, Any

import PIL.Image
import cv2
import data
import models
import numpy as np
import requests
import tensorboardX
import torch
import torch.distributed as dist
import torchvision.transforms
import tqdm
import utils
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

best_state = None


def load_model(path: str, device: torch.device = None) -> nn.Module:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()

        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location = device)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location = device)


class BasicVAE(nn.Module):

    def get_codebook_indices(self, images):
        raise NotImplementedError()

    def decode(self, img_seq):
        raise NotImplementedError()

    def get_codebook_probs(self, img_seq):
        raise NotImplementedError()

    def get_image_tokens_size(self):
        pass

    def get_image_size(self):
        pass


class Dalle_VAE(BasicVAE):
    def __init__(self, image_size):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.image_size = image_size

    def load_model(self, model_dir, device):
        self.encoder = load_model(os.path.join(model_dir, "encoder.pkl"), device)
        self.decoder = load_model(os.path.join(model_dir, "decoder.pkl"), device)

    def decode(self, img_seq):
        bsz = img_seq.size()[0]
        img_seq = img_seq.view(bsz, self.image_size // 8, self.image_size // 8)
        z = F.one_hot(img_seq, num_classes = self.encoder.vocab_size).permute(0, 3, 1, 2).float()
        return self.decoder(z).float()

    def get_codebook_indices(self, images):
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, dim = 1)

    def get_codebook_probs(self, images):
        z_logits = self.encoder(images)
        return nn.Softmax(dim = 1)(z_logits)

    def forward(self, img_seq_prob, no_process = False):
        if no_process:
            return self.decoder(img_seq_prob.float()).float()
        else:
            bsz, seq_len, num_class = img_seq_prob.size()
            z = img_seq_prob.view(bsz, self.image_size // 8, self.image_size // 8, self.encoder.vocab_size)
            return self.decoder(z.permute(0, 3, 1, 2).float()).float()


def get_dalle_vae(weight_path, image_size, device):
    vae = Dalle_VAE(image_size)
    vae.load_model(model_dir = weight_path, device = device)
    return vae


class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches = 4, max_num_patches = None,
            min_aspect = 0.3, max_aspect = None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape = self.get_shape(), dtype = int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class BaseDataset(Dataset):
    def __init__(self, input_path_list: list, get_label_path: callable, input_open_method: callable,
                 label_open_method: callable, aug_method: callable, transformer4input_arr: callable,
                 transformer4label_arr: callable, aug_dilation_rate = 1):
        # class2rgb_lut[class_index]=[r,g,b]
        self.input_path_list = input_path_list
        self.get_label_path = get_label_path
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
        # ngp is the acronym of "new game plus"
        input_path = self.input_path_list[index]
        input_arr, label_arr = self.aug_method(np.array(self.input_open_method(input_path)),
                                               np.array(self.label_open_method(self.get_label_path(input_path))),
                                               is_ngp)
        return self.transformer4input_arr(input_arr), self.transformer4label_arr(label_arr)


def main():
    # set current work space
    try:
        os.chdir(os.path.dirname(__file__))
    except:
        print("failing to set work space as the file parent dir")

    # set parameters of base variable type for following steps
    split_rate = 0.2
    batch_size = 4
    weight_path = './saved/vae_weight/'
    image_size = (224, 224)
    img_size4vae = (112, 112)
    clip_grad = 3.0
    window_size = (14, 14)
    num_masking_patches = 75
    max_mask_patches_per_block = None
    min_mask_patches_per_block = 16
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
    aug_method = utils.img_augment(height = 224, width = 224, h_flip = True, v_flip = True,
                                   brightness = 0.2, rotation = 0.2, aug_in_ngp = True)

    class Transformer4InputArr:
        def __init__(self, mean = None, std = None, input_path_list = None, input_open_method = cv2.imread):

            if not mean or not std:
                if input_path_list:
                    mean, std = utils.get_mean_and_std(input_path_list, input_open_method)
                else:
                    raise Exception("require std and mean or input_path_list but get non")
            normalize = torchvision.transforms.Normalize(mean, std)
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

            self.masked_position_generator = MaskingGenerator(
                input_size = window_size, num_masking_patches = num_masking_patches,
                max_num_patches = max_mask_patches_per_block,
                min_num_patches = min_mask_patches_per_block)

        def __call__(self, input_arr):
            arr4vae = cv2.pyrDown(input_arr, dstsize = img_size4vae)
            return self.transformer4img(input_arr), self.transformer4visual_token(
                arr4vae), self.masked_position_generator()

    def constr_transformer4label_arr(class2rgb_lut):
        def transformer4label_arr(label_arr):
            return torch.LongTensor(utils.rgb2class(label_arr, class2rgb_lut))

        return transformer4label_arr

    train_set = BaseDataset(input_path_list = input_path_list,
                            get_label_path = lambda input_path: re.sub("_training.tif", "_manual1.gif",
                                                                       re.sub("/images/", "/1st_manual/", input_path)),
                            transformer4input_arr = Transformer4InputArr(input_path_list = input_path_list),
                            transformer4label_arr = constr_transformer4label_arr([0, 255]), aug_method = aug_method,
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
    model = models.beit_base_patch16_224_8k_vocab(drop_path_rate = 0.1, num_classes = -1, init_values = 0.1).to(device)
    model = DDP(model, device_ids = [local_rank])
    map_location = lambda storage, loc: storage.cuda(device)
    d_vae = get_dalle_vae(weight_path = weight_path, image_size = img_size4vae, device = map_location)
    d_vae = DDP(d_vae, device_ids = [local_rank])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), **{'lr': 1.5e-3, 'betas': (0.9, 0.999), 'eps': 1e-8})

    num_epochs = 0
    mnt_best = math.inf
    scaler = GradScaler()
    writer = tensorboardX.SummaryWriter(tensorboard_path)

    def save(presave_best = False):
        state = {
            'arch': type(model).__name__,
            'epoch': num_epochs,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'monitor_best': mnt_best,
        }
        global best_state
        if presave_best:
            best_state = state
            return
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        filename = os.path.join(checkpoint_dir, f'pretrain-epoch{num_epochs}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)
        if best_state and num_epochs - best_state['epoch'] < save_interval:
            filename = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(best_state, filename)
            print("Saving current best: best_model.pth")

    while True:
        num_epochs += 1
        if local_rank == 0:
            proceeding_bar = tqdm.tqdm(total = len(train_loader), ncols = 0, desc = f"train_epoch_{num_epochs:d}")
            loss_hist = []
        for (img4beit, img4vae, bool_masked_pos), _ in train_loader:
            # d_vae doesnt need normalization but need a preprocess method named map pixels
            with torch.no_grad():
                input_ids = d_vae.module.get_codebook_indices(img4vae).flatten(1)
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                label = input_ids[bool_masked_pos]
            with torch.cuda.amp.autocast():
                output = model(img4beit, bool_masked_pos)
                loss = criterion(output, label)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            # for each backward params can only be unscaled one time
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            if local_rank == 0:
                proceeding_bar.update(1)
                loss_scalar = loss.item()
                loss_hist.append(loss_scalar)
                proceeding_bar.set_postfix({'loss': loss_scalar})
        if local_rank == 0:
            batch_loss = sum(loss_hist) / len(loss_hist)
            proceeding_bar.set_postfix({'batch_loss': batch_loss})
            proceeding_bar.close()
            writer.add_scalar('batch_loss', batch_loss, global_step = num_epochs)
            if batch_loss < mnt_best:
                mnt_best = batch_loss
                save(True)
            if num_epochs % save_interval == 0:
                save()


if __name__ == '__main__':
    # to close all processes with trigger ctrl+c
    try:
        threading.Thread(target = main).start()
    except:
        os.system(f"kill $(ps aux | grep '{os.path.split(__file__)[-1]}' | grep -v grep | awk '{{print $2}}')")
