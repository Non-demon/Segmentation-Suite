# -*- coding=utf-8 -*-
# @Time: 2021/10/12 10:25
# @Author: N
# @Software: PyCharm
import io
import os

import requests
import torch
from torch import nn
from torch.nn import functional as F


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
