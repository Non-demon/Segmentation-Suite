# -*- coding=utf-8 -*-
# @Time: 2021/8/6 16:44
# @Author: N
# @Software: PyCharm

import numpy as np
import torch
from torch.backends import cudnn


def set_seed(is_seed=False,seed=42069):
    torch.backends.cudnn.deterministic=True if is_seed else False
    torch.backends.cudnn.benchmark=False if is_seed else True
    if is_seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)