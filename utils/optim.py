# -*- coding=utf-8 -*-
# @Time: 2021/11/30 15:04
# @Author: N
# @Software: PyCharm

import torch
from torch.optim.optimizer import Optimizer

class LARSV2(Optimizer):
    def __init__(self, params, lr, momentum=0, weight_decay=0, trust_coef=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coef=trust_coef)
        super(LARSV2, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(LARSV2, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coef = group['trust_coef']
            global_lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p_norm = torch.norm(p.data)
                d_p_norm = torch.norm(d_p.data)
                if weight_decay != 0:
                    d_p_norm.add_(weight_decay, p_norm)
                local_lr = torch.div(p_norm, d_p_norm).mul_(trust_coef*global_lr)
                # local_lr.mul_(global_lr)
                # weight decay
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                # d_p.mul_(local_lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.data.add_(-local_lr, d_p)