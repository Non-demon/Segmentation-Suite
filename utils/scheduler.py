# -*- coding=utf-8 -*-
# @Time: 2021/7/29 15:49
# @Author: N
# @Software: PyCharm

import torch.optim.lr_scheduler

class lr_scheduler:
    def __init__(self,optimizer,train_epochs,lr_scheduler_name:str,warmup_epochs=0,last_epochs=-1):
        self.d=locals()
    def get_lr_scheduler(self):
        print(getattr(self,self.d['method'])(1))
        return torch.optim.lr_scheduler.LambdaLR(self.d['optimizer'],getattr(self,self.d['lr_scheduler_name']),self.d['last_epochs'])
    def starting_fast(self,epochs):
        if epochs < self.d['warmup_epochs']:
            return 1
        else:
            return 1-(epochs-self.d['warmup_epochs'])/(self.d['train_epochs']-self.d['warmup_epochs'])*0.6