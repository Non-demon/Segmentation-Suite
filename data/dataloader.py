# -*- coding=utf-8 -*-
# @Time: 2021/8/6 17:06
# @Author: N
# @Software: PyCharm

import data
import sys
import multiprocessing
import utils

class my_dataloader(data.BaseDataLoader):
    def __init__(self,dir_name,dilation_rate,train_val_or_test,batch_size,shuffle,val_split=.0,
                 num_worker=0 if sys.platform.startswith('win') else multiprocessing.cpu_count(),pin_memory=True,
                 drop_last=True,normalization=False,mean=None,std=None,height=0,width=0,crop_height=0,crop_width=0,h_flip=False,
                 v_flip=False,brightness=0,rotation=0,aug_in_ngp=False,mk_conf=False):
        if mk_conf: self.config_dict=locals().copy()
        for arg in locals().values():
            if isinstance(arg,int):
                if arg<0: raise ValueError()
        if (train_val_or_test!='train' and val_split>0): raise ValueError()
        super(my_dataloader,self).__init__(dataset=data.MyDataset(dir_name, utils.img_augment(height, width, crop_height,
                                                                                              crop_width, h_flip, v_flip,
                                                                                              brightness, rotation,
                                                                                              aug_in_ngp),
                                                                  dilation_rate, normalization, mean, std),batch_size=batch_size,shuffle=shuffle,
                                           num_workers=num_worker,pin_memory=pin_memory,drop_last=drop_last,val_split=val_split)

    def get_config_dict(self):
        for v in ['self','mk_conf','__class__']:
            try:
                del self.config_dict[v]
                print(v)
            except: pass
        return self.config_dict