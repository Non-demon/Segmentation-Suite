# -*- coding=utf-8 -*-
# @Time: 2021/7/10 11:27
# @Author: N
# @Software: PyCharm

import os
import json
import random
import threading
import time

import numpy as np

from utils import Image
import cv2
import utils
import models
import torch
import data
import argparse
import grad_cam
from matplotlib import pyplot as plt
from grad_cam.utils.image import show_cam_on_image

def get_trainer(config):
    steps_lut=['train_loader','model','criterion','trainer']
    other_kwargs_lut=['','num_classes=train_loader.dataset.num_classes','','''model=model,criterion=criterion,
                      train_loader=train_loader,val_loader=train_loader.get_val_loader(),config=config,
                      num_classes=train_loader.dataset.num_classes''']
    for i in range(len(steps_lut)):
        exec(f'''{steps_lut[i]}=getattr({config["module_dict"][steps_lut[i]]},
        "{config[steps_lut[i]]["type"]}")(**config[steps_lut[{i}]]["args"],{other_kwargs_lut[i]})''')
    return locals()['trainer'],locals()['train_loader'].dataset.normalize

def get_tensorboard():
    os.system(f'tensorboard --logdir="{trainer.tensorboard_path}"')

if __name__=='__main__':
    utils.set_seed(False)
    # set current work directory as the directory of this file
    os.chdir(os.path.dirname(__file__))
    # argparse
    parser=argparse.ArgumentParser(description='whether to test only')
    parser.add_argument('-test','--test',action='store_const',const=True,default=False,metavar='',
                        help='type in it for testing only')
    parser.add_argument('-tensorboard','--tensorboard',action='store_const',const=True,default=False,
                        metavar='',help='type in it to close tensorboard service')
    args=parser.parse_args()
    # load config and get trainer
    config=json.load(open('config.json','r'))
    trainer,normalize=get_trainer(config)
    # loading tensorboard server
    if args.tensorboard:
        threading.Thread(target=get_tensorboard).start()
        time.sleep(3)
    trainer.model.eval()
    cam=pytorch_grad_cam.GradCAM(trainer.model,target_layer=trainer.model.master_branch,use_cuda=True)
    img_array=np.array(Image.open(os.path.join('CamVid','train','0001TP_007920.png')))
    down_img_array=cv2.resize(img_array,(img_array.shape[1]//2,img_array.shape[0]//2))/255
    input_tensor=normalize(torch.FloatTensor(down_img_array.transpose((2,0,1)))).view(1,3,360,480)
    pred=trainer.model(input_tensor.to(trainer.device)).view(-1,360,480).argmax(dim=0).cpu()
    pred_rgb=utils.class2rgb(pred,utils.get_class2name_lut_and_class2rgb_lut('CamVid')[1]).permute((1,2,0)).numpy()
    # please check the get_loss method in class base_cam first
    grayscale_cam=cam(input_tensor=input_tensor,target_category=4)
    visualization=show_cam_on_image(down_img_array,grayscale_cam[0])
    plt.figure()
    plt.imshow(visualization)
    plt.figure()
    print(pred_rgb.shape)
    plt.imshow(pred_rgb)
    plt.show()
