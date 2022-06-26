# -*- coding=utf-8 -*-
# @Time: 2021/7/10 11:27
# @Author: N
# @Software: PyCharm

import argparse
import json
import os
import threading

import data
import models
import utils


def get_trainer(config):
    train_loader = getattr(data, "my_dataloader")(**config['train_loader']["args"], )
    model = getattr(models, "DualResNet_imagenet")(**config['model']["args"],
                                                   num_classes = train_loader.dataset.num_classes)
    criterion = getattr(utils, "CrossEntropy")(**config['criterion']["args"], )
    trainer = getattr(utils, "trainer")(**config['trainer']["args"], model = model, criterion = criterion,
                                        train_loader = train_loader, val_loader = train_loader.get_val_loader(),
                                        config = config, num_classes = train_loader.dataset.num_classes)

    return locals()['trainer']


if __name__ == '__main__':
    # set current work directory as the directory of this file
    try:
        os.chdir(os.path.dirname(__file__))
    except:
        pass
    # argparse
    parser = argparse.ArgumentParser(description = 'typing in -h for help')
    parser.add_argument('-test', '--test', action = 'store_const', const = True, default = False, metavar = '',
                        help = 'type in it for testing only')
    parser.add_argument('-tensorboard', '--tensorboard', action = 'store_const', const = True, default = False,
                        metavar = '', help = 'type in it to close tensorboard service')
    args = parser.parse_args()
    # load config
    config = json.load(open('config.json', 'r'))
    # seed
    utils.set_seed(False)
    # get trainer
    trainer = get_trainer(config)
    # loading tensorboard server
    if args.tensorboard:
        threading.Thread(target = os.system,
                         args = [f'tensorboard --logdir="{trainer.tensorboard_path}" --port 9999']).start()
    # do training or testing
    if args.test:
        trainer.validate()
    else:
        trainer.train()
