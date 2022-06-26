# -*- coding=utf-8 -*-
# @Time: 2021/7/10 11:27
# @Author: N
# @Software: PyCharm

import json
import utils
import models
import torch
import data


def generate_config():
    keys = ['train_loader', 'model', 'criterion', 'trainer']
    objs = ['train_loader', 'model', 'criterion', 'trainer_obj']
    modules = ['data', 'models', 'torch.nn', 'utils']
    config = {}
    for i in range(len(keys)):
        try:
            exec(f'config["{keys[i]}"]={{"type":{objs[i]}.__class__.__name__,"args":{objs[i]}.get_config_dict()}}')
        except:
            exec(f'config["{keys[i]}"]={{"type":{objs[i]}.__class__.__name__,"args":{objs[i]}_args}}')
    config['module_dict'] = {keys[i]: modules[i] for i in range(len(keys))}
    json.dump(config, open('PSPNet_cfg.json', 'w'))
    # print(config)
    # for k,v in {**train_loader_args,**model_args,**trainer_obj_args}.items():
    #     if not isinstance(v,str): print(k,v)
    # print(trainer_obj_args['device'])


if __name__ == '__main__':
    utils.set_seed()
    criterion_args = {}
    optimizer_args = {'lr': 1.5e-3, 'betas': (0.9, 0.999), 'eps': 1e-8}
    train_loader = data.my_dataloader(dir_name = 'CamVid', dilation_rate = 1, train_val_or_test = 'train',
                                      batch_size = 8,
                                      shuffle = True,
                                      val_split = 0.3, normalization = True, crop_width = 480, crop_height = 360,
                                      aug_in_ngp = False,
                                      mk_conf = True)
    train_loader_dict = train_loader.get_config_dict()
    model = models.PSPNet(train_loader.dataset.num_classes, pretrained = False, backbone = 'resnet18', mk_conf = True)
    criterion = torch.nn.CrossEntropyLoss(**criterion_args)
    trainer_obj = utils.trainer(model = model, criterion = criterion, train_loader = train_loader,
                                val_loader = train_loader.get_val_loader(),
                                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                config = json.load(open('config.json', 'r')),
                                num_classes = train_loader.dataset.num_classes,
                                train_epochs = 100, val_interval = 5, optim_args = optimizer_args, optim_type = 'Adam',
                                optim_module = 'torch.optim',
                                save_interval = 10, aux_loss_factor = 0.4,
                                resume_path = 'saved/PSPNet/CamVid/best_model.pth', mk_conf = True)
    generate_config()
    trainer_obj.train()
