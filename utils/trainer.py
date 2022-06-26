# -*- coding=utf-8 -*-
# @Time: 2021/8/1 12:51
# @Author: N
# @Software: PyCharm

import datetime
import math
import os
import time
import data
import numpy
import tensorboardX
import torch
import torchvision.utils
import utils
from tqdm import tqdm


class Trainer:

    def __init__(self, model, num_classes, criterion, config, train_epochs, save_interval, val_interval, train_loader,
                 optim_module: str, optim_type: str,
                 optim_args: dict, resume_path: str = None, val_loader = None,
                 device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), epochs = 0,
                 tensorboard = True, backbone_lr_factor = 0.2, aux_loss_factor = .0,
                 mnt_metric = 'mean_IoU', early_stopping = 10,
                 lr_scheduler_name: str = None, freeze_bn = False, mk_conf = False, data_name = ''):
        if mk_conf: self.config_dict = locals().copy()
        self.model, self.num_classes, self.criterion, self.config, self.train_epochs, self.save_interval, self.val_interval, self.resume_path, self.train_loader, self.val_loader, self.device, self.tensorboard, self.epochs, self.aux_loss_factor, self.freeze_bn, self.lr_scheduler_name = model, num_classes, criterion, config, train_epochs, save_interval, val_interval, resume_path, train_loader, val_loader, device, tensorboard, epochs, aux_loss_factor, freeze_bn, lr_scheduler_name
        # for item in list(locals().items()):
        #   if item[0]!='self': setattr(self,item[0],item[1])
        # define tensorboard writer
        arch_name = model.__class__.__name__
        self.tensorboard_path = os.path.join('saved/runs', arch_name, datetime.datetime.now().strftime("%m-%d_%H-%M"))
        self.checkpoint_dir, self.writer = os.path.join('saved', arch_name, data_name), tensorboardX.SummaryWriter(
            self.tensorboard_path, flush_secs = 60) if tensorboard else None
        # set monitoring metric and rounds to stop early
        self.mnt_metric, self.early_stop, self.mnt_best, self.not_improved_count = mnt_metric, early_stopping, -math.inf, 0
        # set parallel calculation
        if torch.cuda.device_count() > 1 and device != torch.device('cpu'):
            self.model = torch.nn.DataParallel(self.model, device_ids = list(range(torch.cuda.device_count()))).to(
                device)
        elif device == torch.device('cpu'):
            print('training sans gpu')
        else:
            self.model = self.model.to(device)
        # setting different learning rate for the backbone and decoder
        if arch_name.lower().startswith('psp') and backbone_lr_factor < 1 and backbone_lr_factor >= 0:
            if isinstance(self.model, torch.nn.DataParallel):
                trainable_params = [
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_decoder_params())},
                    {'params': filter(lambda p: p.requires_grad, self.model.module.get_backbone_params()),
                     'lr': optim_args['lr'] * backbone_lr_factor}]
            else:
                trainable_params = [{'params': filter(lambda p: p.requires_grad, self.model.get_decoder_params())},
                                    {'params': filter(lambda p: p.requires_grad, self.model.get_backbone_params()),
                                     'lr': optim_args['lr'] * backbone_lr_factor}]
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        # optimizer
        exec(f'self.optimizer=getattr({optim_module},optim_type)(trainable_params,**optim_args)')
        # lr_scheduler
        if self.lr_scheduler_name: self.lr_scheduler = utils.lr_scheduler(self.optimizer, train_epochs = train_epochs,
                                                                          lr_scheduler_name = lr_scheduler_name).get_lr_scheduler()
        # prefetch
        if device != torch.device('cpu'): self.train_loader, self.val_loader = data.DataPrefetcher(train_loader,
                                                                                                   self.device), data.DataPrefetcher(
            val_loader, self.device) if val_loader is not None else None
        # resume
        if resume_path: self._resume()

    def train(self):
        if self.freeze_bn:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        for self.epochs in range(self.epochs, self.train_epochs):
            self.model.train()
            proc_bar = tqdm(desc = f'training_epoch_{self.epochs + 1}', total = len(self.train_loader), ncols = 0)
            self._init_epoch_metrics()
            if self.lr_scheduler_name: self.lr_scheduler.step(self.epochs - 1)
            for inputs, labels in self.train_loader:
                loss, other_metrics = self._model_fn(inputs, labels)
                self._updata_epoch_metrics(loss.item(), other_metrics)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                proc_bar.update()
                proc_bar.set_postfix(
                    {'loss': round(loss.item(), 3),
                     'pixel_acc': str(round(other_metrics[0] / other_metrics[1] * 100, 3)) + '%',
                     'mIoU': str(
                         round((other_metrics[2] / (other_metrics[3] + numpy.spacing(1))).mean() * 100, 3)) + '%'})
            proc_bar.close()
            epoch_metrics = self._get_epoch_metrics()
            for k, v in epoch_metrics.items(): self.writer.add_scalar(f'training_{k}', v, self.epochs + 1)
            for i, opt_group in enumerate(self.optimizer.param_groups): self.writer.add_scalar(f'learning_rate_{i}',
                                                                                               opt_group['lr'],
                                                                                               self.epochs + 1)
            if self.mnt_metric:
                try:
                    mnt_current = epoch_metrics[self.mnt_metric] if self.mnt_metric != 'mean_loss' else -epoch_metrics[
                        self.mnt_metric]
                except:
                    print(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    raise Exception(f'except {self.mnt_metric} which has not been calculated as the monitor metric')
                self.not_improved_count, self.mnt_best = (0, mnt_current) if mnt_current > self.mnt_best else (
                    self.not_improved_count + 1, self.mnt_best)
                if self.not_improved_count == 0:
                    self._save(presave_best = True)
                elif self.not_improved_count >= self.early_stop:
                    print(f'\nPerformance didn\'t improve for {self.early_stop} epochs')
                    break
            if (self.epochs + 1) % self.save_interval == 0: self._save()
            if (self.epochs + 1) % self.val_interval == 0 and self.val_loader: self.validate()

    def validate(self):
        self.model.eval()
        self._init_epoch_metrics()
        proc_bar = tqdm(total = len(self.val_loader), ncols = 0, desc = 'validating')
        visual = []
        for inputs, labels in self.val_loader:
            logits, loss, other_metrics = self._model_fn(inputs, labels, True)
            self._updata_epoch_metrics(loss.item(), other_metrics)
            proc_bar.update()
            proc_bar.set_postfix(
                {'loss': round(loss.item(), 3),
                 'pixel_acc': str(round(other_metrics[0] / other_metrics[1] * 100, 3)) + '%',
                 'mIoU': str(round((other_metrics[2] / (other_metrics[3] + numpy.spacing(1))).mean() * 100, 3)) + '%'})
            if len(visual) < 15: visual.extend(
                [(self.train_loader.dataset.denormalize(inputs[0].cpu().permute((1, 2, 0)))).permute((2, 0, 1)),
                 utils.class2rgb(logits[0].cpu().argmax(dim = 0), self.train_loader.dataset.class2rgb_lut) / 255,
                 utils.class2rgb(labels[0].cpu(), self.train_loader.dataset.class2rgb_lut) / 255])
        proc_bar.close()
        val_metrics = self._get_epoch_metrics()
        for k, v in val_metrics.items(): self.writer.add_scalar(f'validating_{k}', v,
                                                                (self.epochs + 1) // self.val_interval)
        self.writer.add_image('val_examples', torchvision.utils.make_grid(visual, 3),
                              (self.epochs + 1) // self.val_interval)

    def _model_fn(self, inputs, labels, is_val = False):
        logits = self.model(inputs)
        if self.model.training and self.aux_loss_factor > 0:
            if isinstance(logits, torch.Tensor): raise Exception('wrong cfg about auxiliary Loss')
            loss, logits = self.criterion(logits[0], labels) + self.aux_loss_factor * self.criterion(logits[1], labels),\
                           logits[0]
        else:
            if not isinstance(logits, torch.Tensor): logits = logits[0]
            loss = self.criterion(logits, labels)
        if isinstance(loss, torch.nn.DataParallel):
            loss = loss.mean()
        with torch.no_grad():
            other_metrics = self._get_other_metrics(logits, labels)
        if not is_val:
            return loss, other_metrics
        else:
            return logits, loss, other_metrics

    def _get_other_metrics(self, logits, labels):
        predicts = torch.argmax(logits.permute((0, 2, 3, 1)), dim = -1)
        num_accurate = (predicts == labels).sum().item()
        num_labeled = labels.size()[0] * labels.size()[1] * labels.size()[2]
        # when object num was equal to both the left boundary and the right, it'd be threw into the right bin.
        histogram_intersection = torch.histc(((predicts + 1) * (predicts == labels) - 1).float(),
                                             bins = self.num_classes, min = 0, max = self.num_classes - 1)
        histogram_predicts = torch.histc(predicts.float(), bins = self.num_classes, min = 0, max = self.num_classes - 1)
        histogram_labels = torch.histc(labels.float(), bins = self.num_classes, min = 0, max = self.num_classes - 1)
        return num_accurate, num_labeled, histogram_intersection.cpu().numpy(), (
                histogram_predicts + histogram_labels - histogram_intersection).cpu().numpy()

    def _init_epoch_metrics(self):
        self.total_loss, self.num_batchs, self.num_accurate, self.num_labeled, self.histogram_intersection, self.histogram_union = 0, 0, 0, 0, numpy.zeros(
            self.num_classes), numpy.zeros(self.num_classes)

    def _updata_epoch_metrics(self, loss, other_metrics):
        self.total_loss += loss
        self.num_batchs += 1
        self.num_accurate += other_metrics[0]
        self.num_labeled += other_metrics[1]
        self.histogram_intersection += other_metrics[2]
        self.histogram_union += other_metrics[3]

    def _get_epoch_metrics(self):
        return {'mean_loss': round(self.total_loss / self.num_batchs, 3),
                'pixel_accuracy': round(self.num_accurate / self.num_labeled, 3),
                'mean_IoU': round((self.histogram_intersection / (self.histogram_union + numpy.spacing(1))).mean(), 3)}

    def _save(self, presave_best = False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': self.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if presave_best:
            self.best_state = state
            return
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{self.epochs}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)
        if self.best_state and self.epochs - self.best_state['epoch'] < self.save_interval:
            filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(self.best_state, filename)
            print("Saving current best: best_model.pth")

    def _resume(self):
        print(f'Loading checkpoint : {self.resume_path}')
        checkpoint = torch.load(self.resume_path)
        # Load last run info, the models params, the optimizer and do consistence check
        self.epochs = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0
        if checkpoint['config']['model'] != self.config['model']:
            print({'Warning! Current models is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['config']['trainer']['args']['optim_type'] != self.config['trainer']['args']['optim_type']:
            print({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Checkpoint <{self.resume_path}> (epoch {self.epochs}) was loaded')

    def get_config_dict(self):
        for v in ['self', 'mk_conf', '__class__', 'num_classes', 'config', 'val_loader', 'train_loader', 'criterion',
                  'model', 'device', 'trainable_params', 'arch_name']:
            try:
                del self.config_dict[v]
                print(v)
            except:
                pass
        return self.config_dict
