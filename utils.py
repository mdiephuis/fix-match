import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def type_tdouble(use_cuda=False):
    return torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def one_hot(labels, n_class, use_cuda=False):
    # Ensure labels are [N x 1]
    if len(list(labels.size())) == 1:
        labels = labels.unsqueeze(1)
    mask = type_tdouble(use_cuda)(labels.size(0), n_class).fill_(0)
    # scatter dimension, position indices, fill_value
    return mask.scatter_(1, labels, 1)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def cosine_warmup_scheduler(optimizer, num_warmup_steps, num_train_steps):

    def _cosine_warmup_scheduler(curr_step):
        if curr_step < num_warmup_steps:
            return float(curr_step) / float(max(1, num_warmup_steps))
        else:
            value = np.cos(7 * np.pi / 16. * (curr_step - num_warmup_steps) / (num_train_steps - num_warmup_steps))
            return max(0, value)

    return LambdaLR(optimizer, _cosine_warmup_scheduler, last_epoch=-1)
