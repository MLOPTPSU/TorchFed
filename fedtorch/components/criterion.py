# -*- coding: utf-8 -*-

import torch.nn as nn


def define_criterion(cfg):
    if 'least_square' in cfg.model.type:
        criterion = nn.MSELoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    return criterion
