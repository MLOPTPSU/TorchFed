# -*- coding: utf-8 -*-
from fedtorch.utils import Registry, build

LR_SCHEDULER = Registry("scheduler")

from .learning_rate import *

def adjust_learning_rate(cfg, optimizer, lr_scheduler, lr_external=None):
    """Sets the learning rate to the initial LR decayed by # of accessed sample
        We should decay the learning rate based on the number of samples that
        we have accessed.
    """
    # adjust and assign learning rate.
    if lr_external is None:
        lr = lr_scheduler(cfg.epoch_)

        if lr is None:
            lr = cfg.lr.old_learning_rate

        if cfg.lr.old_learning_rate != lr:
            cfg.lr.old_learning_rate = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_external
        lr = lr_external
    return lr


def build_lr_scheduler_from_config(lr_cfg, batch_size, n_nodes, num_epochs):
    # get the learning rate per sample.
    learning_rate_per_samples = lr_cfg.lr / batch_size

    # get a valid learning rate.
    if lr_cfg.scheduler.type == 'multistep':
        lr_cfg.scheduler.init_warmup_lr = lr_cfg.lr

    if lr_cfg.scheduler.type in ['multistep', 'convex_decay']:
        if lr_cfg.lr_scaleup:
            if lr_cfg.lr_scaleup_type == 'linear':
                _lr = learning_rate_per_samples * batch_size
                _scale = n_nodes
            elif lr_cfg.lr_scaleup_type == 'sqrt':
                _lr = lr_cfg.lr
                _scale = (
                    1. * n_nodes * batch_size /
                    lr_cfg.base_batch_size) ** 0.5
            else:
                raise NotImplementedError
        else:
            _lr = learning_rate_per_samples * batch_size
            _scale = 1
        lr_cfg.scheduler.learning_rate = _lr * _scale

        # just backup the current learning rate.
        lr_cfg.old_learning_rate = lr_cfg.scheduler.learning_rate

    lr_cfg.scheduler.num_epochs = num_epochs
    # define the learning rate scheduler.
    return build(lr_cfg.scheduler, LR_SCHEDULER)

