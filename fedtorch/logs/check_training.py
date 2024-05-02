# -*- coding: utf-8 -*-
from copy import deepcopy

import torch

from fedtorch.logs.logging import log


"""check the arguments for training."""

#TODO:Remove this file in the future.
def check_imagenet(cfg):
    # check the value of cfg.
    if 'imagenet' == cfg.data.dataset.type:
        if 'ILSVRC' not in cfg.data.data_dir:
            raise 'your should provide a correct data dir that can point to imagenet'


"""check the model when performing sync."""


def check_model_at_sync(cfg, model, is_weight, is_gradient):
    if cfg.local_index % cfg.training.summary_freq == 0:
        _check_model_at_sync(
            cfg.local_index, cfg.graph.rank, model,
            is_weight=True, is_gradient=True, debug=cfg.graph.debug)


def _check_model_at_sync(iter, gpu_id, model, is_weight=False, is_gradient=True, debug=True):
    model_parameters = list(model.parameters())
    param = model_parameters[0]
    if is_weight:
        log("iter:{}, check process {}'s weights for 1st variable:{}".format(
            iter, gpu_id, torch.norm(param.data)), debug)
    if is_gradient:
        log("iter:{}, check process {}'s gradients for 1st variable:{}".format(
            iter, gpu_id, torch.norm(param.grad.data)), debug)


"""track the model status when performing aggregation."""


def track_model_aggregation(cfg, initial_model, old_model, model):
    def calculate_cos(old_para, para):
        a, b = old_para.grad, para.grad

        if a is not None and b is not None:
            a, b = a.data, b.data
            cosine = torch.sum(a * b) / (a.norm() * b.norm())
            return float(cosine)
        else:
            return 0

    def calculate_parameter_distance_from_init(init_para, para):
        distance = (init_para.data - para.data).norm()
        return float(distance)

    def sample_between_two_model(old_model, model, alpha):
        tmp_model = deepcopy(model)
        for old_para, para in zip(old_model.parameters(),
                                  tmp_model.parameters()):
            para.data = alpha * old_para.data + (1 - alpha) * para.data
        return tmp_model

    list_of_cosine = []
    list_of_distance = []
    for init_para, old_para, para in zip(initial_model.parameters(),
                                         old_model.parameters(),
                                         model.parameters()):
        cos = calculate_cos(old_para, para)
        distance = calculate_parameter_distance_from_init(init_para, para)
        list_of_cosine.append(cos)
        list_of_distance.append(distance)

    cfg.tracking['cosine'].append(list_of_cosine)
    cfg.tracking['distance'].append(list_of_distance)
