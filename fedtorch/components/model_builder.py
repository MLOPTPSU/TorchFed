# Copyright (c) FedTorch. All rights reserved.
import torch.distributed as dist

from fedtorch.utils import Registry, build

MODEL = Registry("model")

from .models import *

def build_model_from_config(model_cfg, graph_cfg, dataset_cfg, is_distributed=True):
    if graph_cfg.rank % 100 == 0:
        print("=> creating model '{}' for rank {}/{}".format(model_cfg.type, graph_cfg.rank , graph_cfg.n_nodes))
    model_cfg.dataset_config = dataset_cfg
    model = build(model_cfg, MODEL)
    if is_distributed:
        consistent_model(graph_cfg, model)
    if graph_cfg.debug:
        get_model_stat(graph_cfg, model)
    return model

def get_model_stat(graph_cfg, model):
    print('Total params for process {}: {}M'.format(
        graph_cfg.rank,
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        ))


def consistent_model(graph_cfg, model):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    print('consistent model for process (rank {})'.format(graph_cfg.rank))
    cur_rank = graph_cfg.rank
    for param in model.parameters():
        param.data = param.data if cur_rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    