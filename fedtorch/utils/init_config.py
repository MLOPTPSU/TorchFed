# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist

from fedtorch.logs.checkpoint import init_checkpoint
from fedtorch.comms.algorithms.distributed import configure_sync_scheme
from fedtorch.utils.topology import FCGraph


def set_local_stat(cfg):
    cfg.local_index = 0
    cfg.client_epoch_total = 0
    cfg.block_index = 0
    cfg.global_index = 0
    cfg.local_data_seen = 0
    cfg.best_prec1 = 0
    cfg.best_epoch = []
    cfg.rounds_comm = 0
    cfg.tracking = {'cosine': [], 'distance': []}
    cfg.comm_time = []


def init_config(cfg):
    # define the graph for the computation.
    cur_rank = dist.get_rank()
    cfg.graph = FCGraph(cur_rank, cfg.device.blocks, cfg.device.on_cuda, cfg.device.world)

    # TODO: Add parameter for this to enable it for other nodes
    if cfg.graph.rank != 0:
        if not cfg.chekpoint.debug:
            cfg.graph.debug=False


    if cfg.device.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(cfg.training.manual_seed)
        torch.cuda.set_device(cfg.graph.device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # local conf.
    set_local_stat(cfg)

    # define checkpoint for logging.
    init_checkpoint(cfg)

    # define sync scheme.
    configure_sync_scheme(cfg)


def init_config_centered(cfg,rank):
    # define the graph for the computation.
    cur_rank =  rank
    cfg.graph = FCGraph(cur_rank, cfg.device.blocks, cfg.device.on_cuda, cfg.device.world)

    # TODO: Add parameter for this to enable it for other nodes
    if cfg.graph.rank != 0:
        cfg.graph.debug=False


    if cfg.device.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(cfg.training.manual_seed)
        # torch.cuda.set_device(cfg.graph.device)
        torch.cuda.set_device(0)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # local conf.
    set_local_stat(cfg)

    # define checkpoint for logging.
    init_checkpoint(cfg)

    # define sync scheme.
    configure_sync_scheme(cfg)