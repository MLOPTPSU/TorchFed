# -*- coding: utf-8 -*-
from fedtorch.components.optimizer_builder import build_optimizer_from_config
from fedtorch.components.criterion import define_criterion
from fedtorch.components.metrics import define_metrics
from fedtorch.components.model_builder import build_model_from_config
from fedtorch.components.scheduler_builder import build_lr_scheduler_from_config
from fedtorch.logs.checkpoint import maybe_resume_from_checkpoint


def create_components(cfg):
    """Create model, criterion and optimizer.
    If cfg.device.on_cuda is True, use ps_id as GPU_id.
    """
    model = build_model_from_config(cfg.model, cfg.graph, cfg.data.dataset, is_distributed=cfg.device.is_distributed)

    # define the criterion and metrics.
    criterion = define_criterion(cfg)
    metrics = define_metrics(cfg, model)

    # define the lr scheduler.
    scheduler = build_lr_scheduler_from_config(cfg.lr, cfg.training.batch_size, cfg.graph.n_nodes, cfg.training.num_epochs)

    # define the optimizer.
    optimizer = build_optimizer_from_config(cfg.optimizer, model, cfg.graph.n_nodes, cfg.lr.lr)

    # place model and criterion.
    if cfg.graph.on_cuda:
        model.cuda()
        criterion = criterion.cuda()

    # (optional) reload checkpoint
    maybe_resume_from_checkpoint(cfg, model, optimizer)
    return model, criterion, scheduler, optimizer, metrics
