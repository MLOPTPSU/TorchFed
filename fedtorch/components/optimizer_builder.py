# -*- coding: utf-8 -*-
from fedtorch.utils import Registry, build

OPTIMIZER = Registry("optimizer")

from .optimizers import *

def build_optimizer_from_config(opt_cfg, model, n_nodes, lr):
    # define the param to optimize.
    params_dict = dict(model.named_parameters())
    params = [
        {
            'params': [value],
            'name': key,
            'weight_decay': opt_cfg.weight_decay if 'bn' not in key else 0.0
        }
        for key, value in params_dict.items()
    ]
    opt_cfg.params = params
    opt_cfg.lr = lr
    if opt_cfg.type == 'SGD':
        opt_cfg.out_momentum = opt_cfg.out_momentum if opt_cfg.out_momentum is not None else 1.0 - 1.0 / n_nodes
    return build(opt_cfg, OPTIMIZER)

    # # define the optimizer.
    # if args.optimizer == 'sgd':
    #     return SGD(
    #         params, lr=args.learning_rate,
    #         in_momentum=args.in_momentum_factor,
    #         out_momentum=(args.out_momentum_factor
    #                       if args.out_momentum_factor is not None
    #                       else 1.0 - 1.0 / args.graph.n_nodes),
    #         nesterov=args.use_nesterov, args=args)
    # else:
    #     return AdamW(
    #         params, lr=args.learning_rate,
    #         correct_wd=args.correct_wd
    #     )
