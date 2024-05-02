# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
import torch.distributed as dist

from fedtorch.utils.auxiliary import deepcopy_model
from fedtorch.comms.utils.flow_utils import quantize_tensor, dequantize_tensor


def fedavg_aggregation(cfg, model_server, model_client, group, online_clients, optimizer, lambda_weight=None):
    """Aggregate gradients for federated learning.

    Each local model first gets the difference between current model and
    previous synchronized model, and then all-reduce these difference by SUM.

    """
    num_online_clients = len(online_clients) if 0 in online_clients else len(online_clients) + 1
    if (0 not in online_clients) and (cfg.graph.rank == 0):
        rank_weight = 0
    else:
        if lambda_weight is None:
            # rank_weight =  cfg.data.num_samples_per_epoch / cfg.train_dataset_size
            rank_weight =  1.0 / num_online_clients
        else:
            #TODO: This is experimental. Test it.
            rank_weight = lambda_weight * cfg.graph.n_nodes / num_online_clients

    
    for i, (server_param, client_param) in enumerate(zip(model_server.parameters(), model_client.parameters())):
        # get model difference.
        client_param.grad.data = (server_param.data - client_param.data) * rank_weight
        # recover to old model.
        client_param.data = server_param.data
        
        # all reduce.This schema is not used in real federated setting
        # dist.all_reduce(client_param.grad.data,  op=dist.ReduceOp.SUM, group=group)

        #### Federated communication #####
        if cfg.federated.quantized:
            grad_q, q_info = quantize_tensor(client_param.grad.data, num_bits= cfg.federated.quantized_bits, adaptive=True)
            gather_list_tensor = [torch.ones_like(grad_q) for _ in range(num_online_clients)] if cfg.graph.rank == 0 else None
            gather_list_info   = [torch.ones(3) for _ in range(num_online_clients)] if cfg.graph.rank == 0 else None
            
            st = time.time()
            dist.gather(q_info, gather_list=gather_list_info, dst=0, group=group)
            dist.gather(grad_q, gather_list=gather_list_tensor, dst=0, group=group)
            cfg.comm_time[-1] += time.time() - st

            if cfg.graph.rank == 0:
                gather_list_tensor = gather_list_tensor if 0 in online_clients else gather_list_tensor[1:]
                gather_list_info = gather_list_info if 0 in online_clients else gather_list_info[1:]
                gather_list_deq = [dequantize_tensor(t,i) for t,i in zip(gather_list_tensor,gather_list_info)]
                d = torch.sum(torch.stack(gather_list_deq,1), dim=1)
                d, avg_info = quantize_tensor(d, num_bits= cfg.federated.quantized_bits, adaptive=True)
            else:
                d = torch.ones_like(grad_q)
                avg_info = torch.ones(3)
            
            st = time.time()
            dist.broadcast(avg_info, src=0, group=group)
            dist.broadcast(d, src=0, group=group)
            cfg.comm_time[-1] += time.time() - st
            client_param.grad.data = dequantize_tensor(d,avg_info)
        else:
            gather_list = [torch.ones_like(client_param.grad.data) for _ in range(num_online_clients)] if cfg.graph.rank == 0 else None
            st = time.time()
            dist.gather(client_param.grad.data, gather_list=gather_list, dst=0, group=group)
            cfg.comm_time[-1] += time.time() - st
            if cfg.graph.rank == 0:
                gather_list = gather_list if 0 in online_clients else gather_list[1:]
                d = torch.sum(torch.stack(gather_list,1), dim=1)
            else:
                d = torch.ones_like(client_param.grad.data)
            st = time.time()
            dist.broadcast(d, src=0, group=group)
            cfg.comm_time[-1] += time.time() - st
            client_param.grad.data = d

        
        if cfg.federated.type == 'fedadam':
            # Based on https://arxiv.org/abs/2003.00295
            cfg.federated.fedadam_v[i] = cfg.federated.fedadam_beta * cfg.federated.fedadam_v[i] + (1-cfg.federated.fedadam_beta) * torch.norm(client_param.grad.data)
            client_param.grad.data /= (np.sqrt(cfg.federated.fedadam_v[i]) + cfg.federated.fedadam_tau)
            


    # apply gradient to each client's model
    optimizer.step(
        apply_lr=False,
        scale=cfg.lr.lr_scale_at_sync,
        apply_in_momentum=False,
        apply_out_momentum=cfg.training.out_momentum,
    )

    # Reassign model_client to model_server
    model_server = deepcopy_model(cfg, model_client)

    return model_server