# -*- coding: utf-8 -*-
import time
from copy import deepcopy
import numpy as np

import torch
import torch.distributed as dist

from fedtorch.components.scheduler_builder import adjust_learning_rate
from fedtorch.components.dataset_builder import load_data_batch
from fedtorch.logs.checkpoint import save_to_checkpoint
from fedtorch.comms.utils.flow_utils import (get_current_epoch, 
                                             get_current_local_step,
                                             zero_copy, 
                                             is_sync_fed)
from fedtorch.comms.utils.eval import inference, do_validate
from fedtorch.comms.algorithms.federated import (fedavg_aggregation,
                                                 fedgate_aggregation,
                                                 scaffold_aggregation,
                                                 qsparse_aggregation, 
                                                 distribute_model_server_control,
                                                 set_online_clients,
                                                 distribute_model_server)
from fedtorch.logs.logging import (log,
                                   logging_computing,
                                   logging_sync_time,
                                   logging_display_training,
                                   logging_load_time,
                                   logging_globally)
from fedtorch.logs.meter import define_local_training_tracker

def train_and_validate_federated(client):
    """The training scheme of Federated Learning systems.
        The basic model is FedAvg https://arxiv.org/abs/1602.05629
        TODO: Merge different models under this method
    """
    log('start training and validation with Federated setting.', client.cfg.graph.debug)


    if client.cfg.evaluate and client.cfg.graph.rank==0:
        # Do the training on the server and return
        do_validate(client.cfg, client.model, client.optimizer,  client.criterion, client.metrics,
                         client.test_loader, client.all_clients_group, data_mode='test')
        return

    # init global variable.

    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.', client.cfg.graph.debug)

    # Number of communication rounds in federated setting should be defined
    for n_c in range(client.cfg.federated.num_comms):
        client.cfg.rounds_comm += 1
        client.cfg.comm_time.append(0.0)
        # Configuring the devices for this round of communication
        log("Starting round {} of training".format(n_c+1), client.cfg.debug)
        online_clients = set_online_clients(client.cfg)
        if (n_c == 0) and  (0 not in online_clients):
            online_clients += [0]
        online_clients_server = online_clients if 0 in online_clients else online_clients + [0]
        online_clients_group = dist.new_group(online_clients_server)
        
        if client.cfg.graph.rank in online_clients_server:
            if  client.cfg.federated.type == 'scaffold':
                st = time.time()
                client.model_server, client.model_server_control = distribute_model_server_control(client.model_server, 
                                                                                                   client.model_server_control, 
                                                                                                   online_clients_group, 
                                                                                                   src=0)
                client.cfg.comm_time[-1] += time.time() - st
            else:
                st = time.time()
                client.model_server = distribute_model_server(client.model_server, online_clients_group, src=0)
                client.cfg.comm_time[-1] += time.time() - st
            client.model.load_state_dict(client.model_server.state_dict())
            local_steps = 0
            if client.cfg.graph.rank in online_clients:
                is_sync = False
                while not is_sync:
                    for _input, _target in client.train_loader:
                        local_steps += 1
                        client.model.train()

                        # update local step.
                        logging_load_time(tracker)

                        # update local index and get local step
                        client.cfg.local_index += 1
                        client.cfg.local_data_seen += len(_target)
                        get_current_epoch(client.cfg)
                        local_step = get_current_local_step(client.cfg)

                        # adjust learning rate (based on the # of accessed samples)
                        lr = adjust_learning_rate(client.cfg, client.optimizer, client.scheduler)

                        # load data
                        _input, _target = load_data_batch(client.cfg, _input, _target, tracker)

                        # Skip batches with one sample because of BatchNorm issue in some models!
                        if _input.size(0)==1:
                            is_sync = is_sync_fed(client.cfg)
                            break

                        # inference and get current performance.
                        client.optimizer.zero_grad()
                       
                        loss, performance = inference(client.model, client.criterion, client.metrics, _input, _target)

                        # compute gradient and do local SGD step.
                        loss.backward()

                        if client.cfg.federated.type == 'fedgate':
                            # Update gradients with control variates
                            for client_param, delta_param  in zip(client.model.parameters(), client.model_delta.parameters()):
                                client_param.grad.data -= delta_param.data 
                        elif client.cfg.federated.type == 'scaffold':
                            for cp, ccp, scp  in zip(client.model.parameters(), client.model_client_control.parameters(), client.model_server_control.parameters()):
                                cp.grad.data += scp.data - ccp.data
                        elif client.cfg.federated.type == 'fedprox':
                            # Adding proximal gradients and loss for fedprox
                            for client_param, server_param in zip(client.model.parameters(), client.model_server.parameters()):
                                if client.cfg.graph.rank == 0:
                                    print("distance norm for prox is:{}".format(torch.norm(client_param.data - server_param.data )))
                                loss += client.cfg.federated.fedprox_mu /2 * torch.norm(client_param.data - server_param.data)
                                client_param.grad.data += client.cfg.federated.fedprox_mu * (client_param.data - server_param.data)
                        
                        if getattr(client.cfg.model, 'robust', False):
                            client.model.noise.grad.data *= -1

                        client.optimizer.step(
                            apply_lr=True,
                            apply_in_momentum=client.cfg.training.in_momentum, apply_out_momentum=False
                        )

                        if getattr(client.cfg.model, 'robust', False):
                            if torch.norm(client.model.noise.data) > 1:
                                client.model.noise.data /= torch.norm(client.model.noise.data)
                        
                        # logging locally.
                        # logging_computing(tracker, loss_v, performance_v, _input, lr)
                        
                        # display the logging info.
                        # logging_display_training(cfg, tracker)


                        # reset load time for the tracker.
                        tracker['start_load_time'] = time.time()
                        # model_local = deepcopy(model_client)
                        is_sync = is_sync_fed(client.cfg)
                        if is_sync:
                            break

            else:
                log("Offline in this round. Waiting on others to finish!", client.cfg.graph.debug)

            # Validate the local models befor sync
            do_validate(client.cfg, client.model, client.optimizer, client.criterion, client.metrics, 
                        client.train_loader, online_clients_group, data_mode='train', local=True)
            if client.cfg.federated.personal:
                do_validate(client.cfg, client.model, client.optimizer, client.criterion, client.metrics, 
                            client.val_loader, online_clients_group, data_mode='validation', local=True)
            # Sync the model server based on client models
            log('Enter synching', client.cfg.graph.debug)
            tracker['start_sync_time'] = time.time()
            client.cfg.global_index += 1

            if client.cfg.federated.type == 'fedgate':
                client.model_server, client.model_delta = fedgate_aggregation(client.cfg, client.model_server, client.model, 
                                                                              client.model_delta, client.model_memory, 
                                                                              online_clients_group, online_clients, 
                                                                              client.optimizer, lr, local_steps)
            elif client.cfg.federated.type == 'scaffold':
                client.model_server, client.model_client_control, client.model_server_control = scaffold_aggregation(client.cfg, client.model_server, 
                                                                                                                     client.model, client.model_server_control, 
                                                                                                                     client.model_client_control, online_clients_group, 
                                                                                                                     online_clients, client.optimizer, lr, local_steps)
            elif client.cfg.federated.type == 'qsparse':
                client.model_server = qsparse_aggregation(client.cfg, client.model_server, client.model, 
                                                          online_clients_group, online_clients, 
                                                          client.optimizer, client.model_memory)
            else:
                client.model_server = fedavg_aggregation(client.cfg, client.model_server, client.model, 
                                                         online_clients_group, online_clients, client.optimizer)
             # evaluate the sync time
            logging_sync_time(tracker)
            
            # Do the validation on the server model
            do_validate(client.cfg, client.model_server, client.optimizer, client.criterion, client.metrics, 
                        client.train_loader, online_clients_group, data_mode='train')
            if client.cfg.federated.personal:
                do_validate(client.cfg, client.model_server, client.optimizer, client.criterion, client.metrics, 
                            client.val_loader, online_clients_group, data_mode='validation')

            # logging.
            logging_globally(tracker, start_global_time)
            
            # reset start round time.
            start_global_time = time.time()

            # validate the model at the server
            if client.cfg.graph.rank == 0:
                do_validate(client.cfg, client.model_server, client.optimizer, client.criterion, client.metrics, 
                            client.test_loader, online_clients_group, data_mode='test')
            log('This round communication time is: {}'.format(client.cfg.comm_time[-1]), client.cfg.graph.debug)
        else:
            log("Offline in this round. Waiting on others to finish!", client.cfg.graph.debug)
        dist.barrier(group=client.all_clients_group)

    return