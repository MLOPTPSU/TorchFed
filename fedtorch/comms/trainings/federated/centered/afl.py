# -*- coding: utf-8 -*-
import time
from copy import deepcopy
import numpy as np

import torch

from fedtorch.components.scheduler_builder import adjust_learning_rate
from fedtorch.components.dataset_builder import load_data_batch
from fedtorch.comms.utils.flow_utils import (get_current_epoch, 
                                             get_current_local_step, 
                                             is_sync_fed,
                                             projection_simplex_sort)
from fedtorch.comms.utils.eval import inference
from fedtorch.comms.utils.eval_centered import (do_validate_centered, 
                                                log_validation_centered,
                                                log_validation_per_client_centered,
                                                log_test_centered)
from fedtorch.comms.algorithms.federated import fedavg_aggregation_centered, set_online_clients_centered
from fedtorch.logs.logging import (log, 
                                   logging_sync_time,
                                   logging_load_time,
                                   logging_globally)
from fedtorch.logs.meter import define_local_training_tracker

def train_and_validate_afl_centered(Clients, Server):
    log('start training and validation with Federated setting in a centered way.')

    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.')

    for oc in range(Server.cfg.graph.n_nodes):
        Server.lambda_vector[oc] = Clients[oc].cfg.data.num_samples_per_epoch
    Server.lambda_vector /= Server.lambda_vector.sum()
    # Number of communication rounds in federated setting should be defined
    for n_c in range(Server.cfg.federated.num_comms):
        Server.cfg.rounds_comm += 1
        Server.cfg.local_index += 1
        Server.cfg.quant_error = 0.0
        
        # Preset variables for this round of communication
        Server.zero_grad()
        Server.reset_tracker(Server.local_val_tracker)
        Server.reset_tracker(Server.global_val_tracker)
        Server.reset_tracker(Server.global_test_tracker)
        if Server.cfg.federated.personal:
            Server.reset_tracker(Server.local_personal_val_tracker)
            Server.reset_tracker(Server.global_personal_val_tracker) 

        # Configuring the devices for this round of communication
        log("Starting round {} of training".format(n_c+1))
        online_clients = set_online_clients_centered(Server.cfg)
        
        Server.cfg.federated.drfa_gamma *= 0.9

        for oc in online_clients:
            Clients[oc].model.load_state_dict(Server.model.state_dict())
            Clients[oc].cfg.rounds_comm = Server.cfg.rounds_comm
            loss_tensor = torch.zeros(Server.cfg.graph.n_nodes)
            local_steps = 0
            is_sync = False

            do_validate_centered(Clients[oc].cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                 Clients[oc].train_loader, Server.global_val_tracker, val=False, local=False)
            if Server.cfg.federated.per_class_acc:
                Clients[oc].reset_tracker(Clients[oc].local_val_tracker)
                Clients[oc].reset_tracker(Clients[oc].global_val_tracker)
                if Server.cfg.federated.personal:
                    Clients[oc].reset_tracker(Clients[oc].local_personal_val_tracker)
                    Clients[oc].reset_tracker(Clients[oc].global_personal_val_tracker)
                    do_validate_centered(Clients[oc].cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                                            Clients[oc].val_loader, Clients[oc].global_personal_val_tracker, val=True, local=False) 
                do_validate_centered(Clients[oc].cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                    Clients[oc].train_loader, Clients[oc].global_val_tracker, val=False, local=False)
            if Server.cfg.federated.personal:
                do_validate_centered(Clients[oc].cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                    Clients[oc].val_loader, Server.global_personal_val_tracker, val=True, local=False)
            
            while not is_sync:
                if Server.cfg.model.type == 'rnn':
                    Clients[oc].model.init_hidden(Server.cfg.training.batch_size)
                for _input, _target in Clients[oc].train_loader:
                    local_steps += 1
                    Clients[oc].model.train()

                    # update local step.
                    logging_load_time(tracker)
                    
                    # update local index and get local step
                    Clients[oc].cfg.local_index += 1
                    Clients[oc].cfg.local_data_seen += len(_target)
                    get_current_epoch(Clients[oc].cfg)
                    local_step = get_current_local_step(Clients[oc].cfg)

                    # adjust learning rate (based on the # of accessed samples)
                    lr = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer, Clients[oc].scheduler)

                    # load data
                    _input, _target = load_data_batch(Clients[oc].cfg, _input, _target, tracker)
        
                    # Skip batches with one sample because of BatchNorm issue in some models!
                    if _input.size(0)==1:
                        is_sync = is_sync_fed(Clients[oc].cfg)
                        break

                    # inference and get current performance.
                    Clients[oc].optimizer.zero_grad()
                    
                    loss, performance = inference(Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, 
                                                    _input, _target, rnn=Server.cfg.model.type in ['rnn'])
                    # print("loss in rank {} is {}".format(oc,loss))
                    loss_tensor[oc] = loss.item()
                    # compute gradient and do local SGD step.
                    loss.backward()
                    # if oc == 0:
                    #     print(loss.data)

                    if 'fedgate' in Clients[oc].cfg.federated.type:
                        # Update gradients with control variates
                        for client_param, delta_param  in zip(Clients[oc].model.parameters(), Clients[oc].model_delta.parameters()):
                            client_param.grad.data -= delta_param.data 
                    elif Clients[oc].cfg.federated.type == 'scaffold':
                        for cp, ccp, scp  in zip(Clients[oc].model.parameters(), Clients[oc].model_client_control.parameters(), Server.model_server_control.parameters()):
                            cp.grad.data += scp.data - ccp.data
                    elif Clients[oc].cfg.federated.type == 'fedprox':
                        # Adding proximal gradients and loss for fedprox
                        for client_param, server_param in zip(Clients[oc].model.parameters(),Server.model.parameters()):
                            loss += Clients[oc].cfg.federated.fedprox_mu /2 * torch.norm(client_param.data - server_param.data)
                            client_param.grad.data += Clients[oc].cfg.federated.fedprox_mu * (client_param.data - server_param.data)


                    Clients[oc].optimizer.step(
                        apply_lr=True,
                        apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                    )
                    
                    if Clients[oc].cfg.epoch_ % 1 == 0:
                        Clients[oc].cfg.finish_one_epoch = True
                    
                        # refresh the logging cache at the begining of each epoch.
                        Clients[oc].cfg.finish_one_epoch = False
                        tracker = define_local_training_tracker()

                    # reset load time for the tracker.
                    tracker['start_load_time'] = time.time()
                    # model_local = deepcopy(model_client)
                    is_sync = is_sync_fed(Clients[oc].cfg)
                    if is_sync:
                        break

            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].train_loader, Server.local_val_tracker, val=False, local=True)
            if Server.cfg.federated.per_class_acc:
                do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                    Clients[oc].train_loader, Clients[oc].local_val_tracker, val=False, local=True)
                if Server.cfg.fed_personal:
                    do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                                                Clients[oc].val_loader, Clients[oc].local_personal_val_tracker, val=True, local=True)
            if Server.cfg.federated.personal:
                do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].val_loader, Server.local_personal_val_tracker, val=True, local=True)
            # Sync the model server based on model_clients
            tracker['start_sync_time'] = time.time()
            Server.cfg.global_index += 1

            logging_sync_time(tracker)

        fedavg_aggregation_centered(Clients, Server, online_clients, Server.lambda_vector.numpy())

        # Log performance
        # Client training performance
        log_validation_centered(Server.cfg, Server.local_val_tracker, val=False, local=True)
        # Server training performance
        log_validation_centered(Server.cfg, Server.global_val_tracker, val=False, local=False)
        if Server.cfg.federated.personal:
            # Client validation performance
            log_validation_centered(Server.cfg, Server.local_personal_val_tracker, val=True, local=True)
            # Server validation performance
            log_validation_centered(Server.cfg, Server.global_personal_val_tracker, val=True, local=False)

        # Per client stats
        if Server.cfg.federated.per_class_acc:
            log_validation_per_client_centered(Server.cfg, Clients, online_clients, val=False, local=False)
            log_validation_per_client_centered(Server.cfg, Clients, online_clients, val=False, local=True)
            if Server.cfg.federated.personal:
                log_validation_per_client_centered(Server.cfg, Clients, online_clients, val=True, local=False)
                log_validation_per_client_centered(Server.cfg, Clients, online_clients, val=True, local=True)

            

        # Test on server
        do_validate_centered(Server.cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                 Server.test_loader, Server.global_test_tracker, val=False, local=False)
        log_test_centered(Server.cfg,Server.global_test_tracker)


        
        Server.lambda_vector += Server.cfg.drfa_gamma * loss_tensor
        lambda_vector = projection_simplex_sort(Server.lambda_vector.detach().numpy())
        print(lambda_vector)
        # Avoid zero probability
        lambda_zeros = np.argwhere(lambda_vector <= 1e-3)
        if len(lambda_zeros)>0:
            lambda_vector[lambda_zeros[0]] = 1e-3
            lambda_vector /= np.sum(lambda_vector)
        Server.lambda_vector = torch.tensor(lambda_vector)

        # logging.
        logging_globally(tracker, start_global_time)
        # reset start round time.
        start_global_time = time.time()
    return