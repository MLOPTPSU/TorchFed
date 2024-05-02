# -*- coding: utf-8 -*-
import time
from copy import deepcopy
import os

import torch
import ray

from fedtorch.components.scheduler_builder import adjust_learning_rate
from fedtorch.components.dataset_builder import load_data_batch
from fedtorch.comms.utils.flow_utils import (get_current_epoch, 
                                             get_current_local_step, 
                                             is_sync_fed,
                                             is_sync_fed_robust,
                                             projection_simplex_tensor)
from fedtorch.comms.utils.eval import inference
from fedtorch.comms.utils.eval_centered import (do_validate_centered, 
                                                log_validation_centered,
                                                log_test_centered)
from fedtorch.comms.algorithms.federated import (set_online_clients_centered,
                                                 fedavg_aggregation_centered,
                                                 robust_noise_average,
                                                 calc_clients_coefficient_centered,
                                                 fedshuffle_aggregation_centered)
from fedtorch.logs.logging import (log, 
                                   logging_sync_time,
                                   logging_load_time,
                                   logging_globally)
from fedtorch.logs.meter import define_local_training_tracker
from .main import train_and_validate_federated_centered

def train_and_validate_sgdap_centered(Clients, Server):
    log('start training and validation with Federated setting in a centered way.')

    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.')

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
        
        for oc in online_clients:
            Clients[oc].model.load_state_dict(Server.model.state_dict())
            Clients[oc].cfg.rounds_comm = Server.cfg.rounds_comm
            local_steps = 0
            is_sync = False

            do_validate_centered(Clients[oc].cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                 Clients[oc].train_loader, Server.global_val_tracker, val=False, local=False)
            if Server.cfg.federated.personal:
                do_validate_centered(Clients[oc].cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                    Clients[oc].val_loader, Server.global_personal_val_tracker, val=True, local=False)

            while not is_sync:
                if Server.cfg.model.type == 'rnn':
                    Clients[oc].model.init_hidden(Server.cfg.data.batch_size)
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

                    # compute gradient and do local SGD step.
                    loss.backward(retain_graph=True)
                    if getattr(Clients[oc].cfg.model, 'robust', False):
                        Server.avg_noise_model.noise.data = Clients[oc].model.noise.data
                        Server.avg_noise_optimizer.zero_grad()
                        loss_noise, _ = inference(Server.avg_noise_model, Clients[oc].criterion, Clients[oc].metrics, _input, _target)
                        if Server.avg_noise_model.noise.grad is None:
                            loss_noise.backward(retain_graph=True)
                            Server.avg_noise_optimizer.zero_grad()
                            loss_noise, _ = inference(Server.avg_noise_model, Clients[oc].criterion, Clients[oc].metrics, _input, _target)
                        Clients[oc].model.noise.grad.data = -1 * torch.autograd.grad(loss_noise, Server.avg_noise_model.noise)[0]
                        
 

                    Clients[oc].optimizer.step(
                        apply_lr=True,
                        apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                    )

                    if getattr(Clients[oc].cfg.model, 'robust', False):
                        if torch.norm(Clients[oc].model.noise.data) > 1:
                            Clients[oc].model.noise.data /= torch.norm(Clients[oc].model.noise.data)

                    # reset load time for the tracker.
                    tracker['start_load_time'] = time.time()
                    # model_local = deepcopy(model_client)
                    is_sync = is_sync_fed(Clients[oc].cfg)
                    if is_sync:
                        break



            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].train_loader, Server.local_val_tracker, val=False, local=True)
            if Server.cfg.federated.personal:
                do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].val_loader, Server.local_personal_val_tracker, val=True, local=True)
            # Sync the model server based on model_clients
            tracker['start_sync_time'] = time.time()
            Server.cfg.global_index += 1

            logging_sync_time(tracker)
        
        if is_sync_fed_robust(Clients[0].cfg):
            robust_noise_average(Clients, Server, online_clients)
        fedavg_aggregation_centered(Clients, Server, online_clients)

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

        # Test on server
        do_validate_centered(Server.cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
                 Server.test_loader, Server.global_test_tracker, val=False, local=False)
        log_test_centered(Server.cfg,Server.global_test_tracker)


        # logging.
        logging_globally(tracker, start_global_time)
        
        # reset start round time.
        start_global_time = time.time()

    return


def train_and_validate_fedshuffle_centered(Clients, Server):
    log('Start training with FedAvg in a centered way to find clients coefficients.')
    n_c = Server.cfg.federated.num_comms
    server_model = deepcopy(Server.model.state_dict())
    if hasattr(Server.cfg.federated,'num_comms_warmup'):
        Server.cfg.federated.num_comms = Server.cfg.federated.num_comms_warmup
    else:
        Server.cfg.federated.num_comms /= 5
    Server.cfg.federated.type= 'fedavg'
    train_and_validate_federated_centered(Clients, Server)
    # Server.alphas = torch.nn.functional.softmax(
    #     calc_clients_coefficient_centered(Clients,Server) *  Server.cfg.graph.n_nodes,
    #     dim=1
    # )
    # Server.alphas = projection_simplex_tensor(calc_clients_coefficient_centered(Clients,Server))
    Server.alphas = calc_clients_coefficient_centered(Clients,Server)
    torch.save(Server.alphas, os.path.join(Server.cfg.work_dir,'alpha.pt'))
    # Server.alphas = torch.ones(Server.cfg.graph.n_nodes,Server.cfg.graph.n_nodes)/Server.cfg.graph.n_nodes
    log('The learned alphas for clients are:\n {}'.format(Server.alphas))
    Server.model.load_state_dict(server_model)
    Server.cfg.federated.num_comms = n_c
    log('start training and validation with FedShuffle setting in a centered way.')

    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    log('enter the training.')

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
        
        for oc in online_clients:
            # Clients[oc].val_tracker = define_val_tracker()
            # if n_c==0:
            #     Clients[oc].enable_grad(Clients[oc].model_personal, Clients[oc].optimizer_personal, Clients[oc].criterion,  Clients[oc].train_loader)
            model_loc = Server.shuffle_list[oc].item()
            Clients[oc].model.load_state_dict(Clients[model_loc].model_personal.state_dict())
            Clients[oc].cfg.rounds_comm = Server.cfg.rounds_comm
            local_steps = 0
            is_sync = False

            do_validate_centered(Clients[oc].cfg, Clients[oc].model_personal, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].val_loader, Server.global_val_tracker, val=True, local=False)
            # do_validate_centered(Clients[model_loc].cfg, Clients[oc].model, Clients[model_loc].criterion, Clients[model_loc].metrics, Clients[model_loc].optimizer,
            #      Clients[model_loc].val_loader, Clients[oc].val_tracker, val=True, local=True)
            if Server.alphas[oc,model_loc] < 0:
                continue
            # adjust learning rate (based on the # of accessed samples)
            lr = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer, Clients[oc].scheduler)
            while not is_sync:
                # self_update = (Server.shuffle_list == torch.arange(Server.cfg.graph.n_nodes)).all()
                if Server.cfg.model.type == 'rnn':
                    Clients[oc].model.init_hidden(Server.cfg.data.batch_size)
                    Clients[oc].model_personal.init_hidden(Server.cfg.data.batch_size)
                for _input, _target in Clients[oc].train_loader:
                    local_steps += 1
                    Clients[oc].model.train()
                    Clients[oc].model_personal.train()

                    # update local step.
                    logging_load_time(tracker)
                    
                    # update local index and get local step
                    Clients[oc].cfg.local_index += 1
                    Clients[oc].cfg.local_data_seen += len(_target)
                    get_current_epoch(Clients[oc].cfg)
                    local_step = get_current_local_step(Clients[oc].cfg)

                    

                    # load data
                    _input, _target = load_data_batch(Clients[oc].cfg, _input, _target, tracker)

                    if n_c == 0:
                        # enable gradient for the personal model to be able to update it later
                        Clients[oc].optimizer_personal.zero_grad()
                        output = Clients[oc].model_personal(_input)
                        loss_personal = Clients[oc].criterion(output, _target)
                        loss_personal.backward()
                        Clients[oc].optimizer_personal.zero_grad()
                    # Skip batches with one sample because of BatchNorm issue in some models!
                    if _input.size(0)==1:
                        is_sync = is_sync_fed(Clients[oc].cfg)
                        break

                    # inference and get current performance.
                    # For the first round each model only works on its personal model
                    # from n_c=1 the personal models are updated at the aggregation time
                    # opt = Clients[oc].optimizer_personal if self_update else Clients[oc].optimizer
                    # model = Clients[oc].model_personal if self_update else Clients[oc].model
                    
                    opt = Clients[oc].optimizer
                    model = Clients[oc].model

                    opt.zero_grad()
                    
                    loss, performance = inference(model, Clients[oc].criterion, Clients[oc].metrics, 
                                                    _input, _target, rnn=Server.cfg.model.type in ['rnn'])

                    # compute gradient and do local SGD step.
                    loss.backward()


                    opt.step(
                        apply_lr=True,
                        apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                    )

                    # reset load time for the tracker.
                    tracker['start_load_time'] = time.time()
                    # model_local = deepcopy(model_client)
                    is_sync = is_sync_fed(Clients[oc].cfg)
                    if is_sync:
                        break


            
            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[model_loc].train_loader, Server.local_val_tracker, val=False, local=True)
            # do_validate_centered(Clients[model_loc].cfg, Clients[oc].model, Clients[model_loc].criterion, Clients[model_loc].metrics, Clients[model_loc].optimizer,
            #      Clients[model_loc].val_loader, Clients[oc].val_tracker, val=True, local=True)
            # Sync the model server based on model_clients
            tracker['start_sync_time'] = time.time()
            Server.cfg.global_index += 1
            # Clients[model_loc].model_personal.load_state_dict(Clients[oc].model.state_dict())
            logging_sync_time(tracker)
        

        fedshuffle_aggregation_centered(Clients, Server, online_clients)

        
        # Log performance
        # Client training performance
        log_validation_centered(Server.cfg, Server.local_val_tracker, val=False, local=True)
        # Client test performance
        log_validation_centered(Server.cfg, Server.global_val_tracker, val=True, local=True)
        # val_data = {}
        # for oc in online_clients:
        #     val_data[str(oc)] = Clients[oc].val_tracker['top1'].avg
        # Server.cfg.tb_writer.add_scalars('validation/Personal/Acc', val_data, n_c)

        # Test on server
        # do_validate_centered(Server.cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
        #          Server.test_loader, Server.global_test_tracker, val=False, local=False)
        log_test_centered(Server.cfg,Server.global_test_tracker)


        # logging.
        logging_globally(tracker, start_global_time)
        
        # reset start round time.
        start_global_time = time.time()

    return



def train_and_validate_fedshuffle_one_centered(Clients, Server):
    log('start training and validation with FedShuffle one stage setting in a centered way.')
    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    if not hasattr(Server.cfg.federated,'num_comms_warmup'):
        Server.cfg.federated.num_comms_warmup = Server.cfg.federated.num_comms / 5
        
    log('enter the training.')


    # Number of communication rounds in federated setting should be defined
    for n_c in range(Server.cfg.federated.num_comms):
        Server.cfg.rounds_comm += 1
        Server.cfg.local_index += 1
        Server.cfg.quant_error = 0.0
        
        # Preset variables for this round of communication
        Server.enable_grad(Clients[0].val_loader)
        Server.reset_tracker(Server.local_val_tracker)
        Server.reset_tracker(Server.global_val_tracker)
        Server.reset_tracker(Server.global_test_tracker)
        if Server.cfg.federated.personal:
            Server.reset_tracker(Server.local_personal_val_tracker)
            Server.reset_tracker(Server.global_personal_val_tracker) 

        # Configuring the devices for this round of communication
        log("Starting round {} of training".format(n_c+1))
        online_clients = set_online_clients_centered(Server.cfg)
        
        for oc in online_clients:
            Clients[oc].model.load_state_dict(Server.model.state_dict())
            per_model_id = Server.shuffle_list[oc].item()
            if n_c == Server.cfg.federated.num_comms_warmup:
                Server.personal_models[oc].load_state_dict(Clients[oc].model.state_dict())
            elif n_c > Server.cfg.federated.num_comms_warmup:
                Clients[oc].model_personal.load_state_dict(Server.personal_models[per_model_id].state_dict())
            Clients[oc].cfg.rounds_comm = Server.cfg.rounds_comm
            local_steps = 0
            is_sync = False

            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].val_loader, Server.global_val_tracker, val=True, local=False)
            if Server.cfg.federated.personal:
                do_validate_centered(Clients[oc].cfg, Server.personal_models[oc],  Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                    Clients[oc].val_loader, Server.global_personal_val_tracker, val=True)
            while not is_sync:
                if Server.cfg.model.type == 'rnn':
                    Clients[oc].model.init_hidden(Server.cfg.data.batch_size)
                    Clients[oc].model_personal.init_hidden(Server.cfg.data.batch_size)
                for _input, _target in Clients[oc].train_loader:
                    local_steps += 1
                    Clients[oc].model.train()
                    Clients[oc].model_personal.train()

                    # update local step.
                    logging_load_time(tracker)
                    
                    # update local index and get local step
                    Clients[oc].cfg.local_index += 1
                    Clients[oc].cfg.local_data_seen += len(_target)
                    get_current_epoch(Clients[oc].cfg)
                    local_step = get_current_local_step(Clients[oc].cfg)

                    # adjust learning rate (based on the # of accessed samples)
                    lr = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer, Clients[oc].scheduler)
                    lr_personal = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer_personal, Clients[oc].scheduler)

                    # load data
                    _input, _target = load_data_batch(Clients[oc].cfg, _input, _target, tracker)
        
                    # Skip batches with one sample because of BatchNorm issue in some models!
                    if _input.size(0)==1:
                        is_sync = is_sync_fed(Clients[oc].cfg)
                        break

                    # inference and get current performance.
                    # For the first round each model only works on its personal model
                    # from n_c=1 the personal models are updated at the aggregation time
                    # opt = Clients[oc].optimizer_personal if self_update else Clients[oc].optimizer
                    # model = Clients[oc].model_personal if self_update else Clients[oc].model
                    # cp = deepcopy(Clients[oc].model_personal)
                    Clients[oc].optimizer_personal.zero_grad()
                    Clients[oc].optimizer.zero_grad()
                    
                    loss, _ = inference(Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, 
                                                    _input, _target, rnn=Server.cfg.model.type in ['rnn'])
                    # compute gradient and do local SGD step.
                    loss.backward()

                    if (Server.alphas[oc, per_model_id] > 0) and (n_c > Server.cfg.federated.num_comms_warmup):
                        loss_personal, _  = inference(Clients[oc].model_personal, Clients[oc].criterion, Clients[oc].metrics, 
                                                        _input, _target, rnn=Server.cfg.model.type in ['rnn'])
                        loss_personal.backward()
                        Clients[oc].optimizer_personal.step(
                                scale=Server.alphas[oc, per_model_id],
                                apply_lr=True,
                                apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                            )
                    
                    


                    
                    Clients[oc].optimizer.step(
                        apply_lr=True,
                        apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                    )
                    # models_eq = chk_models(cp,Clients[oc].model_personal )
                    # if not models_eq:
                    #     print("Models are not the same!", loss_personal)   
                    # reset load time for the tracker.
                    tracker['start_load_time'] = time.time()
                    # model_local = deepcopy(model_client)
                    is_sync = is_sync_fed(Clients[oc].cfg)
                    if is_sync:
                        # saving the personal model in the server
                        # models_eq = chk_models(Server.personal_models[per_model_id],Clients[oc].model_personal )
                        # if models_eq:
                        #     print("Models are the same!")
                        if n_c > Server.cfg.federated.num_comms_warmup:
                            Server.personal_models[per_model_id].load_state_dict(Clients[oc].model_personal.state_dict())
                        break



            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer_personal,
                 Clients[oc].train_loader, Server.local_val_tracker, val=False, local=True)

            # Sync the model server based on model_clients
            tracker['start_sync_time'] = time.time()
            Server.cfg.global_index += 1

            logging_sync_time(tracker)
        

        # fedshuffle_aggregation_centered(Clients, Server, online_clients)
        fedavg_aggregation_centered(Clients, Server, online_clients)
        # Server.alphas =  torch.nn.functional.softmax(
        #                     calc_clients_coefficient_centered(Clients,Server) * Server.cfg.graph.n_nodes, 
        #                     dim=1
        #                 )
        Server.alphas = projection_simplex_tensor(calc_clients_coefficient_centered(Clients,Server))
        # log('The learned alphas for clients are:\n {}'.format(Server.alphas))
        Server.shuffle_list = Server.shuffle_list.roll(1)
        # Log performance
        # Client training performance
        log_validation_centered(Server.cfg, Server.local_val_tracker, val=False, local=True)
        # Client test performance
        log_validation_centered(Server.cfg, Server.global_val_tracker, val=True, local=True)
        log_validation_centered(Server.cfg, Server.global_personal_val_tracker, val=True, personal=True, model_name="Personal")
        

        # Test on server
        # do_validate_centered(Server.cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
        #          Server.test_loader, Server.global_test_tracker, val=False, local=False)
        # log_test_centered(Server.cfg,Server.global_test_tracker)


        # logging.
        logging_globally(tracker, start_global_time)
        
        # reset start round time.
        start_global_time = time.time()

    log('The learned alphas for clients are:\n {}'.format(Server.alphas))
    torch.save(Server.alphas, os.path.join(Server.cfg.work_dir,'alpha.pt'))
    return

def train_and_validate_fedrolexs_centered(Clients, Server):
    log('start training and validation with FedShuffle one stage setting in a centered way.')
    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    if not hasattr(Server.cfg.federated,'num_comms_warmup'):
        Server.cfg.federated.num_comms_warmup = Server.cfg.federated.num_comms / 5
        
    log('enter the training.')


    # Number of communication rounds in federated setting should be defined
    for n_c in range(Server.cfg.federated.num_comms):
        Server.cfg.rounds_comm += 1
        Server.cfg.local_index += 1
        Server.cfg.quant_error = 0.0
        
        # Preset variables for this round of communication
        Server.enable_grad(Clients[0].val_loader)
        Server.reset_tracker(Server.local_val_tracker)
        Server.reset_tracker(Server.global_val_tracker)
        Server.reset_tracker(Server.global_test_tracker)
        if Server.cfg.federated.personal:
            Server.reset_tracker(Server.local_personal_val_tracker)
            Server.reset_tracker(Server.global_personal_val_tracker) 

        # Configuring the devices for this round of communication
        log("Starting round {} of training".format(n_c+1))
        online_clients = set_online_clients_centered(Server.cfg)
        
        for oc in online_clients:
            Clients[oc].model.load_state_dict(Server.model.state_dict())
            per_model_id = Server.shuffle_list[oc].item()
            if n_c == Server.cfg.federated.num_comms_warmup:
                Server.personal_models[oc].load_state_dict(Clients[oc].model.state_dict())
            elif n_c > Server.cfg.federated.num_comms_warmup:
                Clients[oc].model_personal.load_state_dict(Server.personal_models[per_model_id].state_dict())
            Clients[oc].cfg.rounds_comm = Server.cfg.rounds_comm
            local_steps = 0
            is_sync = False

            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                 Clients[oc].val_loader, Server.global_val_tracker, val=True, local=False)
            if Server.cfg.federated.personal:
                do_validate_centered(Clients[oc].cfg, Server.personal_models[oc],  Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                    Clients[oc].val_loader, Server.global_personal_val_tracker, val=True)
            while not is_sync:
                if Server.cfg.model.type == 'rnn':
                    Clients[oc].model.init_hidden(Server.cfg.data.batch_size)
                    Clients[oc].model_personal.init_hidden(Server.cfg.data.batch_size)
                for _input, _target in Clients[oc].train_loader:
                    local_steps += 1
                    Clients[oc].model.train()
                    Clients[oc].model_personal.train()

                    # update local step.
                    logging_load_time(tracker)
                    
                    # update local index and get local step
                    Clients[oc].cfg.local_index += 1
                    Clients[oc].cfg.local_data_seen += len(_target)
                    get_current_epoch(Clients[oc].cfg)
                    local_step = get_current_local_step(Clients[oc].cfg)

                    # adjust learning rate (based on the # of accessed samples)
                    lr = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer, Clients[oc].scheduler)
                    lr_personal = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer_personal, Clients[oc].scheduler)

                    # load data
                    _input, _target = load_data_batch(Clients[oc].cfg, _input, _target, tracker)
        
                    # Skip batches with one sample because of BatchNorm issue in some models!
                    if _input.size(0)==1:
                        is_sync = is_sync_fed(Clients[oc].cfg)
                        break

                    # inference and get current performance.
                    # For the first round each model only works on its personal model
                    # from n_c=1 the personal models are updated at the aggregation time
                    # opt = Clients[oc].optimizer_personal if self_update else Clients[oc].optimizer
                    # model = Clients[oc].model_personal if self_update else Clients[oc].model
                    # cp = deepcopy(Clients[oc].model_personal)
                    Clients[oc].optimizer_personal.zero_grad()
                    Clients[oc].optimizer.zero_grad()
                    
                    loss, _ = inference(Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, 
                                                    _input, _target, rnn=Server.cfg.model.type in ['rnn'])
                    # compute gradient and do local SGD step.
                    loss.backward()

                    if (Server.alphas[oc, per_model_id] > 0) and (n_c > Server.cfg.federated.num_comms_warmup):
                        loss_personal, _  = inference(Clients[oc].model_personal, Clients[oc].criterion, Clients[oc].metrics, 
                                                        _input, _target, rnn=Server.cfg.model.type in ['rnn'])
                        loss_personal.backward()
                        Clients[oc].optimizer_personal.step(
                                scale=Server.alphas[oc, per_model_id],
                                apply_lr=True,
                                apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                            )
                    
                    


                    
                    Clients[oc].optimizer.step(
                        apply_lr=True,
                        apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                    )
                    # models_eq = chk_models(cp,Clients[oc].model_personal )
                    # if not models_eq:
                    #     print("Models are not the same!", loss_personal)   
                    # reset load time for the tracker.
                    tracker['start_load_time'] = time.time()
                    # model_local = deepcopy(model_client)
                    is_sync = is_sync_fed(Clients[oc].cfg)
                    if is_sync:
                        # saving the personal model in the server
                        # models_eq = chk_models(Server.personal_models[per_model_id],Clients[oc].model_personal )
                        # if models_eq:
                        #     print("Models are the same!")
                        if n_c > Server.cfg.federated.num_comms_warmup:
                            Server.personal_models[per_model_id].load_state_dict(Clients[oc].model_personal.state_dict())
                        break



            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer_personal,
                 Clients[oc].train_loader, Server.local_val_tracker, val=False, local=True)

            # Sync the model server based on model_clients
            tracker['start_sync_time'] = time.time()
            Server.cfg.global_index += 1

            logging_sync_time(tracker)
        

        # fedshuffle_aggregation_centered(Clients, Server, online_clients)
        fedavg_aggregation_centered(Clients, Server, online_clients)
        # Server.alphas =  torch.nn.functional.softmax(
        #                     calc_clients_coefficient_centered(Clients,Server) * Server.cfg.graph.n_nodes, 
        #                     dim=1
        #                 )
        Server.alphas = projection_simplex_tensor(calc_clients_coefficient_centered(Clients,Server))
        # log('The learned alphas for clients are:\n {}'.format(Server.alphas))
        Server.shuffle_list = Server.shuffle_list.roll(1)
        # Log performance
        # Client training performance
        log_validation_centered(Server.cfg, Server.local_val_tracker, val=False, local=True)
        # Client test performance
        log_validation_centered(Server.cfg, Server.global_val_tracker, val=True, local=True)
        log_validation_centered(Server.cfg, Server.global_personal_val_tracker, val=True, personal=True, model_name="Personal")
        

        # Test on server
        # do_validate_centered(Server.cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
        #          Server.test_loader, Server.global_test_tracker, val=False, local=False)
        # log_test_centered(Server.cfg,Server.global_test_tracker)


        # logging.
        logging_globally(tracker, start_global_time)
        
        # reset start round time.
        start_global_time = time.time()

    log('The learned alphas for clients are:\n {}'.format(Server.alphas))
    torch.save(Server.alphas, os.path.join(Server.cfg.work_dir,'alpha.pt'))
    return


def train_and_validate_fedshuffle_one_centered_ray(Clients, Server):
    ray.init()
    log('start training and validation with FedShuffle one stage setting in a centered way.')
    tracker = define_local_training_tracker()
    start_global_time = time.time()
    tracker['start_load_time'] = time.time()
    if not hasattr(Server.cfg.federated,'num_comms_warmup'):
        Server.cfg.federated.num_comms_warmup = Server.cfg.federated.num_comms / 5
        
    log('enter the training.')

    @ray.remote
    def client_update(oc):
        Clients[oc].model.load_state_dict(Server.model.state_dict())
        per_model_id = Server.shuffle_list[oc].item()
        if n_c == Server.cfg.federated.num_comms_warmup:
            Server.personal_models[oc].load_state_dict(Clients[oc].model.state_dict())
        elif n_c > Server.cfg.federated.num_comms_warmup:
            Clients[oc].model_personal.load_state_dict(Server.personal_models[per_model_id].state_dict())
        Clients[oc].cfg.rounds_comm = Server.cfg.rounds_comm
        local_steps = 0
        is_sync = False

        do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                Clients[oc].val_loader, Server.global_val_tracker, val=True, local=False)
        if Server.cfg.federated.personal:
            do_validate_centered(Clients[oc].cfg, Server.personal_models[oc],  Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer,
                Clients[oc].val_loader, Server.global_personal_val_tracker, val=True)
        while not is_sync:
            if Server.cfg.model.type == 'rnn':
                Clients[oc].model.init_hidden(Server.cfg.data.batch_size)
                Clients[oc].model_personal.init_hidden(Server.cfg.data.batch_size)
            for _input, _target in Clients[oc].train_loader:
                local_steps += 1
                Clients[oc].model.train()
                Clients[oc].model_personal.train()

                # update local step.
                logging_load_time(tracker)
                
                # update local index and get local step
                Clients[oc].cfg.local_index += 1
                Clients[oc].cfg.local_data_seen += len(_target)
                get_current_epoch(Clients[oc].cfg)
                local_step = get_current_local_step(Clients[oc].cfg)

                # adjust learning rate (based on the # of accessed samples)
                lr = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer, Clients[oc].scheduler)
                lr_personal = adjust_learning_rate(Clients[oc].cfg, Clients[oc].optimizer_personal, Clients[oc].scheduler)

                # load data
                _input, _target = load_data_batch(Clients[oc].cfg, _input, _target, tracker)
    
                # Skip batches with one sample because of BatchNorm issue in some models!
                if _input.size(0)==1:
                    is_sync = is_sync_fed(Clients[oc].cfg)
                    break

                # inference and get current performance.
                # For the first round each model only works on its personal model
                # from n_c=1 the personal models are updated at the aggregation time
                # opt = Clients[oc].optimizer_personal if self_update else Clients[oc].optimizer
                # model = Clients[oc].model_personal if self_update else Clients[oc].model
                # cp = deepcopy(Clients[oc].model_personal)
                Clients[oc].optimizer_personal.zero_grad()
                Clients[oc].optimizer.zero_grad()
                
                loss, _ = inference(Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, 
                                                _input, _target, rnn=Server.cfg.model.type in ['rnn'])
                # compute gradient and do local SGD step.
                loss.backward()

                if (Server.alphas[oc, per_model_id] > 0) and (n_c > Server.cfg.federated.num_comms_warmup):
                    loss_personal, _  = inference(Clients[oc].model_personal, Clients[oc].criterion, Clients[oc].metrics, 
                                                    _input, _target, rnn=Server.cfg.model.type in ['rnn'])
                    loss_personal.backward()
                    Clients[oc].optimizer_personal.step(
                            scale=Server.alphas[oc, per_model_id],
                            apply_lr=True,
                            apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                        )
                
                


                
                Clients[oc].optimizer.step(
                    apply_lr=True,
                    apply_in_momentum=Clients[oc].cfg.training.in_momentum, apply_out_momentum=False
                )
                # models_eq = chk_models(cp,Clients[oc].model_personal )
                # if not models_eq:
                #     print("Models are not the same!", loss_personal)   
                # reset load time for the tracker.
                tracker['start_load_time'] = time.time()
                # model_local = deepcopy(model_client)
                is_sync = is_sync_fed(Clients[oc].cfg)
                if is_sync:
                    # saving the personal model in the server
                    # models_eq = chk_models(Server.personal_models[per_model_id],Clients[oc].model_personal )
                    # if models_eq:
                    #     print("Models are the same!")
                    if n_c > Server.cfg.federated.num_comms_warmup:
                        Server.personal_models[per_model_id].load_state_dict(Clients[oc].model_personal.state_dict())
                    return
        return

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

        for oc in online_clients:
            client_update.remote(oc)



            do_validate_centered(Clients[oc].cfg, Clients[oc].model, Clients[oc].criterion, Clients[oc].metrics, Clients[oc].optimizer_personal,
                 Clients[oc].train_loader, Server.local_val_tracker, val=False, local=True)

            # Sync the model server based on model_clients
            tracker['start_sync_time'] = time.time()
            Server.cfg.global_index += 1

            logging_sync_time(tracker)
        

        # fedshuffle_aggregation_centered(Clients, Server, online_clients)
        fedavg_aggregation_centered(Clients, Server, online_clients)
        # Server.alphas =  torch.nn.functional.softmax(
        #                     calc_clients_coefficient_centered(Clients,Server) * Server.cfg.graph.n_nodes, 
        #                     dim=1
        #                 )
        Server.alphas = projection_simplex_tensor(calc_clients_coefficient_centered(Clients,Server))
        # log('The learned alphas for clients are:\n {}'.format(Server.alphas))
        Server.shuffle_list = Server.shuffle_list.roll(1)
        # Log performance
        # Client training performance
        log_validation_centered(Server.cfg, Server.local_val_tracker, val=False, local=True)
        # Client test performance
        log_validation_centered(Server.cfg, Server.global_val_tracker, val=True, local=True)
        log_validation_centered(Server.cfg, Server.global_personal_val_tracker, val=True, personal=True, model_name="Personal")
        

        # Test on server
        # do_validate_centered(Server.cfg, Server.model, Server.criterion, Server.metrics, Server.optimizer,
        #          Server.test_loader, Server.global_test_tracker, val=False, local=False)
        # log_test_centered(Server.cfg,Server.global_test_tracker)


        # logging.
        logging_globally(tracker, start_global_time)
        
        # reset start round time.
        start_global_time = time.time()

    log('The learned alphas for clients are:\n {}'.format(Server.alphas))
    return