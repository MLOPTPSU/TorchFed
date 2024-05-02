# -*- coding: utf-8 -*-
import platform
from copy import deepcopy,copy

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from fedtorch.components.comps import create_components
from fedtorch.components.optimizer_builder import build_optimizer_from_config
from fedtorch.utils.init_config import init_config
from fedtorch.components.dataset_builder import build_dataset_from_config, build_dataset
from fedtorch.comms.utils.flow_utils import zero_copy
from fedtorch.logs.logging import log, configure_log, log_cfgs
from fedtorch.logs.meter import define_val_tracker
from fedtorch.comms.algorithms.distributed import configure_sync_scheme

class Node():
    def __init__(self, rank):
        self.rank = rank

    def initialize(self):
        pass

    def reset_tracker(self, tracker):
        for k in tracker.keys():
            tracker[k].reset()

class Client(Node):
    def __init__(self, cfg, rank):
        super(Client, self).__init__(rank)
        self.cfg = copy(cfg)

        # Initialize the node
        # self.initialize()
        # Initialize the dataset if not downloaded
        # self.initialize_dataset()
        # Load the dataset
        # self.load_local_dataset()
        # Generate auxiliary models
        # self.gen_aux_models()

    def initialize(self):
        init_config(self.cfg)
        if self.cfg.graph.rank == 0:
            self.cfg.tb_writer = SummaryWriter(log_dir=self.cfg.work_dir) 
        self.model, self.criterion, self.scheduler, self.optimizer, self.metrics = create_components(self.cfg)
        self.cfg.finish_one_epoch = False
        # Create a model server on each client to keep a copy of the server model at each communication round.
        self.model_server = deepcopy(self.model)

        configure_log(self.cfg)
        log_cfgs(self.cfg, debug=self.cfg.graph.debug)
        log(
        'Rank {} with block {} on {} {}-{}'.format(
            self.cfg.graph.rank,
            self.cfg.graph.ranks_with_blocks[self.cfg.graph.rank],
            platform.node(),
            'GPU' if (self.cfg.device.on_cuda and torch.cuda.is_available()) else 'CPU',
            self.cfg.graph.device
            ),
        debug=self.cfg.graph.debug)

        self.all_clients_group = dist.new_group(self.cfg.graph.ranks)

    def initialize_dataset(self):
        # If the data is not downloaded, for the first time the server only needs to download the data
        if self.cfg.graph.rank == 0:
            data_loader = build_dataset_from_config(self.cfg, split='train')
            del data_loader
            data_loader = get_dataset(self.args, self.args.data, self.args.data_dir, split='test')
            del data_loader
        dist.barrier(group=self.all_clients_group)

    def load_local_dataset(self):
        load_test = True if self.cfg.graph.rank ==0 else False
        if hasattr(self.cfg,'federated') and self.cfg.federated.personal:
            self.train_loader, self.test_loader, self.val_loader= build_dataset(self.cfg, test=load_test)
        else:
            self.train_loader, self.test_loader = build_dataset(self.cfg, test=load_test)
        
        # if self.cfg.data in ['mnist','fashion_mnist','cifar10', 'cifar100']:
        #     self.cfg.classes = torch.arange(10)
        # elif self.cfg.data in ['synthetic']:
        #     self.cfg.classes = torch.arange(5)
        # elif self.cfg.data in ['adult']:
        #     self.cfg.classes = torch.arange(2)

    def gen_aux_models(self):
        if self.cfg.training.type == 'federated':
            if self.cfg.federated.type == 'fedgate':
                self.model_delta = zero_copy(self.model)
                self.model_memory = zero_copy(self.model)
            elif self.cfg.federated.type == 'qsparse':
                self.model_memory = zero_copy(self.model)
            elif self.cfg.federated.type == 'scaffold':
                self.model_client_control = zero_copy(self.model)
                self.model_server_control = zero_copy(self.model)
            elif self.cfg.federated.type == 'fedadam':
                # Initialize the parameter for FedAdam https://arxiv.org/abs/2003.00295
                self.cfg.federated.fedadam_v =  [self.cfg.federated.fedadam_tau ** 2] * len(list(self.model.parameters()))
            elif self.cfg.federated.type == 'apfl':
                self.model_personal = deepcopy(self.model)
                self.optimizer_personal = build_optimizer_from_config(self.cfg.optimizer, self.model_personal, self.cfg.graph.n_nodes, self.cfg.lr.lr)
            elif self.cfg.federated.type == 'afl':
                self.lambda_vector = torch.zeros(self.cfg.graph.n_nodes)
            elif self.cfg.federated.type == 'perfedme':
                self.model_personal = deepcopy(self.model)
                self.optimizer_personal = build_optimizer_from_config(self.cfg.optimizer, self.model_personal, self.cfg.graph.n_nodes, self.cfg.lr.lr)
            elif self.cfg.federated.type == 'qffl':
                self.full_loss = 0.0
            if self.cfg.training.drfa:
                self.kth_model  = zero_copy(self.model)
                self.lambda_vector = torch.zeros(self.cfg.graph.n_nodes)
    
    def zero_avg(self):
        self.model_avg = zero_copy(self.model)
        self.model_avg_tmp = zero_copy(self.model)