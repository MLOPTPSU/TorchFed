# -*- coding: utf-8 -*-
import time

import torch
import numpy as np

from fedtorch.logs.logging import log
from fedtorch.utils import Registry, build

DATASET = Registry("dataset")
PARTITIONER = Registry("partitioner")

from .datasets import *

def build_dataset_from_config(cfg, split='train'):
    """Build dataset from config file."""
    cfg.data.dataset.split = split
    if cfg.data.dataset.type in ['emnist', 'synthetic', 'synthetic_polar', 'shakespeare']:
        cfg.data.dataset.client_id = cfg.graph.rank
    if cfg.data.dataset.type in ['synthetic', 'synthetic_polar']:
        cfg.data.dataset.num_tasks = cfg.graph.n_nodes
    return build(cfg.data.dataset, DATASET)


def build_dataset(cfg, test=True, Partitioner=None, return_partitioner=False):
    log('create {} dataset for rank {}'.format(cfg.data.dataset.type, cfg.graph.rank), cfg.graph.debug)

    train_loader = partition_dataset(cfg, dataset_type='train', 
                            Partitioner=Partitioner, return_partitioner=return_partitioner)
    if return_partitioner:
        train_loader, Partitioner = train_loader
    fed_personal = (getattr(cfg, 'federated', None) is not None) and (cfg.federated.personal)
    if fed_personal:
        if cfg.federated.type == 'perfedavg':
            train_loader, val_loader, val_loader1 = train_loader
        else:
            train_loader, val_loader = train_loader
    if test:
        test_loader = partition_dataset(cfg, dataset_type='test')
    else:
        test_loader=None
    get_data_stat(cfg, train_loader, test_loader)
    if fed_personal:
        if cfg.federated.type == 'perfedavg':
            out = [train_loader, test_loader, val_loader, val_loader1]
        else:
            out = [train_loader, test_loader, val_loader]
    else:
        out = [train_loader, test_loader]
    if return_partitioner:
        out = (out, Partitioner)
    return out

def build_partitioner_from_config(data, cfg, return_partitioner=False):
    """Build partitioner from config file."""
    partition_sizes = [1.0 / cfg.graph.n_nodes for _ in range(cfg.graph.n_nodes)]
    cfg.partitioner.sizes = partition_sizes
    cfg.partitioner.graph_cfg = cfg.graph
    cfg.partitioner.data = data
    if cfg.partitioner.type == 'GrowingBatchPartitioner':
        cfg.partitioner.num_epochs = cfg.training.num_epochs
    elif cfg.partitioner.type == 'FederatedPartitioner':
        cfg.partitioner.dataset_name = cfg.data.dataset.type
        cfg.partitioner.num_classes = cfg.data.dataset.num_classes
    partition = build(cfg.partitioner, PARTITIONER)
    if return_partitioner:
        return partition.use(cfg.graph.rank), partition
    else:
        return partition.use(cfg.graph.rank)

def _load_data_batch(cfg, _input, _target):
    if 'least_square' in cfg.model.type:
        _input = _input.float()
        _target = _target.unsqueeze_(1).float()
    else:
        if cfg.data.dataset.type in ['epsilon', 'url', 'rcv1', 'higgs']:
            _input, _target = _input.float(), _target.long()

    if cfg.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()
    return _input, _target


def load_data_batch(cfg, _input, _target, tracker):
    """Load a mini-batch and record the loading time."""
    # get variables.
    start_data_time = time.time()

    _input, _target = _load_data_batch(cfg, _input, _target)

    # measure the data loading time
    end_data_time = time.time()
    tracker['data_time'].update(end_data_time - start_data_time)
    tracker['end_data_time'] = end_data_time
    return _input, _target




def partition_dataset(cfg, dataset_type, Partitioner=None, return_partitioner=False):
    cfg.partitioner, cfg.data, cfg.graph = cfg.partitioner, cfg.data, cfg.graph
    """ Given a dataset, partition it. """
    if Partitioner is None:
        dataset = build_dataset_from_config(cfg, split=dataset_type)
    else:
        dataset = Partitioner.data

    # partition data.
    if dataset_type == 'train':
        if cfg.partitioner.type=='DataPartitioner':
            if cfg.data.dataset.type in ['emnist', 'emnist_full','synthetic','shakespeare']:
                raise ValueError('The dataset {} does not have a structure for iid distribution of data'.format(cfg.data.dataset.type))

        elif cfg.partitioner.type=='FederatedPartitioner':
            if cfg.data.dataset.type not in ['mnist','fashion_mnist','emnist', 'emnist_full','cifar10','cifar100','adult','synthetic', 'synthetic_polar','shakespeare']:
                    raise NotImplementedError("""Non-iid distribution of data for dataset {} is not implemented.
                        Set the distribution to iid.""".format(cfg.data.dataset.type))

        
        if Partitioner is None:
            if return_partitioner:
                data_to_load, Partitioner = build_partitioner_from_config(dataset, cfg , return_partitioner=True)
            else:
                data_to_load = build_partitioner_from_config(dataset, cfg)
            log('Make {} data partitions and use the subdata.'.format(cfg.partitioner.type), cfg.graph.debug)
        else:
            data_to_load = Partitioner.use(cfg.graph.rank)
            log('use the loaded partitioner to load the data.', cfg.graph.debug)
    else:
        if Partitioner is not None:
            raise ValueError('Partitioner is provided but data partition method is not defined!')
        # If test dataset needs to be partitioned this should be changed
        data_to_load = dataset
        log('used whole data.', cfg.graph.debug)

    # Log stats about the dataset to laod
    if dataset_type == 'train':
        cfg.data.train_dataset_size = len(dataset)
        log('  We have {} samples for {}, \
            load {} data for process (rank {}).'.format(
            len(dataset), dataset_type, len(data_to_load), cfg.graph.rank), cfg.graph.debug)
    else:
        cfg.data.val_dataset_size = len(dataset)
        log('  We have {} samples for {}, \
            load {} val data for process (rank {}).'.format(
            len(dataset), dataset_type, len(data_to_load), cfg.graph.rank), cfg.graph.debug)

    # Batching
    if (cfg.partitioner.type=='GrowingBatchPartitioner') and (dataset_type == 'train'):
        batch_sampler = GrowingMinibatchSampler(data_source=data_to_load,
                                                num_epochs=cfg.training.num_epochs,
                                                num_iterations=cfg.training.num_iterations,
                                                base_batch_size=cfg.data.base_batch_size,
                                                max_batch_size=cfg.data.max_batch_size
                                                )
        cfg.training.num_epochs = batch_sampler.num_epochs
        cfg.training.num_iterations = batch_sampler.num_iterations
        cfg.data.total_data_size = len(data_to_load)
        cfg.data.num_samples_per_epoch = len(data_to_load) / cfg.training.num_epochs
        data_loader = torch.utils.data.DataLoader(
            data_to_load, batch_sampler=batch_sampler,
            num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)
        log('we have {} batches for {} for rank {}.'.format(
                len(data_loader), dataset_type, cfg.graph.rank), cfg.graph.debug)
    elif dataset_type == 'train':
        # Adjust stopping criteria
        if cfg.training.stop_criteria == 'epoch':
            cfg.training.num_iterations = int(len(data_to_load) * cfg.training.num_epochs / cfg.training.batch_size)
        else:
            cfg.training.num_epochs = int(cfg.training.num_iterations * cfg.training.batch_size / len(data_to_load))
        cfg.data.total_data_size = len(data_to_load) * cfg.training.num_epochs
        cfg.data.num_samples_per_epoch = len(data_to_load)

        # Generate validation data part
        if getattr(cfg, 'federated', None) is not None and cfg.federated.personal:
            if cfg.data.dataset.type in ['emnist', 'emnist_full', 'shakespeare']:
                data_to_load_train = data_to_load
                data_to_load_val = build_dataset_from_config(cfg, split='val')
            
            if cfg.federated.type == "perfedavg":
                if cfg.data.dataset.type in ['emnist', 'emnist_full', 'shakespeare']:
                    #TODO: make this size a paramter
                    val_size = int(0.1*len(data_to_load))
                    data_to_load_train, data_to_load_val1 = torch.utils.data.random_split(data_to_load,[len(data_to_load) - val_size, val_size])
                else:
                    val_size = int(0.1*len(data_to_load))
                    data_to_load_train, data_to_load_val, data_to_load_val1 = torch.utils.data.random_split(data_to_load,[len(data_to_load) - 3 * val_size, 2 * val_size, val_size])
                data_loader_val1 = torch.utils.data.DataLoader(
                                    data_to_load_val1, batch_size=cfg.training.batch_size,
                                    shuffle=True,
                                    num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                                    drop_last=False)
            else:
                if cfg.data.dataset.type not in ['emnist', 'emnist_full', 'shakespeare']:
                    val_size = int(0.2*len(data_to_load))
                    data_to_load_train, data_to_load_val = torch.utils.data.random_split(data_to_load,[len(data_to_load) - val_size, val_size])
            # Generate data loaders
            data_loader_val = torch.utils.data.DataLoader(
                                    data_to_load_val, batch_size=cfg.training.batch_size,
                                    shuffle=True,
                                    num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                                    drop_last=False)
        
            data_loader_train = torch.utils.data.DataLoader(
                                        data_to_load_train, batch_size=cfg.training.batch_size,
                                        shuffle=True,
                                        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                                        drop_last=False)
            if cfg.federated.type == 'perfedavg':
                data_loader = [data_loader_train, data_loader_val, data_loader_val1]
            else:
                data_loader = [data_loader_train, data_loader_val]
                
            log('we have {} batches for {} for rank {}.'.format(
                len(data_loader[0]), 'train', cfg.graph.rank), cfg.graph.debug)
            log('we have {} batches for {} for rank {}.'.format(
                len(data_loader[1]), 'val', cfg.graph.rank), cfg.graph.debug)
        else:
            data_loader = torch.utils.data.DataLoader(
                                        data_to_load, batch_size=cfg.training.batch_size,
                                        shuffle=True,
                                        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                                        drop_last=False)
            log('we have {} batches for {} for rank {}.'.format(
            len(data_loader), 'train', cfg.graph.rank), cfg.graph.debug)
    else:
        data_loader = torch.utils.data.DataLoader(
            data_to_load, batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
            drop_last=False)
        log('we have {} batches for {} for rank {}.'.format(
            len(data_loader), dataset_type, cfg.graph.rank), cfg.graph.debug)
    if return_partitioner:
        return data_loader, Partitioner
    else:
        return data_loader


def get_data_stat(cfg, train_loader, test_loader=None):
    # get the data statictics (on behalf of each worker) for train.
    # cfg.num_batches_train_per_device_per_epoch = \
    #     len(train_loader) // cfg.graph.n_nodes \
    #     if not cfg.partition_data else len(train_loader)
    cfg.num_batches_train_per_device_per_epoch = len(train_loader)
    cfg.num_whole_train_batches_per_worker = \
        cfg.num_batches_train_per_device_per_epoch * cfg.training.num_epochs
    cfg.num_warmup_train_batches_per_worker = \
        cfg.num_batches_train_per_device_per_epoch * getattr(cfg.lr.scheduler, 'lr_warmup_epochs', 0)
    cfg.num_iterations_per_worker = cfg.training.num_iterations #// cfg.graph.n_nodes

    # get the data statictics (on behalf of each worker) for val.
    if test_loader is not None:
        cfg.num_batches_val_per_device_per_epoch = len(test_loader)
    else:
        cfg.num_batches_val_per_device_per_epoch=0


    # define some parameters for training.
    log('we have {} epochs, \
        {} mini-batches per device for training. \
        {} mini-batches per device for test. \
        The batch size: {}.'.format(
            cfg.training.num_epochs,
            cfg.num_batches_train_per_device_per_epoch,
            cfg.num_batches_val_per_device_per_epoch,
            cfg.training.batch_size), cfg.graph.debug)


class GrowingMinibatchSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, 
                data_source,
                num_epochs=None, 
                num_iterations=None, 
                base_batch_size=2,
                rho=1.01, 
                max_batch_size=0):
        self.data_source = data_source
        self.base_batch_size = base_batch_size
        self.rho = rho
        self.num_samples_per_epoch = len(data_source)
        self.idx_pool = []
        self.max_batch_size = max_batch_size
        if num_epochs is None:
            if num_iterations is None:
                raise ValueError('One of the number of epochs or number of iterations should be provided.')
            self.num_iterations = num_iterations
            self.num_epochs = int(self.base_batch_size * (rho**self.num_iterations - 1) / ((rho - 1) * self.num_samples_per_epoch)) + 1
        else:
            self.num_epochs = num_epochs
            self.num_iterations = int(np.log(self.num_samples_per_epoch*self.num_epochs*(self.rho-1)/self.base_batch_size + 1) / np.log(self.rho))+1
        for _ in range(self.num_epochs):
            self.idx_pool.extend(np.random.permutation(self.num_samples_per_epoch).tolist())
    
        self.batch_size=[int(self.base_batch_size * self.rho**i)+1 for i in range(self.num_iterations)]
        if max_batch_size:
            b = np.array(self.batch_size)
            idx = np.squeeze(np.argwhere(b > max_batch_size))
            if len(idx) >= 1:
                self.batch_size = self.batch_size[:idx[0]] + [max_batch_size] * (np.sum(b[idx]) // max_batch_size)
                if np.sum(b[idx]) // max_batch_size:
                    self.batch_size += [np.sum(b[idx]) % max_batch_size]
                self.num_iterations = len(self.batch_size)
        self.total_num_data = np.sum(self.batch_size)

    def __iter__(self):
        for bs in self.batch_size:
            batch = self.idx_pool[:bs]
            self.idx_pool = self.idx_pool[bs:]
            yield batch

    def __len__(self):
        return self.num_iterations
