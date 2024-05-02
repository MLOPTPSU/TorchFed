# -*- coding: utf-8 -*-
import gc
import shutil
import time
from os.path import join, isfile

import torch

from fedtorch.utils.op_paths import build_dirs, remove_folder


def get_checkpoint_folder_name(cfg):
    # return datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    time_id = str(int(time.time()))
    if cfg.partiontioner.type == 'GrowingBatchPartitioner':
        mode = 'growing_batch_size' 
    elif cfg.partiontioner.type == 'FederatedPartitioner':
        mode = 'federated'
    else:
        mode = 'distributed'
    
    if getattr(cfg, 'federated', False):
        time_id += '_l2-{}_lr-{}_num_comms-{}_num_epochs-{}_batchsize-{}_blocksize-{}_localstep-{}_mode-{}_{}_clients_rate-{}'.format(
            cfg.optimizer.weight_decay,
            cfg.lr.lr,
            cfg.federated.num_comms,
            cfg.training.num_epochs_per_comm,
            cfg.training.batch_size,
            cfg.graph.blocks,
            cfg.training.local_step,
            mode,
            cfg.federated.federated_type,
            cfg.federated.online_client_rate
        )
    else:
        time_id += '_l2-{}_lr-{}_epochs-{}_batchsize-{}_blocksize-{}_localstep-{}_mode-{}'.format(
            cfg.optimizer.weight_decay,
            cfg.lr.lr,
            cfg.training.num_epochs,
            cfg.training.batch_size,
            cfg.graph.blocks,
            cfg.training.local_step,
            mode
        )
    return time_id


def init_checkpoint(cfg):
    # init checkpoint dir.
    # cfg.checkpoint.root = join(
    #     cfg.work_dir, cfg.data.dataset.type, cfg.model.type, get_checkpoint_folder_name(cfg))
    cfg.checkpoint.root = cfg.work_dir
    cfg.checkpoint.checkpoint_dir = join(cfg.checkpoint.root, str(cfg.graph.rank))
    cfg.checkpoint.save_some_models = cfg.checkpoint.save_some_models.split(',')

    # if the directory does not exists, create them.
    if cfg.graph.debug:
        build_dirs(cfg.checkpoint.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def save_to_checkpoint(state, is_best, dirname, filename, save_all=False):
    # save full state.
    cfg = state['arguments']
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(checkpoint_path, join(
            dirname,
            'checkpoint_epoch_%s.pth.tar' % state['current_epoch']))
    elif str(state['current_epoch']) in cfg.checkpoint.save_some_models:
        shutil.copyfile(checkpoint_path, join(
            dirname,
            'checkpoint_epoch_%s.pth.tar' % state['current_epoch']))


def check_resume_status(cfg, old_cfg):
    signal = (cfg.data.dataset.type == old_cfg.data.dataset.type) and \
        (cfg.training.batch_size == old_cfg.training.batch_size) and \
        (cfg.training.num_epochs >= old_cfg.training.num_epochs)
    print('the status of previous resume: {}'.format(signal))
    return signal


def maybe_resume_from_checkpoint(cfg, model, optimizer):
    if cfg.checkpoint.resume:
        if cfg.checkpoint.checkpoint_index is not None:
            # reload model from a specific checkpoint index.
            checkpoint_index = '_epoch_' + cfg.checkpoint.checkpoint_index
        else:
            # reload model from the latest checkpoint.
            checkpoint_index = ''
        checkpoint_path = join(
            cfg.checkpoint.resume, 'checkpoint{}.pth.tar'.format(checkpoint_index))
        print('try to load previous model from the path:{}'.format(
              checkpoint_path))

        if isfile(checkpoint_path):
            print("=> loading checkpoint {} for {}".format(
                cfg.checkpoint.resume, cfg.graph.rank))

            # get checkpoint.
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if not check_resume_status(cfg, checkpoint['arguments']):
                print('=> the checkpoint is not correct. skip.')
            else:
                # restore some run-time info.
                cfg.local_index = checkpoint['local_index']
                cfg.best_prec1 = checkpoint['best_prec1']
                cfg.best_epoch = checkpoint['arguments'].best_epoch

                # reset path for log.
                # remove_folder(cfg.checkpoint.root)
                cfg.checkpoint.root = cfg.checkpoint.resume
                cfg.checkpoint.checkpoint_dir = join(cfg.checkpoint.resume, str(cfg.graph.rank))
                # restore model.
                model.load_state_dict(checkpoint['state_dict'])
                # restore optimizer.
                optimizer.load_state_dict(checkpoint['optimizer'])
                # logging.
                print("=> loaded model from path '{}' checkpointed at (epoch {})"
                      .format(cfg.checkpoint.resume, checkpoint['current_epoch']))

                # try to solve memory issue.
                del checkpoint
                torch.cuda.empty_cache()
                gc.collect()
                return
        else:
            print("=> no checkpoint found at '{}'".format(cfg.checkpoint.resume))
