import argparse
import os
import os.path as osp
from time import time

from fedtorch.utils import Config, DictAction
from fedtorch.nodes import build_nodes_from_config
from fedtorch.comms.trainings import build_training_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Running a federated/distributed training using FedTorch')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='arguments in dict')
    args = parser.parse_args()
    return args


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            './work_dirs',
            osp.splitext(osp.basename(args.config))[0] + "_" + str(int(time())))

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    if cfg.training.type == 'federated':
        if cfg.training.centered:
            cfg.device.is_distributed = False
            cfg.partitioner.distributed = False
        cfg.training.num_epochs = int(cfg.training.num_epochs_per_comm * cfg.federated.num_comms * cfg.federated.online_client_rate)
        if cfg.federated.type == 'afl':
            cfg.federated.sync_type = 'local_step'
            cfg.training.local_step = 1
        if cfg.federated.type == 'qsparse':
            cfg.compressed = True
        if cfg.federated.quantized and cfg.federated.compressed:
            raise ValueError("Quantization is mutually exclusive with compression! Choose only one of them.")

    nodes = build_nodes_from_config(cfg)
    training_func = build_training_from_config(cfg.training, getattr(cfg,'federated',None))
    training_func(*nodes)   


if __name__ == '__main__':
    main()
  
