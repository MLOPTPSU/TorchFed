from copy import deepcopy

from .nodes import Client
from .nodes_centered import ClientCentered, ServerCentered


def build_nodes_from_config(cfg):
    if not getattr(cfg.training, 'centered', False):
        """distributed training via mpi backend."""
        import torch.distributed as dist
        dist.init_process_group('mpi')
        cfg.device.blocks, cfg.device.world = cal_blocks_and_world(cfg.device.num_clients, cfg.device.num_nodes)
        client = Client(cfg, dist.get_rank())
        # Initialize the node
        client.initialize()
        # Initialize the dataset if not downloaded
        client.initialize_dataset()
        # Load the dataset
        client.load_local_dataset()
        # Generate auxiliary models and params for training
        client.gen_aux_models()
        return (client,)
    else:
        """Centered training, simulating federated learning"""
        ClientNodes ={}
        cfg.device.blocks, cfg.device.world = cal_blocks_and_world(cfg.device.num_clients, 1)
        for i in range(cfg.device.num_clients):
            if cfg.data.dataset.type in ['emnist', 'emnist_full','synthetic', 'synthetic_polar'] or i==0:
                ClientNodes[i] = ClientCentered(cfg,i)
            else:
                ClientNodes[i] = ClientCentered(cfg, i, Partitioner=ClientNodes[0].Partitioner)
        ServerNode = ServerCentered(deepcopy(ClientNodes[0].cfg), deepcopy(ClientNodes[0].model)) 
        ServerNode.enable_grad(ClientNodes[0].train_loader)
        return (ClientNodes, ServerNode)

def cal_blocks_and_world(num_clients, num_nodes=1):
    num_clients_per_worker =  num_clients // num_nodes
    residual = num_clients % num_nodes
    num_clients_nodes = [num_clients_per_worker] * num_nodes
    num_clients_nodes[-1] += residual

    blocks=(',').join([str(i) for i in num_clients_nodes])
    world = ",".join([ ",".join([str(x) for x in range(i)]) for i in num_clients_nodes])
    return blocks, world