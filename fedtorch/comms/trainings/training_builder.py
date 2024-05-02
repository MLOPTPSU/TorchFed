from .distributed import distributed_training
from .federated import (train_and_validate_federated,
                        train_and_validate_federated_apfl,
                        train_and_validate_federated_drfa,
                        train_and_validate_federated_afl,
                        train_and_validate_federated_centered,
                        train_and_validate_apfl_centered,
                        train_and_validate_drfa_centered,
                        train_and_validate_afl_centered,
                        train_and_validate_perfedme_centered,
                        train_and_validate_sgdap_centered,
                        train_and_validate_fedshuffle_centered,
                        train_and_validate_fedshuffle_one_centered)


# from fedtorch.utils import Registry, build

# TRAINING = Registry("training")

def build_training_from_config(train_cfg, federated_cfg=None):
    if train_cfg.type == 'distributed':
        return distributed_training
    elif train_cfg.type == 'federated':
        if train_cfg.centered:
            if train_cfg.drfa:
                return train_and_validate_drfa_centered
            else:
                if federated_cfg.type == 'apfl':
                    return train_and_validate_apfl_centered
                elif federated_cfg.type == 'perfedme':
                    return train_and_validate_perfedme_centered
                elif federated_cfg.type == 'afl':
                    return train_and_validate_afl_centered
                elif federated_cfg.type in ['fedavg','scaffold','fedgate','qsparse','fedprox','qffl','perfedavg']:
                    return train_and_validate_federated_centered
                elif federated_cfg.type == 'sgdap':
                    return train_and_validate_sgdap_centered
                elif federated_cfg.type == 'fedshuffle':
                    return train_and_validate_fedshuffle_centered
                elif federated_cfg.type == 'fedshuffleone':
                    return train_and_validate_fedshuffle_one_centered
                else:
                    raise NotImplementedError
        else:
            if train_cfg.drfa:
                return train_and_validate_federated_drfa
            else:
                if federated_cfg.type == 'apfl':
                    return train_and_validate_federated_apfl
                elif federated_cfg.type =='afl':
                    return train_and_validate_federated_afl
                elif federated_cfg.type in ['fedavg','scaffold','fedgate','qsparse','fedprox']:
                    return train_and_validate_federated
                else:
                    raise NotImplementedError
            
