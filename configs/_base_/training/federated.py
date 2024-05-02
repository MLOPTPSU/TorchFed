_base_ = "./_main_.py"
training=dict(
    type='federated',
    drfa=False,
    drfa_gamma=0.1, # Setting the gamma value for DRFA algorithm.
    centered=False,
    num_epochs_per_comm=1, # In case the cfg.federated.sync_type is set to 'epoch', this parameter will be used to determine the number of epochs per communication round.
)

"""
The drfa ndicator for using DRFA algorithm for training. 
Paper: https://papers.nips.cc/paper/2020/hash/ac450d10e166657ec8f93a1b65ca1b14-Abstract.html"
"""