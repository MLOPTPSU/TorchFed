_base_ = "./_main_.py"
federated = dict(
    type='fedadam',
    fedadam_beta=0.9, #The beta vaiabale for FedAdam training. See paper: https://arxiv.org/pdf/2003.00295.pdf
    fedadam_tau=0.1,  #The tau vaiabale for FedAdam training.
)