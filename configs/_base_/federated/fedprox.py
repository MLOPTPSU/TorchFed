_base_ = "./_main_.py"
federated = dict(
    type='fedprox',
    fedprox_mu=0.002, # The Mu parameter in the FedProx algorithm. See paper: https://arxiv.org/pdf/1812.06127.pdf
)