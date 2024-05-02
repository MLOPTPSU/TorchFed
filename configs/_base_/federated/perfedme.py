_base_ = "./_main_.py"
federated = dict(
    type='perfedme',
    perfedme_lambda=0.001, # The lambda parameter in PerFedMe algorithm. See paper: https://arxiv.org/pdf/2002.07948.pdf
    personal_step=5, # The number of local steps before updating the personal model
)