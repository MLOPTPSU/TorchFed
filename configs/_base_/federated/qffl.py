_base_ = "./_main_.py"
federated = dict(
    type='qffl',
    qffl_q=0.5, # The q parameter in QFFL algorithm. See paper: https://arxiv.org/pdf/1905.10497.pdf
)