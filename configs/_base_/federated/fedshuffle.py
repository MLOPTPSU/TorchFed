_base_ = "./_main_.py"
federated = dict(
    type='fedshuffle',
    num_comms_warmup=10,
    personal=True,
)