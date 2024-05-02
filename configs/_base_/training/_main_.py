training=dict(
    batch_size=128,
    local_step=1,
    num_epochs=None,
    num_iterations=None,
    stop_criteria='epoch',
    evaluate=True, # Evaluate model on validation set
    eval_freq=1,
    local_step_warmup_type=None, # Choose from 'linear', 'exp', 'constant', and None. default is None.
    local_step_warmup_period=None,
    turn_on_local_step_from=None,
    turn_off_local_step_from=None,
    local_step_warmup_per_interval=False,
    manual_seed=7,
    summary_freq=10,
    out_momentum=False,
    in_momentum=False,
)