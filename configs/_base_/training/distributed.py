_base_ = "./_main_.py"
training=dict(
    type='distributed',
    reshuffle_per_epoch=False,
    avg_model=True,
)