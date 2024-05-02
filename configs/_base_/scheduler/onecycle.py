_base_ = "./_main_.py"
lr = dict(
    scheduler=dict(
        type='onecycle',
        lr_onecycle_low=0.15, 
        lr_onecycle_high=3, 
        lr_onecycle_extra_low=0.0015,
        lr_onecycle_num_epoch=46,
    )
)
