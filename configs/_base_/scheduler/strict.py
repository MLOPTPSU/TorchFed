_base_ = "./_main_.py"
lr = dict(
    scheduler=dict(
        type='strict',
        lr_change_epochs='0,60,120,160',
        lr_fields='0.1,0.01,0.001,0.0001',
        lr_scale_indicators='0,0,0,0',
    )
)
