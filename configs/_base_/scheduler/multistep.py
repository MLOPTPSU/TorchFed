_base_ = "./_main_.py"
lr = dict(
    scheduler=dict(
        type='strict',
        lr_change_epochs='0,60,120,160',
        lr_warmup=False, 
        lr_decay=10, 
        lr_warmup_epochs=5,
        init_warmup_lr=0.1
    )
)
