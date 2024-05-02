_base_ = "./_main_.py"
lr = dict(
    scheduler=dict(
        type='convex_decay',
        lr_gamma=0.1,
        lr_mu=1.0,
        lr_alpha=1.0
    )
)
