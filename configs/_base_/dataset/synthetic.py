_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='synthetic',
        root="./data/synthetic", 
        alpha=0.0, 
        beta=0.0, 
        regression=False, 
        num_dim=60,
        num_classes=2,
        dimension=[60],
        min_num=500,
        max_num=1000,
        )
)