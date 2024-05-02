_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='epsilon',
        root="./data/epsilon",
        num_classes=2,
        dimension=[2000],
        )
)