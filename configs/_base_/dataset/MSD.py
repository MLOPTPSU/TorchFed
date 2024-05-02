_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='MSD',
        root="./data/MSD",
        num_classes=1,
        dimension=[90],
        )
)