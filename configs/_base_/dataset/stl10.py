_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='stl10',
        root="./data/stl10", 
        transform=None, 
        target_transform=None, 
        download=False,
        num_classes=10,
        dimension=[3,96,96],
    )
)