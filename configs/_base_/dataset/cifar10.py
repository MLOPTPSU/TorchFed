_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='cifar10',
        root="./data/cifar10", 
        transform=None, 
        target_transform=None, 
        download=False,
        num_classes=10,
        dimension=[3,32,32],
    )
)