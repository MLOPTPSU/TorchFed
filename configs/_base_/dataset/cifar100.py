_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='cifar100',
        root="./data/cifar100", 
        transform=None, 
        target_transform=None, 
        download=False,
        num_classes=100,
        dimension=[3,32,32],
        )
)