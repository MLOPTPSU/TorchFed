_base_ = "./_main_.py"
data = dict(
    dataset=dict(
        type='mnist',
        root="./data/mnist", 
        transform=None, 
        target_transform=None, 
        download=False,
        num_classes=10,
        dimension=[28,28],
        )
)